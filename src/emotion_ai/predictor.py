from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision import models

from .config import AppConfig, load_config
from .utils import clamp_box

logger = logging.getLogger(__name__)


class EmotionPredictor:
    """High-level predictor used by both the API and the Streamlit UI."""

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        self.config_path = config_path
        self.config: AppConfig = load_config(config_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.labels = dict(self.config.labels)

        self.model: Optional[torch.nn.Module] = None
        self.face_cascade: Optional[cv2.CascadeClassifier] = None
        self.transform: Optional[transforms.Compose] = None

        self._load_model()
        self._init_face_detector()
        self._init_preprocess()

    # ---------------------------
    # Weights (HF auto-download)
    # ---------------------------
    def _maybe_download_weights(self, weights_path: Path) -> None:
        auto = self.config.model.auto_download
        if not auto.enabled:
            return
        if weights_path.exists():
            return
        if not auto.repo_id or not auto.filename:
            return

        logger.info("Weights not found at %s. Attempting download...", weights_path)
        weights_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            from huggingface_hub import hf_hub_download

            downloaded = hf_hub_download(repo_id=auto.repo_id, filename=auto.filename)
            weights_path.write_bytes(Path(downloaded).read_bytes())
            logger.info("Downloaded weights -> %s", weights_path)
        except Exception as e:
            logger.warning("Failed to auto-download weights: %s", e)

    # ---------------------------
    # Model loading (Option A: ResNet50)
    # ---------------------------
    def _extract_state_dict(self, ckpt: Any) -> Tuple[Optional[dict], Optional[list]]:
        """Return (state_dict, class_names) from common checkpoint formats."""
        class_names = None

        if isinstance(ckpt, dict):
            class_names = ckpt.get("class_names")
            sd = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt.get("model") or ckpt.get("net")
            if isinstance(sd, dict):
                return sd, class_names
            # sometimes ckpt itself is a state dict
            if all(isinstance(k, str) for k in ckpt.keys()):
                return ckpt, class_names

        return None, class_names

    def _strip_module_prefix(self, state_dict: dict) -> dict:
        # Strip DataParallel prefix "module."
        if any(k.startswith("module.") for k in state_dict.keys()):
            return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        return state_dict

    def _load_model(self) -> None:
        model_cfg = self.config.model
        weights_path = Path(model_cfg.weights_path)
        self._maybe_download_weights(weights_path)

        arch = getattr(model_cfg, "arch", "ddamfn").lower()
        if arch not in ("resnet50", "resnet"):
            raise RuntimeError(
                f"Config model.arch='{model_cfg.arch}' but Option A requires 'resnet50'. "
                f"Set model.arch: resnet50 and model.input_size: 224."
            )

        # Build ResNet50 classifier
        num_classes = int(model_cfg.num_classes)
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        if not weights_path.exists():
            logger.warning("Weights not found at %s. Using random weights.", weights_path)
            self.model = model.to(self.device).eval()
            return

        ckpt = torch.load(weights_path, map_location="cpu")
        state_dict, class_names = self._extract_state_dict(ckpt)
        if state_dict is None:
            raise RuntimeError("Could not extract a state_dict from checkpoint. (Unsupported .pth format)")

        state_dict = self._strip_module_prefix(state_dict)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("Missing keys when loading ResNet weights (count=%d). Example: %s", len(missing), missing[:10])
        if unexpected:
            logger.warning("Unexpected keys when loading ResNet weights (count=%d). Example: %s", len(unexpected), unexpected[:10])

        # Prefer checkpoint class names if provided (prevents label-order mistakes)
        if isinstance(class_names, (list, tuple)) and len(class_names) == num_classes:
            self.labels = {i: str(name) for i, name in enumerate(class_names)}
            logger.info("Loaded class_names from checkpoint: %s", self.labels)

        self.model = model.to(self.device).eval()
        logger.info("ResNet50 model loaded. device=%s weights=%s", self.device, weights_path)

    # ---------------------------
    # Face detection
    # ---------------------------
    def _init_face_detector(self) -> None:
        fd = self.config.face_detection
        cascade_path = cv2.data.haarcascades + fd.cascade_file
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def detect_faces(self, image_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        if self.face_cascade is None:
            return []

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=float(self.config.face_detection.scale_factor),
            minNeighbors=int(self.config.face_detection.min_neighbors),
            minSize=(int(self.config.face_detection.min_size), int(self.config.face_detection.min_size)),
        )
        out: List[Tuple[int, int, int, int]] = []
        for (x, y, w, h) in faces:
            out.append((int(x), int(y), int(w), int(h)))
        return out

    # ---------------------------
    # Preprocess
    # ---------------------------
    def _init_preprocess(self) -> None:
        size = int(self.config.model.input_size)

        # ResNet uses ImageNet normalization
        self.transform = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    # ---------------------------
    # Predict
    # ---------------------------
    def predict(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        if self.model is None or self.transform is None:
            return []

        faces = self.detect_faces(image_bgr)
        results: List[Dict[str, Any]] = []
        if len(faces) == 0:
            return results

        with torch.no_grad():
            for (x, y, w, h) in faces:
                x, y, w, h = clamp_box(x, y, w, h, image_bgr.shape)
                face = image_bgr[y : y + h, x : x + w]
                if face.size == 0:
                    continue

                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                inp = self.transform(Image.fromarray(face_rgb)).unsqueeze(0).to(self.device)

                t0 = time.perf_counter()
                logits = self.model(inp)
                probs = torch.softmax(logits, dim=1)[0]
                inference_ms = (time.perf_counter() - t0) * 1000.0

                class_id = int(torch.argmax(probs).item())
                confidence = float(probs[class_id].item())
                emotion = self.labels.get(class_id, str(class_id))

                results.append(
                    {
                        "box": [int(x), int(y), int(w), int(h)],
                        "emotion": emotion,
                        "confidence": round(confidence, 4),
                        "class_id": class_id,
                        "inference_ms": round(float(inference_ms), 3),
                    }
                )

        return results

    def annotate_image(self, image_bgr: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        annotated = image_bgr.copy()
        for det in detections:
            x, y, w, h = det["box"]
            label = f"{det['emotion']} ({det['confidence']:.2f})"
            color = (0, 255, 0)

            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            cv2.putText(annotated, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return annotated
