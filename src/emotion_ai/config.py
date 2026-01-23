from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml


@dataclass(frozen=True)
class AutoDownloadConfig:
    enabled: bool = False
    repo_id: str = ""
    filename: str = ""


@dataclass(frozen=True)
class ModelConfig:
    arch: str = "ddamfn"
    num_classes: int = 7
    num_heads: int = 3
    input_size: int = 112
    weights_path: str = "models/model.pth"
    backbone_pretrained_path: str = "pretrained/MFN_msceleb.pth"
    auto_download: AutoDownloadConfig = AutoDownloadConfig()


@dataclass(frozen=True)
class FaceDetectionConfig:
    method: str = "haar"
    cascade_file: str = "haarcascade_frontalface_default.xml"
    scale_factor: float = 1.1
    min_neighbors: int = 5
    min_size: int = 30


@dataclass(frozen=True)
class ApiConfig:
    host: str = "0.0.0.0"
    port: int = 8000


@dataclass(frozen=True)
class UiConfig:
    host: str = "0.0.0.0"
    port: int = 8501
    default_confidence_threshold: float = 0.4


@dataclass(frozen=True)
class AppConfig:
    model: ModelConfig
    face_detection: FaceDetectionConfig
    labels: Dict[int, str]
    api: ApiConfig
    ui: UiConfig


def _coerce_int_keys(d: Mapping[Any, Any]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for k, v in d.items():
        try:
            out[int(k)] = str(v)
        except Exception:
            # best effort
            continue
    return out


def load_config(config_path: str | Path) -> AppConfig:
    """Load YAML config into typed dataclasses.

    Raises:
        FileNotFoundError: if config does not exist.
        ValueError: if required fields are missing.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    data: Dict[str, Any]
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    model_raw = data.get("model", {}) or {}
    auto_raw = model_raw.get("auto_download", {}) or {}
    model = ModelConfig(
        arch=str(model_raw.get("arch", "ddamfn")),
        num_classes=int(model_raw.get("num_classes", 7)),
        num_heads=int(model_raw.get("num_heads", 3)),
        input_size=int(model_raw.get("input_size", 112)),
        weights_path=str(model_raw.get("weights_path", "models/model.pth")),
        backbone_pretrained_path=str(
            model_raw.get("backbone_pretrained_path", "pretrained/MFN_msceleb.pth")
        ),
        auto_download=AutoDownloadConfig(
            enabled=bool(auto_raw.get("enabled", False)),
            repo_id=str(auto_raw.get("repo_id", "")),
            filename=str(auto_raw.get("filename", "")),
        ),
    )

    fd_raw = data.get("face_detection", {}) or {}
    face_detection = FaceDetectionConfig(
        method=str(fd_raw.get("method", "haar")),
        cascade_file=str(fd_raw.get("cascade_file", "haarcascade_frontalface_default.xml")),
        scale_factor=float(fd_raw.get("scale_factor", 1.1)),
        min_neighbors=int(fd_raw.get("min_neighbors", 5)),
        min_size=int(fd_raw.get("min_size", 30)),
    )

    labels_raw = data.get("labels", {}) or {}
    labels = _coerce_int_keys(labels_raw)
    if not labels:
        raise ValueError("Config must include a non-empty 'labels' mapping")

    api_raw = data.get("api", {}) or {}
    api = ApiConfig(
        host=str(api_raw.get("host", "0.0.0.0")),
        port=int(api_raw.get("port", 8000)),
    )

    ui_raw = data.get("ui", {}) or {}
    ui = UiConfig(
        host=str(ui_raw.get("host", "0.0.0.0")),
        port=int(ui_raw.get("port", 8501)),
        default_confidence_threshold=float(ui_raw.get("default_confidence_threshold", 0.4)),
    )

    return AppConfig(model=model, face_detection=face_detection, labels=labels, api=api, ui=ui)
