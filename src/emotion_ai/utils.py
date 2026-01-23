from __future__ import annotations

import base64
from typing import Tuple

import cv2
import numpy as np


def bgr_from_bytes(data: bytes) -> np.ndarray:
    """Decode image bytes into an OpenCV BGR uint8 image."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Unsupported format or corrupted file.")
    return img


def bgr_to_jpeg_bytes(image_bgr: np.ndarray, quality: int = 90) -> bytes:
    """Encode BGR image to JPEG bytes."""
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, buf = cv2.imencode(".jpg", image_bgr, encode_params)
    if not ok:
        raise RuntimeError("Failed to encode image as JPEG")
    return bytes(buf)


def bgr_to_base64_jpeg(image_bgr: np.ndarray, quality: int = 90) -> str:
    """Encode BGR image to base64 jpeg string (no data: prefix)."""
    raw = bgr_to_jpeg_bytes(image_bgr, quality=quality)
    return base64.b64encode(raw).decode("utf-8")


def clamp_box(x: int, y: int, w: int, h: int, img_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
    """Clamp box to image boundaries."""
    H, W = img_shape[0], img_shape[1]
    x = max(0, min(int(x), W - 1))
    y = max(0, min(int(y), H - 1))
    w = max(0, min(int(w), W - x))
    h = max(0, min(int(h), H - y))
    return x, y, w, h
