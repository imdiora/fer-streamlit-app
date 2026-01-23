import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.api import create_app


class DummyPredictor:
    def __init__(self):
        self.device = "cpu"

    def predict(self, image_bgr):
        return [
            {
                "box": [0, 0, 10, 10],
                "emotion": "Happy",
                "confidence": 0.99,
                "class_id": 3,
                "inference_ms": 1.0,
            }
        ]

    def annotate_image(self, image_bgr, detections):
        return image_bgr


def test_health_endpoint():
    app = create_app(predictor=DummyPredictor())
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_endpoint_returns_json():
    app = create_app(predictor=DummyPredictor())
    client = TestClient(app)

    img = np.zeros((64, 64, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok

    files = {"file": ("test.jpg", bytes(buf), "image/jpeg")}
    r = client.post("/predict", files=files)
    assert r.status_code == 200
    data = r.json()
    assert "detections" in data
    assert data["count"] == 1
    assert data["detections"][0]["emotion"] == "Happy"
