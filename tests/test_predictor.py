import numpy as np

from emotion_ai.predictor import EmotionPredictor


def test_predictor_initializes_without_weights(predictor: EmotionPredictor):
    # The repo ships without a checkpoint by default.
    # Predictor should still initialize (random weights) and not crash.
    assert predictor.model is not None


def test_detect_faces_on_blank_image(predictor: EmotionPredictor):
    blank = np.zeros((224, 224, 3), dtype=np.uint8)
    faces = predictor.detect_faces(blank)
    assert len(faces) == 0


def test_predict_returns_list(predictor: EmotionPredictor):
    blank = np.zeros((224, 224, 3), dtype=np.uint8)
    out = predictor.predict(blank)
    assert isinstance(out, list)
