import pytest

from emotion_ai.predictor import EmotionPredictor


@pytest.fixture(scope="session")
def predictor() -> EmotionPredictor:
    return EmotionPredictor(config_path="config/config.yaml")
