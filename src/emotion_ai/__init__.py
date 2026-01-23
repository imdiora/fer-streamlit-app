"""Emotion AI package.

This repo is structured as a standard `src/` layout so it can be installed with:

    pip install -e .

Main entry points:
- emotion_ai.predictor.EmotionPredictor
- app.api (FastAPI)
- app.ui (Streamlit)
"""

from .version import __version__

__all__ = ["__version__"]
