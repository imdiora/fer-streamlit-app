"""Neural network architectures.

This folder contains a cleaned-up implementation of the DDAMFN-style architecture used
in the original project, made importable under the `emotion_ai` package.

Main export:
- emotion_ai.networks.DDAM.DDAMNet
"""

from .DDAM import DDAMNet

__all__ = ["DDAMNet"]
