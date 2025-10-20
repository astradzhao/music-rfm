"""
MusicRFM: Representation Finetuning for Music Generation Control

A Python library for fine-grained controllable music generation using
Representation Finetuning Methods (RFM).
"""

from .rfm_controllers import MusicGenController
from .control_toolkits import (
    MusicRFMToolkit,
    MusicLinearProbeToolkit,
    MusicLogisticRegressionToolkit,
)

__version__ = "0.1.0"

__all__ = [
    # Main controller
    "MusicGenController",
    
    # Control toolkits
    "RFMToolkit",
    "MusicRFMToolkit",
    "MusicLinearProbeToolkit",
    "MusicLogisticRegressionToolkit",
]

