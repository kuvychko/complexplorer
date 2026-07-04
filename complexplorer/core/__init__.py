"""Core functionality for complexplorer.

This module contains the fundamental building blocks of the library:
- Domain classes for defining regions in the complex plane
- Colormap classes for mapping complex values to colors
- Mathematical functions and utilities
- Modulus scaling for visualization
"""

from .colormap import (
    OUT_OF_DOMAIN_COLOR_HSV,
    Chessboard,
    Colormap,
    LogRings,
    Phase,
    PolarChessboard,
)
from .domain import Annulus, CompositeDomain, Disk, Domain, Rectangle
from .functions import (
    inverse_stereographic,
    phase,
    sawtooth,
    sawtooth_log,
    stereographic_projection,
)
from .scaling import SCALING_PRESETS, ModulusScaling, apply_scaling_mode, get_scaling_preset

__all__ = [
    # Scaling
    "ModulusScaling",
    "SCALING_PRESETS",
    "apply_scaling_mode",
    "get_scaling_preset",
    # Domains
    "Domain",
    "Rectangle",
    "Disk",
    "Annulus",
    "CompositeDomain",
    # Colormaps
    "Colormap",
    "Phase",
    "Chessboard",
    "PolarChessboard",
    "LogRings",
    "OUT_OF_DOMAIN_COLOR_HSV",
    # Functions
    "phase",
    "sawtooth",
    "sawtooth_log",
    "stereographic_projection",
    "inverse_stereographic",
]
