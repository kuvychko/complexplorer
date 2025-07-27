"""Core functionality for complexplorer.

This module contains the fundamental building blocks of the library:
- Domain classes for defining regions in the complex plane
- Colormap classes for mapping complex values to colors
- Mathematical functions and utilities
- Modulus scaling for visualization
"""

from .scaling import ModulusScaling, SCALING_PRESETS, get_scaling_preset
from .domain import Domain, Rectangle, Disk, Annulus, CompositeDomain
from .colormap import (
    Colormap, Phase, Chessboard, PolarChessboard, LogRings,
    OUT_OF_DOMAIN_COLOR_HSV
)
from .functions import (
    phase, sawtooth, sawtooth_log,
    stereographic_projection, inverse_stereographic, stereographic
)

__all__ = [
    # Scaling
    'ModulusScaling',
    'SCALING_PRESETS',
    'get_scaling_preset',
    # Domains
    'Domain',
    'Rectangle', 
    'Disk',
    'Annulus',
    'CompositeDomain',
    # Colormaps
    'Colormap',
    'Phase',
    'Chessboard',
    'PolarChessboard',
    'LogRings',
    'OUT_OF_DOMAIN_COLOR_HSV',
    # Functions
    'phase',
    'sawtooth',
    'sawtooth_log',
    'stereographic_projection',
    'inverse_stereographic',
    'stereographic',
]