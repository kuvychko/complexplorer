"""
STL-specific utility functions.

This module imports mesh generation utilities from the main module.
All cutting/bisection functionality has been removed - users should
cut meshes in their slicer software.
"""

# Import mesh utilities from the main module
from ..mesh_utils import (
    RectangularSphereGenerator,
    stereographic_projection,
    ModulusScaling
)

__all__ = ['RectangularSphereGenerator', 'stereographic_projection', 'ModulusScaling']