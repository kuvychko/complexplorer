"""STL export for 3D printing complex function visualizations.

This module provides tools to convert Riemann sphere visualizations
into watertight STL files suitable for 3D printing as decorative ornaments.
"""

from .mesh_repair import close_mesh_holes, ensure_consistent_normals, repair_mesh_simple
from .ornament_generator import OrnamentGenerator, create_ornament
from .utils import center_mesh, scale_to_size, validate_printability

__all__ = [
    "OrnamentGenerator",
    "create_ornament",
    "validate_printability",
    "scale_to_size",
    "center_mesh",
    "repair_mesh_simple",
    "close_mesh_holes",
    "ensure_consistent_normals",
]
