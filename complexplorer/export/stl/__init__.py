"""STL export for 3D printing complex function visualizations.

This module provides tools to convert Riemann sphere visualizations
into watertight STL files suitable for 3D printing as decorative ornaments.
"""

from .ornament_generator import OrnamentGenerator, create_ornament
from .utils import validate_printability, scale_to_size, center_mesh
from .mesh_repair import repair_mesh_simple, close_mesh_holes, ensure_consistent_normals

__all__ = [
    'OrnamentGenerator', 
    'create_ornament',
    'validate_printability',
    'scale_to_size',
    'center_mesh',
    'repair_mesh_simple',
    'close_mesh_holes',
    'ensure_consistent_normals'
]