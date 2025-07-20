"""
STL export utilities for creating 3D-printable ornaments from complex functions.

This module provides tools to convert Riemann sphere visualizations into
watertight STL files suitable for 3D printing as decorative ornaments.
"""

from .ornament_generator import OrnamentGenerator
from .mesh_healing import MeshHealer

__all__ = ['OrnamentGenerator', 'MeshHealer']