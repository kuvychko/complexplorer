"""
STL export utilities for creating 3D-printable ornaments from complex functions.

This module provides tools to convert Riemann sphere visualizations into
complete watertight STL files suitable for 3D printing as decorative ornaments.

Users can orient and slice meshes however they want in their slicer software.
"""

from .ornament_generator import OrnamentGenerator

__all__ = ['OrnamentGenerator']