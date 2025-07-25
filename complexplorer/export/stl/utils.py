"""Utility functions for STL export.

This module provides validation and helper functions for 3D printing.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import warnings

# Import PyVista if available
try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    pv = None


def check_pyvista_available():
    """Check if PyVista is available and raise error if not."""
    if not HAS_PYVISTA:
        raise ImportError(
            "PyVista is required for STL export functionality. "
            "Install with: pip install pyvista"
        )


def validate_printability(mesh: 'pv.PolyData', 
                         size_mm: Optional[float] = None,
                         verbose: bool = True) -> Dict[str, any]:
    """Validate mesh for 3D printing requirements.
    
    Parameters
    ----------
    mesh : pv.PolyData
        Mesh to validate.
    size_mm : float, optional
        Target size in millimeters for scaling validation.
    verbose : bool, optional
        Print validation results.
        
    Returns
    -------
    dict
        Validation results with the following keys:
        - is_watertight: bool, whether mesh is closed
        - is_manifold: bool, whether mesh is manifold
        - n_boundary_edges: int, number of boundary edges
        - n_non_manifold_edges: int, number of non-manifold edges
        - bounds: tuple, mesh bounding box
        - dimensions: tuple, mesh dimensions (x, y, z)
        - volume: float, mesh volume
        - surface_area: float, mesh surface area
        - wall_thickness_ok: bool, if size_mm provided
        - recommended_size_mm: float, recommended print size
    """
    check_pyvista_available()
    
    results = {}
    
    # Check if watertight (no boundary edges)
    edges = mesh.extract_feature_edges(boundary_edges=True)
    results['is_watertight'] = edges.n_points == 0
    results['n_boundary_edges'] = edges.n_cells
    
    # Check if manifold (no non-manifold edges)
    nm_edges = mesh.extract_feature_edges(non_manifold_edges=True)
    results['is_manifold'] = nm_edges.n_points == 0
    results['n_non_manifold_edges'] = nm_edges.n_cells
    
    # Get mesh properties
    results['bounds'] = mesh.bounds
    results['dimensions'] = (
        mesh.bounds[1] - mesh.bounds[0],  # x
        mesh.bounds[3] - mesh.bounds[2],  # y
        mesh.bounds[5] - mesh.bounds[4]   # z
    )
    
    # Calculate volume and surface area
    try:
        results['volume'] = mesh.volume
        results['surface_area'] = mesh.area
    except:
        results['volume'] = None
        results['surface_area'] = None
        if verbose:
            warnings.warn("Could not compute volume/area. Mesh may not be watertight.")
    
    # Scaling validation if size provided
    if size_mm is not None:
        max_dim = max(results['dimensions'])
        scale_factor = size_mm / max_dim
        
        # Estimate minimum wall thickness (based on modulus scaling)
        # This is approximate - actual thickness depends on function
        min_thickness_mm = 0.3 * scale_factor  # 30% of radius at thinnest
        
        # Common 3D printing minimum wall thickness
        MIN_WALL_THICKNESS_MM = 0.8
        
        results['wall_thickness_ok'] = min_thickness_mm >= MIN_WALL_THICKNESS_MM
        results['estimated_min_wall_mm'] = min_thickness_mm
        
        # Recommend size if too small
        if not results['wall_thickness_ok']:
            results['recommended_size_mm'] = size_mm * (MIN_WALL_THICKNESS_MM / min_thickness_mm)
        else:
            results['recommended_size_mm'] = size_mm
    
    # Print results if verbose
    if verbose:
        print("=== Mesh Validation Results ===")
        print(f"Watertight: {results['is_watertight']} ({results['n_boundary_edges']} boundary edges)")
        print(f"Manifold: {results['is_manifold']} ({results['n_non_manifold_edges']} non-manifold edges)")
        print(f"Dimensions: {results['dimensions'][0]:.3f} x {results['dimensions'][1]:.3f} x {results['dimensions'][2]:.3f}")
        
        if results['volume'] is not None:
            print(f"Volume: {results['volume']:.3f}")
            print(f"Surface area: {results['surface_area']:.3f}")
        
        if size_mm is not None:
            print(f"\nAt {size_mm}mm size:")
            print(f"Estimated min wall thickness: {results['estimated_min_wall_mm']:.2f}mm")
            if results['wall_thickness_ok']:
                print("✓ Wall thickness OK for printing")
            else:
                print(f"✗ Too thin! Recommend at least {results['recommended_size_mm']:.1f}mm")
        
        # Overall assessment
        print("\n=== Overall Assessment ===")
        if results['is_watertight'] and results['is_manifold']:
            if 'wall_thickness_ok' in results and results['wall_thickness_ok']:
                print("✓ Mesh is ready for 3D printing!")
            else:
                print("✓ Mesh topology OK, but check wall thickness")
        elif results['n_boundary_edges'] < 200:  # Small number of boundary edges
            print("⚠ Mesh has small gaps (typical for Riemann sphere)")
            print("  These are usually acceptable for 3D printing")
            if 'wall_thickness_ok' in results and results['wall_thickness_ok']:
                print("  Wall thickness is OK - should print successfully")
        else:
            print("✗ Mesh needs significant repair before printing")
    
    return results


def scale_to_size(mesh: 'pv.PolyData', 
                  target_size_mm: float,
                  axis: str = 'max') -> 'pv.PolyData':
    """Scale mesh to target size in millimeters.
    
    Parameters
    ----------
    mesh : pv.PolyData
        Mesh to scale.
    target_size_mm : float
        Target size in millimeters.
    axis : str, optional
        Which axis to scale to:
        - 'max': Scale so largest dimension equals target_size_mm
        - 'x', 'y', 'z': Scale specific axis to target_size_mm
        
    Returns
    -------
    pv.PolyData
        Scaled mesh.
    """
    check_pyvista_available()
    
    bounds = mesh.bounds
    dimensions = [
        bounds[1] - bounds[0],  # x
        bounds[3] - bounds[2],  # y
        bounds[5] - bounds[4]   # z
    ]
    
    if axis == 'max':
        current_size = max(dimensions)
    elif axis == 'x':
        current_size = dimensions[0]
    elif axis == 'y':
        current_size = dimensions[1]
    elif axis == 'z':
        current_size = dimensions[2]
    else:
        raise ValueError(f"Invalid axis: {axis}")
    
    scale_factor = target_size_mm / current_size
    
    # Create scaled copy
    scaled = mesh.copy()
    scaled.points *= scale_factor
    
    return scaled


def center_mesh(mesh: 'pv.PolyData') -> 'pv.PolyData':
    """Center mesh at origin.
    
    Parameters
    ----------
    mesh : pv.PolyData
        Mesh to center.
        
    Returns
    -------
    pv.PolyData
        Centered mesh.
    """
    check_pyvista_available()
    
    centered = mesh.copy()
    center = centered.center
    centered.points -= center
    
    return centered