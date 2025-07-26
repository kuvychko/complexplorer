"""Simple mesh repair utilities for STL export.

This module provides lightweight mesh repair functions to ensure
watertight meshes suitable for 3D printing.
"""

import numpy as np
from typing import Optional

from .utils import check_pyvista_available

# Import PyVista if available
try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    pv = None


def close_mesh_holes(mesh: 'pv.PolyData', 
                    verbose: bool = False) -> 'pv.PolyData':
    """Attempt to close holes in mesh.
    
    Note: The rectangular Riemann sphere parameterization naturally
    has small gaps at the poles that are difficult to fill perfectly.
    These small gaps are typically not an issue for 3D printing.
    
    Parameters
    ----------
    mesh : pv.PolyData
        Input mesh with holes.
    verbose : bool, default=False
        Print progress information.
        
    Returns
    -------
    pv.PolyData
        Mesh with attempted hole filling.
    """
    check_pyvista_available()
    
    if verbose:
        # Check initial state
        edges = mesh.extract_feature_edges(boundary_edges=True)
        n_boundary = edges.n_points
        print(f"Initial boundary edges: {n_boundary}")
    
    # Try to fill holes
    try:
        filled = mesh.fill_holes(hole_size=1000.0)
    except Exception as e:
        if verbose:
            print(f"Standard hole filling failed: {e}")
            print("Mesh has small polar gaps typical of Riemann sphere parameterization")
        filled = mesh
    
    if verbose and filled is not mesh:
        # Check result
        edges_after = filled.extract_feature_edges(boundary_edges=True)
        n_boundary_after = edges_after.n_points
        print(f"After filling: {n_boundary_after} boundary edges")
        if n_boundary_after < n_boundary:
            print(f"Reduced boundary edges by {n_boundary - n_boundary_after}")
    
    return filled


def repair_mesh_simple(mesh: 'pv.PolyData',
                      fill_holes: bool = True,
                      clean: bool = True,
                      smooth: bool = False,
                      smooth_iterations: int = 10,
                      verbose: bool = False) -> 'pv.PolyData':
    """Simple mesh repair for 3D printing.
    
    Performs basic repairs to ensure mesh is suitable for STL export.
    
    Parameters
    ----------
    mesh : pv.PolyData
        Input mesh to repair.
    fill_holes : bool, default=True
        Fill holes to make watertight.
    clean : bool, default=True
        Remove duplicate points and degenerate cells.
    smooth : bool, default=False
        Apply smoothing.
    smooth_iterations : int, default=10
        Number of smoothing iterations.
    verbose : bool, default=False
        Print progress.
        
    Returns
    -------
    pv.PolyData
        Repaired mesh.
    """
    check_pyvista_available()
    
    if verbose:
        print("=== Simple Mesh Repair ===")
        print(f"Input: {mesh.n_points} points, {mesh.n_cells} faces")
    
    repaired = mesh.copy()
    
    # Clean first to remove duplicates
    if clean:
        if verbose:
            print("Cleaning mesh...")
        repaired = repaired.clean(tolerance=1e-9)
        if verbose:
            print(f"  After clean: {repaired.n_points} points, {repaired.n_cells} faces")
    
    # Fill holes
    if fill_holes:
        if verbose:
            print("Filling holes...")
        repaired = close_mesh_holes(repaired, verbose=verbose)
    
    # Smooth if requested
    if smooth and smooth_iterations > 0:
        if verbose:
            print(f"Smoothing ({smooth_iterations} iterations)...")
        repaired = repaired.smooth(n_iter=smooth_iterations, 
                                  relaxation_factor=0.1,
                                  feature_smoothing=False,
                                  boundary_smoothing=True)
    
    # Final clean
    if clean:
        repaired = repaired.clean(tolerance=1e-9)
    
    if verbose:
        print(f"Final: {repaired.n_points} points, {repaired.n_cells} faces")
        
        # Check if watertight
        edges = repaired.extract_feature_edges(boundary_edges=True)
        if edges.n_points == 0:
            print("✓ Mesh is watertight")
        else:
            print(f"✗ Mesh still has {edges.n_points} boundary points")
    
    return repaired


def ensure_consistent_normals(mesh: 'pv.PolyData', 
                            verbose: bool = False) -> 'pv.PolyData':
    """Ensure all face normals point consistently outward.
    
    Parameters
    ----------
    mesh : pv.PolyData
        Input mesh.
    verbose : bool, default=False
        Print progress.
        
    Returns
    -------
    pv.PolyData
        Mesh with consistent normals.
    """
    check_pyvista_available()
    
    if verbose:
        print("Ensuring consistent normals...")
    
    # Compute normals
    mesh_with_normals = mesh.compute_normals(
        cell_normals=True,
        point_normals=False,
        consistency=True,
        auto_orient_normals=True
    )
    
    return mesh_with_normals


__all__ = ['close_mesh_holes', 'repair_mesh_simple', 'ensure_consistent_normals']