"""
Spherical boundary-based mesh healing using PyVista's built-in clipping.

This module uses PyVista's clip_scalar method for robust spherical shell clipping.
"""

import numpy as np
import pyvista as pv
from typing import Tuple, Optional, List
# Validation moved inline


def spherical_shell_clip(mesh: pv.PolyData, 
                        r_min: float,
                        r_max: float,
                        verbose: bool = False) -> pv.PolyData:
    """
    Clip mesh to spherical shell using PyVista's clip_scalar.
    
    This method properly handles triangle clipping at boundaries.
    
    Parameters
    ----------
    mesh : pv.PolyData
        Input mesh to clip.
    r_min : float
        Inner radius of shell (use 0 for solid sphere).
    r_max : float
        Outer radius of shell.
    verbose : bool
        Print progress information.
        
    Returns
    -------
    clipped : pv.PolyData
        Mesh clipped to spherical shell.
    """
    if verbose:
        print(f"Clipping to spherical shell: r_min={r_min:.3f}, r_max={r_max:.3f}")
    
    # Calculate radii for all points
    points = mesh.points
    radii = np.linalg.norm(points, axis=1)
    
    # Add radius as point data
    mesh_with_radius = mesh.copy()
    mesh_with_radius["radius"] = radii
    
    if verbose:
        print(f"  Radius range: [{radii.min():.3f}, {radii.max():.3f}]")
        n_outside = np.sum((radii < r_min) | (radii > r_max))
        print(f"  Points outside shell: {n_outside}/{len(radii)}")
    
    # Clip to outer boundary first
    if radii.max() > r_max:
        if verbose:
            print(f"  Clipping at outer boundary (r={r_max:.3f})...")
        # clip_scalar keeps values below the threshold
        clipped = mesh_with_radius.clip_scalar("radius", value=r_max)
    else:
        clipped = mesh_with_radius.copy()
    
    # Clip to inner boundary if needed
    if r_min > 0 and radii.min() < r_min:
        if verbose:
            print(f"  Clipping at inner boundary (r={r_min:.3f})...")
        # Invert to keep values above threshold
        clipped = clipped.clip_scalar("radius", value=r_min, invert=True)
    
    # Remove the radius data
    if "radius" in clipped.array_names:
        clipped = clipped.copy()
        clipped.point_data.remove("radius")
    
    # Clean the mesh
    clipped = clipped.clean(tolerance=1e-9)
    
    if verbose:
        print(f"  Clipped mesh: {clipped.n_points} vertices, {clipped.n_cells} faces")
    
    return clipped


def cap_spherical_holes(mesh: pv.PolyData, 
                       target_radius: Optional[float] = None,
                       verbose: bool = False) -> pv.PolyData:
    """
    Cap holes at spherical boundaries.
    
    Parameters
    ----------
    mesh : pv.PolyData
        Mesh with boundary edges to cap.
    target_radius : float, optional
        If provided, project cap vertices to this radius.
    verbose : bool
        Print progress information.
        
    Returns
    -------
    capped : pv.PolyData
        Mesh with capped boundaries.
    """
    if verbose:
        print("Capping spherical boundaries...")
    
    # Extract boundary edges
    edges = mesh.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False
    )
    
    if edges.n_points == 0:
        if verbose:
            print("  No boundaries found - mesh is already closed")
        return mesh
    
    if verbose:
        print(f"  Found {edges.n_points} boundary points")
    
    # Group boundary points into loops
    loops = _extract_boundary_loops(edges, mesh)
    
    if verbose:
        print(f"  Found {len(loops)} boundary loops")
    
    # Create caps for each loop
    caps = []
    for i, loop_points in enumerate(loops):
        if len(loop_points) < 3:
            continue
            
        if verbose:
            print(f"  Capping loop {i+1} with {len(loop_points)} points...")
        
        # Create triangulated cap
        cap = _create_radial_cap(loop_points, target_radius)
        if cap is not None:
            caps.append(cap)
    
    # Merge mesh with caps
    if caps:
        result = mesh.copy()
        for cap in caps:
            result = result + cap
        
        # Clean and merge
        result = result.clean(tolerance=1e-6)
        
        if verbose:
            print(f"  Final mesh: {result.n_points} vertices, {result.n_cells} faces")
        
        return result
    else:
        return mesh


def _extract_boundary_loops(edges: pv.PolyData, 
                           original_mesh: pv.PolyData) -> List[np.ndarray]:
    """Extract connected boundary loops from edge mesh."""
    # For now, return all boundary points as one loop
    # TODO: Implement proper loop extraction using connectivity
    
    if edges.n_points == 0:
        return []
    
    # Map edge points back to original mesh
    edge_points = edges.points
    
    # Find corresponding points in original mesh
    original_indices = []
    for ep in edge_points:
        distances = np.linalg.norm(original_mesh.points - ep, axis=1)
        closest = np.argmin(distances)
        if distances[closest] < 1e-9:
            original_indices.append(closest)
    
    # Get unique boundary points
    boundary_points = original_mesh.points[original_indices]
    
    # Group by approximate radius
    radii = np.linalg.norm(boundary_points, axis=1)
    unique_radii = []
    loops = []
    
    # Cluster points by radius
    sorted_indices = np.argsort(radii)
    current_radius = radii[sorted_indices[0]]
    current_loop = [boundary_points[sorted_indices[0]]]
    
    for idx in sorted_indices[1:]:
        r = radii[idx]
        if abs(r - current_radius) < 0.05:  # Same radius level
            current_loop.append(boundary_points[idx])
        else:
            # Start new loop
            if len(current_loop) >= 3:
                loops.append(np.array(current_loop))
            current_radius = r
            current_loop = [boundary_points[idx]]
    
    # Don't forget last loop
    if len(current_loop) >= 3:
        loops.append(np.array(current_loop))
    
    return loops


def _create_radial_cap(boundary_points: np.ndarray, 
                      target_radius: Optional[float] = None) -> Optional[pv.PolyData]:
    """Create a radial triangulation cap for a boundary loop."""
    n_points = len(boundary_points)
    if n_points < 3:
        return None
    
    # Find center and average radius
    center = np.mean(boundary_points, axis=0)
    avg_radius = np.mean(np.linalg.norm(boundary_points, axis=1))
    
    # Project center to sphere if target radius provided
    if target_radius is not None:
        center_norm = np.linalg.norm(center)
        if center_norm > 0:
            center = center * target_radius / center_norm
    else:
        # Use average radius
        center_norm = np.linalg.norm(center)
        if center_norm > 0:
            center = center * avg_radius / center_norm
    
    # Order points by angle for clean triangulation
    # Project to plane perpendicular to radial direction
    radial_dir = center / np.linalg.norm(center)
    
    # Find two perpendicular vectors in the tangent plane
    if abs(radial_dir[2]) < 0.9:
        u = np.cross(radial_dir, [0, 0, 1])
    else:
        u = np.cross(radial_dir, [1, 0, 0])
    u = u / np.linalg.norm(u)
    v = np.cross(radial_dir, u)
    
    # Project boundary points to 2D
    centered = boundary_points - center
    coords_2d = np.column_stack([
        np.dot(centered, u),
        np.dot(centered, v)
    ])
    
    # Sort by angle
    angles = np.arctan2(coords_2d[:, 1], coords_2d[:, 0])
    sorted_indices = np.argsort(angles)
    sorted_points = boundary_points[sorted_indices]
    
    # Create radial triangulation
    all_points = np.vstack([sorted_points, center])
    center_idx = n_points
    
    faces = []
    for i in range(n_points):
        next_i = (i + 1) % n_points
        # Triangle: boundary[i] -> boundary[i+1] -> center
        faces.extend([3, i, next_i, center_idx])
    
    # Create cap mesh
    cap = pv.PolyData(all_points, faces)
    
    return cap


def spherical_shell_healing_v2(mesh: pv.PolyData,
                              r_min: float,
                              r_max: float,
                              smooth: bool = True,
                              smooth_iterations: int = 10,
                              verbose: bool = False) -> pv.PolyData:
    """
    Complete spherical shell healing pipeline using PyVista's clipping.
    
    Parameters
    ----------
    mesh : pv.PolyData
        Input mesh (typically a distorted Riemann sphere).
    r_min : float
        Minimum radius for the shell.
    r_max : float
        Maximum radius for the shell.
    smooth : bool
        Apply smoothing after healing.
    smooth_iterations : int
        Number of smoothing iterations.
    verbose : bool
        Print progress information.
        
    Returns
    -------
    healed : pv.PolyData
        Healed mesh confined to spherical shell.
    """
    if verbose:
        print(f"\nSpherical Shell Healing v2")
        print(f"Target shell: r ∈ [{r_min:.3f}, {r_max:.3f}]")
    
    # Step 1: Clip to shell
    clipped = spherical_shell_clip(mesh, r_min, r_max, verbose)
    
    if clipped.n_points == 0:
        if verbose:
            print("Warning: Clipping removed all geometry!")
        return mesh
    
    # Step 2: Cap any holes
    capped = cap_spherical_holes(clipped, verbose=verbose)
    
    # Step 3: Optional smoothing
    if smooth and smooth_iterations > 0:
        if verbose:
            print(f"\nApplying {smooth_iterations} smoothing iterations...")
        
        # Use Taubin smoothing to preserve volume
        try:
            capped = capped.smooth_taubin(
                n_iter=smooth_iterations,
                pass_band=0.1,
                non_manifold_smoothing=False,
                normalize_coordinates=True
            )
        except:
            # Fallback to standard smoothing
            capped = capped.smooth(
                n_iter=smooth_iterations,
                relaxation_factor=0.1,
                feature_smoothing=False,
                boundary_smoothing=True
            )
    
    # Step 4: Final cleanup
    healed = capped.clean(tolerance=1e-9)
    
    # Step 5: Validate
    if verbose:
        print("\nValidating healed mesh:")
        # Basic validation
        edges = healed.extract_feature_edges(boundary_edges=True)
        print(f"  Boundary edges: {edges.n_cells}")
        print(f"  Is watertight: {edges.n_cells == 0}")
    
    return healed