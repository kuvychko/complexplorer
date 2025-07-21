"""
Fast mesh utility functions for STL export.

This module contains optimized helper functions for mesh generation, cutting, and healing.
"""

import numpy as np
import pyvista as pv
from typing import Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..domain import Domain


def cut_with_flat_plane(mesh: pv.PolyData, axis: str = 'y', 
                       position: float = 0.0, verbose: bool = False) -> Tuple[pv.PolyData, pv.PolyData]:
    """
    Cut mesh along an axis and ensure the cut boundary is perfectly flat.
    
    This function is crucial for 3D printing as it guarantees flat surfaces
    at the cutting plane, preventing gaps and ensuring good bed adhesion.
    
    Parameters
    ----------
    mesh : pv.PolyData
        Input mesh to cut.
    axis : str, default='y'
        Axis to cut along ('x', 'y', or 'z').
    position : float, default=0.0
        Position along the axis to cut.
    verbose : bool, default=False
        Print progress.
        
    Returns
    -------
    positive_half : pv.PolyData
        Half on the positive side of the cut.
    negative_half : pv.PolyData  
        Half on the negative side of the cut.
    """
    # Determine normal based on axis
    if axis == 'x':
        normal = [1, 0, 0]
        axis_idx = 0
    elif axis == 'y':
        normal = [0, 1, 0]
        axis_idx = 1
    elif axis == 'z':
        normal = [0, 0, 1]
        axis_idx = 2
    else:
        raise ValueError(f"Invalid axis: {axis}")
        
    origin = [0, 0, 0]
    origin[axis_idx] = position
    
    # Clip to get both halves
    positive_half = mesh.clip(normal=normal, origin=origin, invert=False)
    negative_half = mesh.clip(normal=[-n for n in normal], origin=origin, invert=False)
    
    # Extract surfaces if needed
    if hasattr(positive_half, 'extract_surface'):
        positive_half = positive_half.extract_surface()
    if hasattr(negative_half, 'extract_surface'):
        negative_half = negative_half.extract_surface()
        
    # Fix boundary points to be perfectly flat
    positive_half = _flatten_boundary(positive_half, axis, position, verbose)
    negative_half = _flatten_boundary(negative_half, axis, position, verbose)
    
    # Cap the boundaries
    positive_half = _cap_flat_boundary(positive_half, axis, position, verbose)
    negative_half = _cap_flat_boundary(negative_half, axis, position, verbose)
    
    return positive_half, negative_half


def _flatten_boundary(mesh: pv.PolyData, axis: str, position: float, 
                     verbose: bool = False) -> pv.PolyData:
    """Flatten boundary points to exact position."""
    # Extract boundary
    edges = mesh.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False
    )
    
    if edges.n_points == 0:
        return mesh
        
    # Get boundary point indices
    boundary_indices = []
    for i in range(edges.n_points):
        # Find closest point in original mesh
        pt = edges.points[i]
        distances = np.linalg.norm(mesh.points - pt, axis=1)
        closest_idx = np.argmin(distances)
        if distances[closest_idx] < 1e-9:
            boundary_indices.append(closest_idx)
            
    boundary_indices = np.array(boundary_indices)
    
    # Flatten these points
    axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
    mesh.points[boundary_indices, axis_idx] = position
    
    if verbose:
        print(f"  Flattened {len(boundary_indices)} boundary points to {axis}={position}")
        
    return mesh


def _cap_flat_boundary(mesh: pv.PolyData, axis: str, position: float,
                      verbose: bool = False) -> pv.PolyData:
    """Cap the flat boundary with triangles."""
    # Extract boundary again (now flattened)
    edges = mesh.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False
    )
    
    if edges.n_points == 0:
        return mesh
        
    boundary_points = edges.points.copy()
    n_boundary = len(boundary_points)
    
    # Ensure all boundary points are at exact position
    axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
    boundary_points[:, axis_idx] = position
    
    # Check if this is a pole-crossing cut
    # For a cut perpendicular to an axis, check variation along that axis
    cut_range = boundary_points[:, axis_idx].max() - boundary_points[:, axis_idx].min()
    
    # The boundary should be planar (all points at same position along cut axis)
    # If there's significant variation, something is wrong
    if cut_range > 0.01:
        print(f"  WARNING: Boundary not planar! Range along {axis}: {cut_range:.4f}")
    
    # Check if boundary has multiple disconnected components (like at poles)
    # This happens when cutting through poles (Z axis cuts)
    is_pole_cut = False
    if axis == 'z':
        # For Z cuts, check if we're near the poles
        is_pole_cut = abs(position) > 0.9  # Near poles if |z| > 0.9
    
    if is_pole_cut and verbose:
        print(f"  Detected pole-crossing cut at z={position:.3f}")
    
    # Special handling for pole-crossing cuts
    if is_pole_cut:
        # For pole cuts, we need to handle the two separate loops
        # Split boundary into upper and lower parts based on Z
        z_median = np.median(boundary_points[:, 2])
        upper_mask = boundary_points[:, 2] > z_median
        lower_mask = ~upper_mask
        
        # Create two separate caps
        all_faces = []
        all_points = [boundary_points]
        
        for mask, name in [(upper_mask, "upper"), (lower_mask, "lower")]:
            if np.sum(mask) > 2:
                loop_points = boundary_points[mask]
                loop_center = np.median(loop_points, axis=0)
                loop_center[axis_idx] = position
                
                # Add center point
                center_idx = len(boundary_points) + len(all_points) - 1
                all_points.append(loop_center.reshape(1, 3))
                
                # Create fan triangulation for this loop
                loop_indices = np.where(mask)[0]
                
                # Sort by angle around center
                if axis == 'x':
                    angles = np.arctan2(loop_points[:, 2] - loop_center[2],
                                      loop_points[:, 1] - loop_center[1])
                elif axis == 'y':
                    angles = np.arctan2(loop_points[:, 2] - loop_center[2],
                                      loop_points[:, 0] - loop_center[0])
                else:  # z
                    angles = np.arctan2(loop_points[:, 1] - loop_center[1],
                                      loop_points[:, 0] - loop_center[0])
                
                sorted_order = np.argsort(angles)
                sorted_indices = loop_indices[sorted_order]
                
                # Build triangles
                for i in range(len(sorted_indices)):
                    j = (i + 1) % len(sorted_indices)
                    all_faces.extend([3, sorted_indices[i], sorted_indices[j], center_idx])
                
                if verbose:
                    print(f"    Created {len(sorted_indices)} triangles for {name} loop")
        
        # Create cap mesh
        all_points = np.vstack(all_points)
        cap = pv.PolyData(all_points, faces=all_faces)
        
    else:
        # Non-pole cut - use standard triangulation
        try:
            # Create a polyline from boundary points
            # First, order the points to form a closed loop
            from scipy.spatial import distance_matrix
            
            # Start with first point
            ordered_indices = [0]
            remaining = set(range(1, n_boundary))
            
            # Greedy nearest neighbor to order points
            while remaining:
                last_idx = ordered_indices[-1]
                distances = np.array([np.linalg.norm(boundary_points[last_idx] - boundary_points[i]) 
                                    for i in remaining])
                nearest_idx = list(remaining)[np.argmin(distances)]
                ordered_indices.append(nearest_idx)
                remaining.remove(nearest_idx)
            
            # Create ordered boundary
            ordered_boundary = boundary_points[ordered_indices]
            
            # Close the loop
            closed_boundary = np.vstack([ordered_boundary, ordered_boundary[0]])
            
            # Create polyline
            lines = np.full((n_boundary, 3), 2, dtype=int)
            lines[:, 1] = np.arange(n_boundary)
            lines[:, 2] = np.arange(1, n_boundary + 1)
            lines[-1, 2] = 0
            
            boundary_poly = pv.PolyData(ordered_boundary, lines=lines)
            
            # Fill the boundary using delaunay
            cap = boundary_poly.delaunay_2d(progress_bar=False)
            
            # Ensure cap is at exact position
            cap.points[:, axis_idx] = position
            
        except Exception as e:
            if verbose:
                print(f"  Delaunay triangulation failed: {e}, using simple fan triangulation")
            
            # Fallback to simple fan triangulation from centroid
            # But use a more robust center calculation
            if axis == 'x':
                other_axes = [1, 2]
            elif axis == 'y':
                other_axes = [0, 2]
            else:  # z
                other_axes = [0, 1]
            
            # Use median for more robust center
            center_3d = np.zeros(3)
            center_3d[other_axes[0]] = np.median(boundary_points[:, other_axes[0]])
            center_3d[other_axes[1]] = np.median(boundary_points[:, other_axes[1]])
            center_3d[axis_idx] = position
            
            # Sort points by angle for simple triangulation
            points_2d = boundary_points[:, other_axes]
            center_2d = points_2d.mean(axis=0)
            angles = np.arctan2(points_2d[:, 1] - center_2d[1],
                               points_2d[:, 0] - center_2d[0])
            sorted_indices = np.argsort(angles)
            
            # Build triangles
            all_points = np.vstack([boundary_points[sorted_indices], center_3d])
            center_idx = n_boundary
            
            faces = []
            for i in range(n_boundary):
                next_i = (i + 1) % n_boundary
                faces.extend([3, i, next_i, center_idx])
                
            # Create cap
            cap = pv.PolyData(all_points, faces=faces)
    
    # Merge with mesh
    capped = mesh + cap
    
    # Clean more aggressively to merge duplicate vertices
    capped = capped.clean(tolerance=1e-6)
    
    # Remove any duplicate faces
    if hasattr(capped, 'remove_duplicate_cells'):
        capped = capped.remove_duplicate_cells()
    
    if verbose:
        n_cap_faces = cap.n_faces if hasattr(cap, 'n_faces') else len(faces)//4
        print(f"  Added cap with {n_cap_faces} triangles")
        
    return capped


def ensure_flat_bottom(mesh: pv.PolyData, z_position: float = 0.0, tolerance: float = 0.1) -> pv.PolyData:
    """
    Ensure the mesh has a perfectly flat bottom at the specified Z position.
    
    This is crucial for first layer adhesion in 3D printing.
    
    Parameters
    ----------
    mesh : pv.PolyData
        Mesh to flatten.
    z_position : float, default=0.0
        Target Z position for the bottom.
    tolerance : float, default=0.1
        Distance tolerance for identifying bottom points.
        
    Returns
    -------
    mesh : pv.PolyData
        Mesh with flattened bottom.
    """
    # Find all points near the minimum Z
    z_min = mesh.bounds[4]
    
    # Get points near bottom
    bottom_mask = np.abs(mesh.points[:, 2] - z_min) < tolerance
    
    # Set them to exact Z position
    mesh.points[bottom_mask, 2] = z_position
    
    # Clean to merge any coincident points
    mesh = mesh.clean(tolerance=1e-9)
    
    return mesh


def remove_spikes_fast(mesh: pv.PolyData, percentile: float = 98.0, 
                      smooth_factor: float = 0.5, verbose: bool = False) -> pv.PolyData:
    """
    Fast spike removal using statistical outlier detection.
    
    Instead of checking all neighbor distances, this uses a percentile-based
    approach to identify potential spikes quickly.
    
    Parameters
    ----------
    mesh : pv.PolyData
        Input mesh.
    percentile : float, default=98.0
        Percentile threshold for spike detection.
    smooth_factor : float, default=0.5
        How much to smooth spike vertices (0-1).
    verbose : bool, default=False
        Print progress.
        
    Returns
    -------
    mesh : pv.PolyData
        Mesh with spikes removed.
    """
    if verbose:
        print("Fast spike detection...")
    
    # Calculate distance from each vertex to mesh centroid
    centroid = mesh.points.mean(axis=0)
    distances = np.linalg.norm(mesh.points - centroid, axis=1)
    
    # Find outliers using percentile
    threshold = np.percentile(distances, percentile)
    spike_mask = distances > threshold
    spike_indices = np.where(spike_mask)[0]
    
    if verbose:
        print(f"  Found {len(spike_indices)} potential spikes (top {100-percentile:.0f}%)")
    
    if len(spike_indices) == 0:
        return mesh
    
    # For spike vertices, move them toward local average
    # This is much faster than checking all neighbors
    for idx in spike_indices:
        # Find nearby points using distance threshold
        vertex_pos = mesh.points[idx]
        nearby_dists = np.linalg.norm(mesh.points - vertex_pos, axis=1)
        
        # Use median distance as radius
        median_dist = np.median(nearby_dists[nearby_dists > 0])
        nearby_mask = (nearby_dists < 2 * median_dist) & (nearby_dists > 0)
        
        if np.any(nearby_mask):
            # Average position of nearby points
            avg_pos = mesh.points[nearby_mask].mean(axis=0)
            # Smooth the spike
            mesh.points[idx] = (1 - smooth_factor) * vertex_pos + smooth_factor * avg_pos
    
    if verbose:
        print("  Spike smoothing complete")
    
    return mesh


def remove_spikes_simple(mesh: pv.PolyData, max_deviation: float = 3.0, 
                        verbose: bool = False) -> pv.PolyData:
    """
    Simplified spike removal - just use fast method.
    
    This replaces the slow O(n²) implementation.
    """
    return remove_spikes_fast(mesh, percentile=95.0, smooth_factor=0.5, verbose=verbose)


# Mesh generation classes
class RectangularSphereGenerator:
    """Generate sphere with rectangular (lat-lon) grid.
    
    Parameters
    ----------
    radius : float, default=1.0
        Sphere radius.
    n_theta : int, default=100
        Number of latitude divisions.
    n_phi : int, default=100
        Number of longitude divisions.
    avoid_poles : bool, default=True
        Whether to avoid exact poles in mesh.
    domain : Domain, optional
        If provided, only generate mesh points whose stereographic projections
        fall within this domain. Helps avoid numerical instability at extreme values.
    """
    
    def __init__(self, radius: float = 1.0, n_theta: int = 100, n_phi: int = 100,
                 avoid_poles: bool = True, domain: Optional['Domain'] = None):
        self.radius = radius
        self.n_theta = n_theta
        self.n_phi = n_phi
        self.avoid_poles = avoid_poles
        self.domain = domain
        
    def generate(self) -> pv.PolyData:
        """Generate sphere mesh, optionally constrained by domain."""
        # Create angles avoiding exact poles if requested
        if self.avoid_poles:
            theta = np.linspace(0.01, np.pi - 0.01, self.n_theta)
        else:
            theta = np.linspace(0, np.pi, self.n_theta)
            
        phi = np.linspace(0, 2 * np.pi, self.n_phi)
        THETA, PHI = np.meshgrid(theta, phi)
        
        # Convert to Cartesian
        X = self.radius * np.sin(THETA) * np.cos(PHI)
        Y = self.radius * np.sin(THETA) * np.sin(PHI)
        Z = self.radius * np.cos(THETA)
        
        # If domain is specified, filter points
        if self.domain is not None:
            # Project points to complex plane
            w = stereographic_projection(X.ravel(), Y.ravel(), Z.ravel())
            
            # Check which points are in domain
            in_domain = self.domain.infunc(w)
            in_domain = in_domain.reshape(X.shape)
            
            # Mark out-of-domain points as NaN
            X = np.where(in_domain, X, np.nan)
            Y = np.where(in_domain, Y, np.nan)
            Z = np.where(in_domain, Z, np.nan)
        
        # Create structured grid
        grid = pv.StructuredGrid(X, Y, Z)
        
        # Convert to PolyData and remove any NaN points
        mesh = grid.extract_surface()
        if self.domain is not None:
            # Remove cells with NaN vertices
            mesh = mesh.remove_cells(np.any(np.isnan(mesh.points), axis=1))
        
        return mesh


def stereographic_projection(x, y, z, from_north=True):
    """Apply stereographic projection from sphere to complex plane."""
    if from_north:
        # Project from north pole (0, 0, 1)
        denominator = 1 - z
        # Avoid division by zero at north pole
        denominator = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)
        w = (x + 1j * y) / denominator
    else:
        # Project from south pole (0, 0, -1)
        denominator = 1 + z
        denominator = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)
        w = (x + 1j * y) / denominator
    return w


class ModulusScaling:
    """Methods for scaling sphere radius based on function modulus."""
    
    @staticmethod
    def constant(moduli, radius=1.0):
        """Constant radius regardless of modulus."""
        return np.full_like(moduli, radius)
    
    @staticmethod
    def arctan(moduli, r_min=0.2, r_max=1.0):
        """Smooth scaling using arctangent."""
        # Map [0, ∞) to [0, 1] using arctan
        normalized = (2/np.pi) * np.arctan(moduli)
        return r_min + (r_max - r_min) * normalized
    
    @staticmethod
    def logarithmic(moduli, base=np.e, r_min=0.2, r_max=1.0):
        """Logarithmic scaling."""
        # Avoid log(0)
        safe_moduli = np.maximum(moduli, 1e-10)
        log_moduli = np.log(safe_moduli) / np.log(base)
        # Map to [0, 1] using sigmoid-like function
        normalized = 1 / (1 + np.exp(-log_moduli))
        return r_min + (r_max - r_min) * normalized
    
    @staticmethod
    def linear_clamp(moduli, m_max=10, r_min=0.2, r_max=1.0):
        """Linear scaling up to m_max, then clamped."""
        normalized = np.minimum(moduli / m_max, 1.0)
        return r_min + (r_max - r_min) * normalized