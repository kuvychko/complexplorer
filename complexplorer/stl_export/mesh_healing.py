"""
Fixed mesh healing utilities compatible with various PyVista versions.

This module provides mesh healing specifically designed to fix issues
that cause empty layers in 3D printing slicers.
"""

import numpy as np
import pyvista as pv
from typing import Optional, Tuple, List, Dict
import warnings


class MeshHealer:
    """
    Standard mesh healer for basic mesh cleanup.
    
    Parameters
    ----------
    tolerance : float, default=1e-6
        Tolerance for duplicate vertex removal.
    smooth : bool, default=False
        If True, apply smoothing after healing.
    smooth_iterations : int, default=50
        Number of smoothing iterations.
    smooth_factor : float, default=0.1
        Relaxation factor for smoothing (0-1).
    """
    
    def __init__(self, tolerance: float = 1e-6, smooth: bool = False,
                 smooth_iterations: int = 50, smooth_factor: float = 0.1):
        self.tolerance = tolerance
        self.smooth = smooth
        self.smooth_iterations = smooth_iterations
        self.smooth_factor = smooth_factor
        
    def heal_mesh(self, mesh: pv.PolyData, verbose: bool = False) -> pv.PolyData:
        """
        Basic mesh healing pipeline.
        
        Parameters
        ----------
        mesh : pv.PolyData
            Input mesh to heal.
        verbose : bool, default=False
            If True, print progress information.
            
        Returns
        -------
        healed_mesh : pv.PolyData
            Healed mesh ready for cutting and printing.
        """
        if verbose:
            print(f"Initial mesh: {mesh.n_points} vertices, {mesh.n_cells} faces")
            
        # Step 1: Clean duplicate vertices
        mesh = mesh.clean(tolerance=self.tolerance)
        if verbose:
            print(f"After cleaning: {mesh.n_points} vertices")
            
        # Step 2: Fill small holes
        try:
            mesh = mesh.fill_holes(hole_size=100)
            if verbose:
                print("Filled small holes")
        except:
            if verbose:
                print("Warning: Could not fill holes")
                
        # Step 3: Extract largest connected component
        connectivity = mesh.connectivity(extraction_mode='largest')
        if connectivity.n_points < mesh.n_points:
            mesh = connectivity
            if verbose:
                print(f"Extracted largest component: {mesh.n_points} vertices")
                
        # Step 4: Ensure consistent normals
        mesh = mesh.compute_normals(consistent_normals=True, auto_orient_normals=True)
        if verbose:
            print("Fixed normal orientations")
            
        # Step 5: Optional smoothing
        if self.smooth:
            mesh = mesh.smooth(
                n_iter=self.smooth_iterations,
                relaxation_factor=self.smooth_factor,
                feature_smoothing=False,
                boundary_smoothing=True,
                edge_angle=120.0,
                feature_angle=45.0
            )
            if verbose:
                print("Applied smoothing")
                
        # Final check
        if verbose:
            print(f"Final mesh: {mesh.n_points} vertices, {mesh.n_cells} faces")
            try:
                print(f"Is manifold: {mesh.is_manifold}")
            except:
                print("Is manifold: Unable to check")
                
        return mesh
    
    def validate_manifold(self, mesh: pv.PolyData) -> dict:
        """
        Validate mesh manifoldness and report issues.
        
        Parameters
        ----------
        mesh : pv.PolyData
            Mesh to validate.
            
        Returns
        -------
        report : dict
            Validation report with various metrics.
        """
        # Extract different edge types
        boundary = mesh.extract_feature_edges(
            boundary_edges=True,
            feature_edges=False,
            manifold_edges=False,
            non_manifold_edges=False
        )
        
        non_manifold = mesh.extract_feature_edges(
            boundary_edges=False,
            feature_edges=False,
            manifold_edges=False,
            non_manifold_edges=True
        )
        
        # Check if mesh has is_manifold property
        try:
            is_manifold = mesh.is_manifold
        except AttributeError:
            # Check if it's closed instead
            is_manifold = boundary.n_cells == 0 and non_manifold.n_cells == 0
        
        report = {
            'is_manifold': is_manifold,
            'n_boundary_edges': boundary.n_cells if boundary.n_cells > 0 else 0,
            'n_non_manifold_edges': non_manifold.n_cells if non_manifold.n_cells > 0 else 0,
            'n_vertices': mesh.n_points,
            'n_faces': mesh.n_cells,
            'bounds': mesh.bounds,
            'is_closed': boundary.n_cells == 0
        }
        
        return report


class ImprovedMeshHealer(MeshHealer):
    """
    Enhanced mesh healer that specifically targets 3D printing issues.
    
    This version includes additional fixes for empty layers and spikes
    without relying on methods that might not be available.
    """
    
    def heal_mesh(self, mesh: pv.PolyData, verbose: bool = False) -> pv.PolyData:
        """
        Comprehensive mesh healing pipeline with layer gap prevention.
        
        Parameters
        ----------
        mesh : pv.PolyData
            Input mesh to heal.
        verbose : bool, default=False
            If True, print progress information.
            
        Returns
        -------
        healed_mesh : pv.PolyData
            Healed mesh ready for cutting and printing.
        """
        if verbose:
            print(f"Initial mesh: {mesh.n_points} vertices, {mesh.n_cells} faces")
            
        # Step 1: Remove degenerate faces
        mesh = self._remove_degenerate_faces(mesh, verbose)
        
        # Step 2: Clean duplicate vertices
        mesh = mesh.clean(tolerance=self.tolerance)
        if verbose:
            print(f"After cleaning: {mesh.n_points} vertices")
            
        # Step 3: Fix vertical gaps that cause empty layers
        mesh = self._fix_vertical_gaps(mesh, verbose)
        
        # Step 4: Fill small holes
        try:
            mesh = mesh.fill_holes(hole_size=100)
            if verbose:
                print("Filled small holes")
        except:
            if verbose:
                print("Warning: Could not fill holes")
                
        # Step 5: Extract largest connected component
        connectivity = mesh.connectivity(extraction_mode='largest')
        if connectivity.n_points < mesh.n_points:
            mesh = connectivity
            if verbose:
                print(f"Extracted largest component: {mesh.n_points} vertices")
                
        # Step 6: Ensure consistent normals
        mesh = mesh.compute_normals(consistent_normals=True, auto_orient_normals=True)
        if verbose:
            print("Fixed normal orientations")
            
        # Step 7: Smooth spikes and artifacts
        if self.smooth:
            mesh = self._smooth_mesh_carefully(mesh, verbose)
            
        # Step 8: Final cleanup
        mesh = self._final_cleanup(mesh, verbose)
        
        # Final validation
        if verbose:
            self._report_mesh_quality(mesh)
            
        return mesh
    
    def _remove_degenerate_faces(self, mesh: pv.PolyData, verbose: bool = False) -> pv.PolyData:
        """Remove faces with zero area or degenerate triangles."""
        if verbose:
            print("Removing degenerate faces...")
            
        # Get face areas
        mesh = mesh.compute_cell_sizes()
        areas = mesh["Area"]
        
        # Find non-degenerate faces
        min_area = 1e-10
        valid_faces = areas > min_area
        
        if not all(valid_faces):
            n_removed = np.sum(~valid_faces)
            if verbose:
                print(f"  Removed {n_removed} degenerate faces")
            
            # Extract valid faces
            mesh = mesh.extract_cells(np.where(valid_faces)[0])
            
        return mesh
    
    def _fix_vertical_gaps(self, mesh: pv.PolyData, verbose: bool = False) -> pv.PolyData:
        """
        Fix vertical gaps that cause empty layers in slicing.
        
        This identifies points that are isolated in Z-layers and adjusts them.
        """
        if verbose:
            print("Fixing vertical gaps...")
            
        # Group points by Z-layers
        z_coords = mesh.points[:, 2]
        z_unique = np.unique(z_coords)
        
        # Find layers with very few points (potential gaps)
        layer_tolerance = 0.1  # mm
        problem_layers = []
        
        for z in z_unique:
            layer_mask = np.abs(z_coords - z) < layer_tolerance
            n_points = np.sum(layer_mask)
            
            # If a layer has very few points, it might cause gaps
            if 0 < n_points < 10:
                problem_layers.append((z, n_points))
                
        if problem_layers and verbose:
            print(f"  Found {len(problem_layers)} problematic Z-layers")
            
        # Fix problematic layers by merging nearby points
        for z_prob, n_points in problem_layers:
            # Find points in this layer
            layer_mask = np.abs(z_coords - z_prob) < layer_tolerance
            layer_indices = np.where(layer_mask)[0]
            
            # Find nearest Z-layer with more points
            z_distances = np.abs(z_unique - z_prob)
            z_distances[z_distances == 0] = np.inf
            
            # Check adjacent layers
            for z_near in z_unique[np.argsort(z_distances)[:2]]:
                near_mask = np.abs(z_coords - z_near) < layer_tolerance
                if np.sum(near_mask) > 20:  # Found a better layer
                    # Move problematic points to this layer
                    mesh.points[layer_indices, 2] = z_near
                    break
                    
        return mesh
    
    def _smooth_mesh_carefully(self, mesh: pv.PolyData, verbose: bool = False) -> pv.PolyData:
        """Apply careful smoothing to reduce spikes without creating gaps."""
        if verbose:
            print("Applying spike-aware smoothing...")
            
        # First pass: General smoothing
        mesh = mesh.smooth(
            n_iter=self.smooth_iterations,
            relaxation_factor=self.smooth_factor,
            feature_smoothing=False,
            boundary_smoothing=True,
            edge_angle=120.0,
            feature_angle=45.0
        )
        
        # Second pass: Taubin smoothing to preserve volume
        try:
            mesh = mesh.smooth_taubin(
                n_iter=self.smooth_iterations // 2,
                pass_band=0.1,
                non_manifold_smoothing=False,
                normalize_coordinates=True
            )
        except:
            if verbose:
                print("  Warning: Taubin smoothing not available")
        
        return mesh
    
    def _final_cleanup(self, mesh: pv.PolyData, verbose: bool = False) -> pv.PolyData:
        """Final cleanup pass to ensure mesh quality."""
        if verbose:
            print("Final cleanup...")
            
        # Remove any remaining duplicate vertices
        mesh = mesh.clean(tolerance=self.tolerance * 0.1)
        
        # Ensure no isolated vertices
        try:
            mesh = mesh.remove_points(remove_vertices=True)
        except:
            pass
            
        return mesh
    
    def _report_mesh_quality(self, mesh: pv.PolyData):
        """Report detailed mesh quality metrics."""
        print("\nMesh Quality Report:")
        print("-" * 40)
        
        # Basic stats
        print(f"Vertices: {mesh.n_points}")
        print(f"Faces: {mesh.n_cells}")
        
        # Calculate face areas
        mesh = mesh.compute_cell_sizes()
        areas = mesh["Area"]
        
        print(f"Face areas:")
        print(f"  Min: {areas.min():.6f}")
        print(f"  Max: {areas.max():.6f}")
        print(f"  Mean: {areas.mean():.6f}")
        print(f"  Tiny faces (< 1e-6): {np.sum(areas < 1e-6)}")
        
        # Z-layer analysis
        z_coords = mesh.points[:, 2]
        z_unique = np.unique(z_coords)
        print(f"\nZ-layers: {len(z_unique)}")
        
        # Find potential gap layers
        layer_counts = []
        for z in z_unique:
            count = np.sum(np.abs(z_coords - z) < 0.01)
            layer_counts.append(count)
            
        layer_counts = np.array(layer_counts)
        print(f"  Points per layer: {layer_counts.min()} - {layer_counts.max()}")
        print(f"  Sparse layers (< 10 points): {np.sum(layer_counts < 10)}")
    
    def validate_manifold(self, mesh: pv.PolyData) -> dict:
        """
        Validate mesh manifoldness and report issues.
        
        Extended version with layer gap analysis.
        """
        # Get base validation
        report = super().validate_manifold(mesh)
        
        # Layer gap analysis
        z_coords = mesh.points[:, 2]
        z_sorted = np.sort(np.unique(z_coords))
        
        gaps = []
        if len(z_sorted) > 1:
            z_diffs = np.diff(z_sorted)
            median_diff = np.median(z_diffs)
            # Gaps are spaces larger than 3x the median spacing
            gap_indices = np.where(z_diffs > 3 * median_diff)[0]
            gaps = [(z_sorted[i], z_sorted[i+1], z_diffs[i]) for i in gap_indices]
        
        report['n_layer_gaps'] = len(gaps)
        report['layer_gaps'] = gaps
        
        return report