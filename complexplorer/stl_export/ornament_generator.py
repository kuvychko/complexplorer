"""
STL ornament generator with improved mesh quality for 3D printing.

This module provides the main interface for generating 3D-printable STL files
from complex function visualizations on the Riemann sphere.
"""

import numpy as np
import pyvista as pv
from typing import Callable, Optional, Tuple, Dict, Union
import os
import warnings

from ..cmap import Phase, Cmap
from .mesh_healing import MeshHealer, ImprovedMeshHealer
from .mesh_utils import (
    cut_with_flat_plane, ensure_flat_bottom, remove_spikes_simple,
    RectangularSphereGenerator, stereographic_projection, ModulusScaling
)


class OrnamentGenerator:
    """
    Generate 3D-printable ornaments from complex functions.
    
    This class handles the complete pipeline from complex function evaluation
    on the Riemann sphere to generating watertight STL files with flat bases
    suitable for 3D printing.
    
    Parameters
    ----------
    func : Callable
        Complex function to visualize.
    resolution : int, default=150
        Mesh resolution (n_theta and n_phi for sphere generation).
        Higher values give more detail but larger files.
    scaling : str, default='constant'
        Modulus scaling method:
        - 'constant': All points at fixed radius
        - 'arctan': Smooth mapping to bounded range
        - 'logarithmic': Emphasizes small modulus values
        - 'linear_clamp': Linear up to threshold then clamped
    scaling_params : dict, optional
        Parameters for the scaling function.
    cmap : Cmap, optional
        Color map for visualization. Default is Phase(12).
    """
    
    def __init__(self, func: Callable, resolution: int = 150,
                 scaling: str = 'constant', scaling_params: Optional[dict] = None,
                 cmap: Optional[Cmap] = None):
        self.func = func
        self.resolution = resolution
        self.scaling = scaling
        self.scaling_params = scaling_params or {}
        self.cmap = cmap or Phase(12)
        
        # Components
        self.healer = MeshHealer(smooth=False)
        self.improved_healer = ImprovedMeshHealer(smooth=False)
        
        # Generated meshes
        self.sphere_mesh = None
        self.healed_mesh = None
        self.top_half = None
        self.bottom_half = None
        
    def generate_sphere(self, verbose: bool = False) -> pv.PolyData:
        """
        Generate the Riemann sphere mesh.
        
        Parameters
        ----------
        verbose : bool, default=False
            Print progress information.
            
        Returns
        -------
        sphere_mesh : pv.PolyData
            Generated sphere mesh with color information.
        """
        if verbose:
            print(f"Generating sphere with resolution {self.resolution}")
            
        # Generate base sphere - always use rectangular for stability
        generator = RectangularSphereGenerator(
            radius=1.0,
            n_theta=self.resolution,
            n_phi=self.resolution,
            avoid_poles=True  # Important for better pole behavior
        )
        sphere = generator.generate()
        
        # Get sphere points and apply stereographic projection
        points = sphere.points
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        w = stereographic_projection(x, y, z, from_north=True)
        
        # Evaluate function
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            f_vals = self.func(w)
        
        # Handle infinities
        finite_mask = np.isfinite(f_vals)
        f_vals[~finite_mask] = 0
        
        # Get colors
        n_points = self.resolution * self.resolution
        f_vals_reshaped = f_vals.reshape((self.resolution, self.resolution))
        rgb = self.cmap.rgb(f_vals_reshaped)
        rgb_flat = rgb.reshape(-1, 3)
        sphere["RGB"] = rgb_flat
        
        # Apply modulus scaling
        moduli = np.abs(f_vals)
        
        # Apply scaling
        if self.scaling == 'constant':
            radius = self.scaling_params.get('radius', 1.0)
            radii = ModulusScaling.constant(moduli, radius)
        elif self.scaling == 'arctan':
            r_min = self.scaling_params.get('r_min', 0.2)
            r_max = self.scaling_params.get('r_max', 1.0)
            radii = ModulusScaling.arctan(moduli, r_min, r_max)
        elif self.scaling == 'logarithmic':
            base = self.scaling_params.get('base', np.e)
            r_min = self.scaling_params.get('r_min', 0.2)
            r_max = self.scaling_params.get('r_max', 1.0)
            radii = ModulusScaling.logarithmic(moduli, base, r_min, r_max)
        elif self.scaling == 'linear_clamp':
            m_max = self.scaling_params.get('m_max', 10)
            r_min = self.scaling_params.get('r_min', 0.2)
            r_max = self.scaling_params.get('r_max', 1.0)
            radii = ModulusScaling.linear_clamp(moduli, m_max, r_min, r_max)
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling}")
            
        # Handle infinities in radii
        radii[~finite_mask] = radii[finite_mask].max() if np.any(finite_mask) else 1.0
        
        # Apply radial scaling
        radii_reshaped = radii.reshape((self.resolution, self.resolution))
        
        # Get original angles
        theta = np.linspace(0.01, np.pi - 0.01, self.resolution)
        phi = np.linspace(0, 2 * np.pi, self.resolution)
        THETA, PHI = np.meshgrid(theta, phi)
        
        # Apply radial scaling
        X = radii_reshaped * np.sin(THETA) * np.cos(PHI)
        Y = radii_reshaped * np.sin(THETA) * np.sin(PHI)
        Z = radii_reshaped * np.cos(THETA)
        
        # Create new scaled grid
        sphere = pv.StructuredGrid(X, Y, Z)
        sphere["RGB"] = rgb_flat
        
        # Convert to PolyData for better processing
        sphere = sphere.extract_surface()
        
        # Store additional data
        sphere["magnitude"] = moduli
        sphere["phase"] = np.angle(f_vals)
        
        self.sphere_mesh = sphere
        
        if verbose:
            print(f"Generated sphere: {sphere.n_points} vertices, {sphere.n_cells} faces")
            
        return sphere
    
    def heal_mesh(self, smooth: bool = True, smooth_iterations: int = 20,
                  remove_spikes_enabled: bool = False, use_improved_healer: bool = True,
                  verbose: bool = False) -> pv.PolyData:
        """
        Heal the sphere mesh to fix defects.
        
        Parameters
        ----------
        smooth : bool, default=True
            Apply smoothing during healing.
        smooth_iterations : int, default=20
            Number of smoothing iterations.
        remove_spikes_enabled : bool, default=False
            Remove spike artifacts. Only enable for functions with known spike issues.
        use_improved_healer : bool, default=True
            Use the improved healer that specifically targets empty layers.
        verbose : bool, default=False
            Print progress information.
            
        Returns
        -------
        healed_mesh : pv.PolyData
            Healed mesh ready for cutting.
        """
        if self.sphere_mesh is None:
            self.generate_sphere(verbose=verbose)
            
        if use_improved_healer:
            # Use improved healer for better layer handling
            self.improved_healer.smooth = smooth
            self.improved_healer.smooth_iterations = smooth_iterations
            self.improved_healer.smooth_factor = 0.2
            
            if verbose:
                print("Healing mesh with improved healer...")
                
            self.healed_mesh = self.improved_healer.heal_mesh(self.sphere_mesh, verbose=verbose)
        else:
            # Use standard healer
            self.healer.smooth = smooth
            self.healer.smooth_iterations = smooth_iterations
            self.healer.smooth_factor = 0.2
            
            if verbose:
                print("Healing mesh...")
                
            self.healed_mesh = self.healer.heal_mesh(self.sphere_mesh, verbose=verbose)
        
        # Remove spikes before smoothing (only if needed)
        if remove_spikes_enabled:
            self.healed_mesh = remove_spikes_simple(self.healed_mesh, max_deviation=3.0, verbose=verbose)
        
        # Additional smoothing specifically for 3D printing
        if smooth and smooth_iterations > 0:
            if verbose:
                print("Applying Taubin smoothing for better surface quality...")
            
            # Taubin smoothing is better at preserving volume while reducing artifacts
            self.healed_mesh = self.healed_mesh.smooth_taubin(
                n_iter=smooth_iterations,
                pass_band=0.1,  # Lower values preserve features better
                non_manifold_smoothing=False,
                normalize_coordinates=True
            )
            
            # Additional pass of standard smoothing for very smooth result
            if smooth_iterations > 30:
                self.healed_mesh = self.healed_mesh.smooth(
                    n_iter=10,
                    relaxation_factor=0.1,
                    feature_smoothing=False,
                    boundary_smoothing=True,
                    edge_angle=120.0,
                    feature_angle=45.0
                )
        
        # Final cleanup
        self.healed_mesh = self.healed_mesh.clean(tolerance=1e-6)
        
        return self.healed_mesh
    
    def cut(self, mode: str = 'real', verbose: bool = False) -> Tuple[pv.PolyData, pv.PolyData]:
        """
        Cut the healed mesh into two halves with perfectly flat boundaries.
        
        Parameters
        ----------
        mode : str, default='real'
            Cutting mode:
            - 'real': Cut along real axis (y=0)
            - 'imaginary': Cut along imaginary axis (x=0)
            - 'angle:degrees': Cut at specified angle
        verbose : bool, default=False
            Print progress information.
            
        Returns
        -------
        top_half : pv.PolyData
            Top/positive half.
        bottom_half : pv.PolyData
            Bottom/negative half.
        """
        if self.healed_mesh is None:
            self.heal_mesh(verbose=verbose)
            
        # Determine cutting axis and position
        if mode == 'real':
            axis = 'y'
            position = 0.0
            if verbose:
                print("Cutting along real axis (y=0)")
        elif mode == 'imaginary':
            axis = 'x'
            position = 0.0
            if verbose:
                print("Cutting along imaginary axis (x=0)")
        elif mode.startswith('angle:'):
            # For angled cuts, rotate the mesh first
            angle_deg = float(mode.split(':')[1])
            angle_rad = np.radians(angle_deg)
            if verbose:
                print(f"Rotating mesh by {-angle_deg}° for angled cut")
            
            # Rotate mesh so the cut angle aligns with y axis
            rotated_mesh = self.healed_mesh.rotate_z(-angle_deg, inplace=False)
            
            # Cut along y axis
            positive_half, negative_half = cut_with_flat_plane(
                rotated_mesh, axis='y', position=0.0, verbose=verbose
            )
            
            # Rotate back
            self.top_half = positive_half.rotate_z(angle_deg, inplace=False)
            self.bottom_half = negative_half.rotate_z(angle_deg, inplace=False)
            
            # Copy color data if present
            self._transfer_colors()
            
            return self.top_half, self.bottom_half
        else:
            raise ValueError(f"Unknown cut_mode: {mode}")
            
        if verbose:
            print("Using flat cutting algorithm...")
            
        # Use the flat_cutter for perfectly flat cuts
        self.top_half, self.bottom_half = cut_with_flat_plane(
            self.healed_mesh, axis=axis, position=position, verbose=verbose
        )
        
        # Copy color data if present
        self._transfer_colors()
        
        # Additional healing of cut meshes
        if verbose:
            print("Healing cut meshes...")
        
        # Quick heal to ensure manifoldness
        healer = MeshHealer(smooth=False)
        self.top_half = healer.heal_mesh(self.top_half, verbose=False)
        self.bottom_half = healer.heal_mesh(self.bottom_half, verbose=False)
        
        return self.top_half, self.bottom_half
    
    def _transfer_colors(self):
        """Transfer color data from healed mesh to cut meshes."""
        if 'RGB' in self.healed_mesh.array_names:
            for mesh in [self.top_half, self.bottom_half]:
                if mesh.n_points > 0:
                    # Use nearest neighbor interpolation
                    closest_points = []
                    for p in mesh.points:
                        distances = np.linalg.norm(self.healed_mesh.points - p, axis=1)
                        closest_idx = np.argmin(distances)
                        closest_points.append(closest_idx)
                    
                    # Copy RGB values
                    if 'RGB' in self.healed_mesh.point_data:
                        mesh['RGB'] = self.healed_mesh['RGB'][closest_points]
    
    def validate_printability(self, mesh: pv.PolyData, verbose: bool = True) -> Dict[str, bool]:
        """
        Validate mesh for 3D printing.
        
        Parameters
        ----------
        mesh : pv.PolyData
            Mesh to validate.
        verbose : bool, default=True
            Print validation results.
            
        Returns
        -------
        validation : dict
            Validation results.
        """
        validation = self.healer.validate_manifold(mesh)
        
        # Additional checks
        validation['has_color'] = 'RGB' in mesh.array_names
        validation['reasonable_size'] = (
            mesh.n_points > 100 and 
            mesh.n_points < 1000000 and
            mesh.n_cells > 100 and
            mesh.n_cells < 2000000
        )
        
        # Check if bottom is flat
        bounds = mesh.bounds
        z_min = bounds[4]
        
        # Get all points at minimum Z
        bottom_points = mesh.points[np.abs(mesh.points[:, 2] - z_min) < 0.001]
        if len(bottom_points) > 10:
            z_variance = np.var(bottom_points[:, 2])
            validation['flat_bottom'] = z_variance < 1e-6
        else:
            validation['flat_bottom'] = False
            
        if verbose:
            print("\nMesh Validation Report:")
            print("-" * 30)
            for key, value in validation.items():
                if isinstance(value, bool):
                    status = "✓" if value else "✗"
                    print(f"{status} {key}: {value}")
                else:
                    print(f"  {key}: {value}")
                    
        return validation
    
    def export(self, filename: str, mesh: pv.PolyData, size_mm: float = 80,
               binary: bool = True, verbose: bool = True) -> None:
        """
        Export mesh to STL file with proper scaling and orientation.
        
        Parameters
        ----------
        filename : str
            Output filename (should end with .stl).
        mesh : pv.PolyData
            Mesh to export.
        size_mm : float, default=80
            Target diameter in millimeters.
        binary : bool, default=True
            Save as binary STL (smaller file size).
        verbose : bool, default=True
            Print export information.
        """
        # First, ensure the mesh is oriented correctly for printing
        # The flat cut should be at Z=0
        bounds = mesh.bounds
        z_min = bounds[4]
        
        # Translate so minimum Z is at 0
        if abs(z_min) > 0.001:
            mesh = mesh.translate([0, 0, -z_min], inplace=False)
            
        # Ensure the bottom is perfectly flat
        mesh = ensure_flat_bottom(mesh, z_position=0.0, tolerance=0.1)
        
        # Scale mesh to target size
        bounds = mesh.bounds
        current_size = max(
            bounds[1] - bounds[0],  # X extent
            bounds[3] - bounds[2],  # Y extent  
            bounds[5] - bounds[4]   # Z extent
        )
        
        scale_factor = size_mm / current_size
        scaled_mesh = mesh.scale(scale_factor, inplace=False)
        
        # Final cleanup
        scaled_mesh = scaled_mesh.clean(tolerance=1e-6)
        
        # Save STL
        if not filename.endswith('.stl'):
            filename += '.stl'
            
        scaled_mesh.save(filename, binary=binary)
        
        if verbose:
            file_size_mb = os.path.getsize(filename) / (1024 * 1024)
            final_bounds = scaled_mesh.bounds
            print(f"\nExported to: {filename}")
            print(f"File size: {file_size_mb:.2f} MB")
            print(f"Dimensions (mm): X={final_bounds[1]-final_bounds[0]:.1f}, "
                  f"Y={final_bounds[3]-final_bounds[2]:.1f}, "
                  f"Z={final_bounds[5]-final_bounds[4]:.1f}")
            print(f"Vertices: {scaled_mesh.n_points}")
            print(f"Triangles: {scaled_mesh.n_cells}")
            
    def generate_ornament(self, cut_mode: str = 'real', size_mm: float = 80,
                         smooth: bool = True, smooth_iterations: int = 30,
                         output_prefix: str = 'ornament',
                         verbose: bool = True) -> Tuple[str, str]:
        """
        Complete pipeline to generate ornament STL files.
        
        Parameters
        ----------
        cut_mode : str, default='real'
            Cutting mode: 'real', 'imaginary', or 'angle:degrees'.
        size_mm : float, default=80
            Target diameter in millimeters.
        smooth : bool, default=True
            Apply smoothing during healing.
        smooth_iterations : int, default=30
            Number of smoothing iterations.
        output_prefix : str, default='ornament'
            Prefix for output filenames.
        verbose : bool, default=True
            Print progress information.
            
        Returns
        -------
        top_filename : str
            Filename of top half STL.
        bottom_filename : str
            Filename of bottom half STL.
        """
        # Generate sphere
        self.generate_sphere(verbose=verbose)
        
        # Heal mesh with smoothing
        self.heal_mesh(smooth=smooth, smooth_iterations=smooth_iterations, 
                      remove_spikes_enabled=False, use_improved_healer=True, verbose=verbose)
        
        # Cut mesh
        self.cut(mode=cut_mode, verbose=verbose)
        
        # Validate
        if verbose:
            print("\nValidating top half:")
        self.validate_printability(self.top_half, verbose=verbose)
        
        if verbose:
            print("\nValidating bottom half:")
        self.validate_printability(self.bottom_half, verbose=verbose)
        
        # Export
        top_filename = f"{output_prefix}_top.stl"
        bottom_filename = f"{output_prefix}_bottom.stl"
        
        self.export(top_filename, self.top_half, size_mm=size_mm, verbose=verbose)
        self.export(bottom_filename, self.bottom_half, size_mm=size_mm, verbose=verbose)
        
        return top_filename, bottom_filename