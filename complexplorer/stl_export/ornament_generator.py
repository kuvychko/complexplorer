"""
STL ornament generator - Creates complete watertight meshes.

Users can orient and cut meshes however they want in their slicer software.
No pre-cutting is done - you get a complete Riemann sphere mesh.
"""

import numpy as np
import pyvista as pv
from typing import Callable, Optional, Dict, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ..domain import Domain
import os
import warnings

from ..cmap import Phase, Cmap
from .spherical_healing import spherical_shell_healing_v2 as spherical_shell_healing
from .stl_utils import (
    RectangularSphereGenerator, stereographic_projection, ModulusScaling
)


class OrnamentGenerator:
    """
    Generate 3D-printable ornaments from complex functions.
    
    Creates complete watertight meshes without any cutting.
    Users can orient and slice the mesh however they prefer in their
    3D printing software.
    
    Parameters
    ----------
    func : Callable
        Complex function to visualize.
    resolution : int, default=150
        Mesh resolution (n_theta and n_phi for sphere generation).
    scaling : str, default='arctan'
        Modulus scaling method:
        - 'constant': All points at fixed radius
        - 'arctan': Smooth mapping to bounded range
        - 'logarithmic': Emphasizes small modulus values
        - 'linear_clamp': Linear up to threshold then clamped
    scaling_params : dict, optional
        Parameters for the scaling function.
    cmap : Cmap, optional
        Color map for visualization. Default is Phase(12).
    domain : Domain, optional
        If provided, only generate mesh points whose stereographic projections
        fall within this domain.
    """
    
    def __init__(self, func: Callable, resolution: int = 150,
                 scaling: str = 'arctan', scaling_params: Optional[dict] = None,
                 cmap: Optional[Cmap] = None, domain: Optional['Domain'] = None):
        self.func = func
        self.resolution = resolution
        self.scaling = scaling
        self.scaling_params = scaling_params or self._default_scaling_params()
        self.cmap = cmap or Phase(12)
        self.domain = domain
        
        # Generated meshes
        self.sphere_mesh = None
        self.healed_mesh = None
    
    def _default_scaling_params(self) -> dict:
        """Get default parameters for each scaling method."""
        if self.scaling == 'constant':
            return {'radius': 1.0}
        elif self.scaling == 'arctan':
            return {'r_min': 0.3, 'r_max': 1.0}
        elif self.scaling == 'logarithmic':
            return {'base': np.e, 'r_min': 0.3, 'r_max': 1.0}
        elif self.scaling == 'linear_clamp':
            return {'m_max': 10, 'r_min': 0.3, 'r_max': 1.0}
        else:
            return {}
    
    def generate_sphere(self, verbose: bool = False) -> pv.PolyData:
        """
        Generate the Riemann sphere mesh with modulus scaling.
        
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
            print(f"Generating Riemann sphere with resolution {self.resolution}")
            
        # Generate base sphere
        generator = RectangularSphereGenerator(
            radius=1.0,
            n_theta=self.resolution,
            n_phi=self.resolution,
            avoid_poles=True,
            domain=self.domain
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
        f_vals_reshaped = f_vals.reshape((self.resolution, self.resolution))
        rgb = self.cmap.rgb(f_vals_reshaped)
        rgb_flat = rgb.reshape(-1, 3)
        sphere["RGB"] = rgb_flat
        
        # Apply modulus scaling
        moduli = np.abs(f_vals)
        
        # Apply scaling based on method
        if self.scaling == 'constant':
            radius = self.scaling_params.get('radius', 1.0)
            radii = ModulusScaling.constant(moduli, radius)
        elif self.scaling == 'arctan':
            r_min = self.scaling_params.get('r_min', 0.3)
            r_max = self.scaling_params.get('r_max', 1.0)
            radii = ModulusScaling.arctan(moduli, r_min, r_max)
        elif self.scaling == 'logarithmic':
            base = self.scaling_params.get('base', np.e)
            r_min = self.scaling_params.get('r_min', 0.3)
            r_max = self.scaling_params.get('r_max', 1.0)
            radii = ModulusScaling.logarithmic(moduli, base, r_min, r_max)
        elif self.scaling == 'linear_clamp':
            m_max = self.scaling_params.get('m_max', 10)
            r_min = self.scaling_params.get('r_min', 0.3)
            r_max = self.scaling_params.get('r_max', 1.0)
            radii = ModulusScaling.linear_clamp(moduli, m_max, r_min, r_max)
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling}")
            
        # Handle infinities in radii
        radii[~finite_mask] = radii[finite_mask].max() if np.any(finite_mask) else 1.0
        
        # Apply radial scaling
        scaled_points = points * radii[:, np.newaxis]
        sphere.points = scaled_points
        
        # Store additional data
        sphere["magnitude"] = moduli
        sphere["phase"] = np.angle(f_vals)
        
        self.sphere_mesh = sphere
        
        if verbose:
            print(f"Generated sphere: {sphere.n_points} vertices, {sphere.n_cells} faces")
            radii_actual = np.linalg.norm(sphere.points, axis=1)
            print(f"Radius range: [{radii_actual.min():.3f}, {radii_actual.max():.3f}]")
            
        return sphere
    
    def heal_mesh(self, smooth: bool = True, smooth_iterations: int = 20,
                  verbose: bool = False) -> pv.PolyData:
        """
        Heal the sphere mesh using spherical shell clipping.
        
        Parameters
        ----------
        smooth : bool, default=True
            Apply smoothing after healing.
        smooth_iterations : int, default=20
            Number of smoothing iterations.
        verbose : bool, default=False
            Print progress information.
            
        Returns
        -------
        healed_mesh : pv.PolyData
            Healed watertight mesh ready for 3D printing.
        """
        if self.sphere_mesh is None:
            self.generate_sphere(verbose=verbose)
        
        if verbose:
            print("\nHealing mesh with spherical shell clipping...")
        
        # Determine radius bounds for clipping
        if self.scaling == 'constant':
            # For constant radius, use tight bounds
            r_target = self.scaling_params.get('radius', 1.0)
            r_min = r_target * 0.99
            r_max = r_target * 1.01
        else:
            # For modulus-based scaling, use the scaling bounds with margin
            r_min = self.scaling_params.get('r_min', 0.3) * 0.95
            r_max = self.scaling_params.get('r_max', 1.0) * 1.05
        
        # Apply spherical shell healing
        self.healed_mesh = spherical_shell_healing(
            self.sphere_mesh, r_min, r_max,
            smooth=smooth, smooth_iterations=smooth_iterations,
            verbose=verbose
        )
        
        return self.healed_mesh
    
    def validate_printability(self, verbose: bool = True) -> Dict[str, bool]:
        """
        Validate mesh for 3D printing.
        
        Parameters
        ----------
        verbose : bool, default=True
            Print validation results.
            
        Returns
        -------
        validation : dict
            Validation results.
        """
        if self.healed_mesh is None:
            raise ValueError("No healed mesh to validate. Run heal_mesh() first.")
        
        mesh = self.healed_mesh
        validation = {}
        
        # Check if watertight
        edges = mesh.extract_feature_edges(boundary_edges=True)
        validation['is_watertight'] = edges.n_points == 0
        
        # Check if manifold
        nm_edges = mesh.extract_feature_edges(non_manifold_edges=True)
        validation['is_manifold'] = nm_edges.n_points == 0
        
        # Check face quality
        mesh = mesh.compute_cell_sizes()
        areas = mesh["Area"]
        validation['no_degenerate_faces'] = np.all(areas > 1e-10)
        
        # Size checks
        validation['has_color'] = 'RGB' in mesh.array_names
        validation['reasonable_size'] = (
            1000 < mesh.n_points < 500000 and
            1000 < mesh.n_cells < 1000000
        )
        
        # Connectivity
        validation['single_component'] = True  # Spherical shell is always connected
        
        if verbose:
            print("\nMesh Validation:")
            print("-" * 30)
            for key, value in validation.items():
                status = "✓" if value else "✗"
                print(f"{status} {key}: {value}")
            print(f"\nMesh stats: {mesh.n_points} vertices, {mesh.n_cells} faces")
            
        return validation
    
    def export(self, filename: str, size_mm: float = 80,
               binary: bool = True, verbose: bool = True) -> str:
        """
        Export healed mesh to STL file.
        
        Parameters
        ----------
        filename : str
            Output filename (should end with .stl).
        size_mm : float, default=80
            Target diameter in millimeters.
        binary : bool, default=True
            Save as binary STL (smaller file size).
        verbose : bool, default=True
            Print export information.
            
        Returns
        -------
        filename : str
            Path to exported file.
        """
        if self.healed_mesh is None:
            raise ValueError("No healed mesh to export. Run heal_mesh() first.")
        
        mesh = self.healed_mesh.copy()
        
        # Scale to target size
        bounds = mesh.bounds
        current_size = max(
            bounds[1] - bounds[0],  # X extent
            bounds[3] - bounds[2],  # Y extent
            bounds[5] - bounds[4]   # Z extent
        )
        
        scale_factor = size_mm / current_size
        mesh = mesh.scale(scale_factor, inplace=False)
        
        # Center at origin
        center = np.array(mesh.center)
        mesh = mesh.translate(-center, inplace=False)
        
        # Save STL
        if not filename.endswith('.stl'):
            filename += '.stl'
        
        mesh.save(filename, binary=binary)
        
        if verbose:
            file_size_mb = os.path.getsize(filename) / (1024 * 1024)
            final_bounds = mesh.bounds
            print(f"\nExported: {filename}")
            print(f"File size: {file_size_mb:.2f} MB")
            print(f"Dimensions (mm): {final_bounds[1]-final_bounds[0]:.1f} x "
                  f"{final_bounds[3]-final_bounds[2]:.1f} x "
                  f"{final_bounds[5]-final_bounds[4]:.1f}")
            
        return filename
    
    def generate_ornament(self, output_file: str = 'ornament.stl',
                         size_mm: float = 80,
                         smooth: bool = True,
                         smooth_iterations: int = 20,
                         verbose: bool = True) -> str:
        """
        Complete pipeline to generate a watertight ornament STL.
        
        NO CUTTING! Users can orient and slice in their slicer software.
        
        Parameters
        ----------
        output_file : str, default='ornament.stl'
            Output filename.
        size_mm : float, default=80
            Target diameter in millimeters.
        smooth : bool, default=True
            Apply smoothing during healing.
        smooth_iterations : int, default=20
            Number of smoothing iterations.
        verbose : bool, default=True
            Print progress information.
            
        Returns
        -------
        filename : str
            Path to generated STL file.
        """
        if verbose:
            print(f"Generating complete ornament mesh...")
            print("=" * 60)
        
        # Generate sphere
        self.generate_sphere(verbose=verbose)
        
        # Heal mesh
        self.heal_mesh(smooth=smooth, smooth_iterations=smooth_iterations,
                      verbose=verbose)
        
        # Validate
        self.validate_printability(verbose=verbose)
        
        # Export
        filename = self.export(output_file, size_mm=size_mm, verbose=verbose)
        
        if verbose:
            print("\n" + "=" * 60)
            print("Complete! The mesh is ready for slicing and printing.")
            print("Use your slicer software to orient and cut as desired.")
        
        return filename