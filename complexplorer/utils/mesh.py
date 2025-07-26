"""Mesh generation utilities for complex function visualization.

This module provides utilities for generating meshes on the Riemann sphere
using rectangular (latitude-longitude) grids.
"""

from typing import Optional, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..core.domain import Domain

# Only import PyVista if available
try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    pv = None


class RectangularSphereGenerator:
    """Generate sphere with rectangular (latitude-longitude) grid.
    
    This approach creates a structured grid that renders well in PyVista
    and provides smooth appearance with standard shading algorithms.
    
    Parameters
    ----------
    radius : float, default=1.0
        Sphere radius.
    n_theta : int, default=100
        Number of latitude divisions (from pole to pole).
    n_phi : int, default=100  
        Number of longitude divisions (around equator).
    avoid_poles : bool, default=True
        If True, slightly offset from exact poles to avoid singularities.
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
        
    def generate(self) -> 'pv.PolyData':
        """Generate the sphere mesh.
        
        Returns
        -------
        mesh : pv.PolyData
            PyVista mesh object representing the sphere.
            
        Raises
        ------
        ImportError
            If PyVista is not installed.
        """
        if not HAS_PYVISTA:
            raise ImportError("PyVista is required for mesh generation. "
                            "Install with: pip install pyvista")
        
        # Create angle arrays
        if self.avoid_poles:
            # Small offset from poles to avoid singularities
            theta = np.linspace(0.01, np.pi - 0.01, self.n_theta)
        else:
            theta = np.linspace(0, np.pi, self.n_theta)
        
        phi = np.linspace(0, 2 * np.pi, self.n_phi)
        
        # Create meshgrid
        THETA, PHI = np.meshgrid(theta, phi)
        
        # Convert to Cartesian coordinates
        X = self.radius * np.sin(THETA) * np.cos(PHI)
        Y = self.radius * np.sin(THETA) * np.sin(PHI) 
        Z = self.radius * np.cos(THETA)
        
        # If domain is specified, filter points
        if self.domain is not None:
            # Project points to complex plane
            w = sphere_to_complex(X.ravel(), Y.ravel(), Z.ravel())
            
            # Check which points are in domain
            in_domain = self.domain.contains(w)
            in_domain = in_domain.reshape(X.shape)
            
            # Mark out-of-domain points as NaN
            X = np.where(in_domain, X, np.nan)
            Y = np.where(in_domain, Y, np.nan)
            Z = np.where(in_domain, Z, np.nan)
        
        # Create structured grid
        grid = pv.StructuredGrid(X, Y, Z)
        
        # Convert to PolyData
        mesh = grid.extract_surface()
        
        # Remove any cells with NaN vertices if domain filtering was applied
        if self.domain is not None:
            # Get point data
            points = mesh.points
            nan_mask = np.any(np.isnan(points), axis=1)
            if np.any(nan_mask):
                # Remove cells that reference NaN points
                cells_to_remove = []
                for i in range(mesh.n_cells):
                    cell_points = mesh.get_cell(i).points
                    if np.any(np.isnan(cell_points)):
                        cells_to_remove.append(i)
                
                if cells_to_remove:
                    mesh = mesh.remove_cells(cells_to_remove)
        
        return mesh


def sphere_to_complex(x: np.ndarray, y: np.ndarray, z: np.ndarray, 
                     from_north: bool = True) -> np.ndarray:
    """Apply stereographic projection from sphere to complex plane.
    
    Projects points from the unit sphere to the complex plane using
    stereographic projection from either north or south pole.
    
    Parameters
    ----------
    x, y, z : array_like
        Coordinates on the sphere.
    from_north : bool, default=True
        If True, project from north pole (0, 0, 1).
        If False, project from south pole (0, 0, -1).
        
    Returns
    -------
    w : ndarray
        Complex coordinates in the plane.
        
    Notes
    -----
    This is a convenience function that calls the inverse_stereographic
    function from core.functions with appropriate parameters.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    
    if from_north:
        # Project from north pole
        denominator = 1 - z
        # Handle division by zero at north pole
        with np.errstate(divide='ignore', invalid='ignore'):
            w = (x + 1j * y) / denominator
        # Points at north pole map to infinity
        if np.isscalar(w):
            if np.abs(denominator) < 1e-10:
                w = np.inf * (1 + 0j)
        else:
            at_pole = np.abs(denominator) < 1e-10
            w = np.where(at_pole, np.inf * (1 + 0j), w)
    else:
        # Project from south pole  
        denominator = 1 + z
        # Handle division by zero at south pole
        with np.errstate(divide='ignore', invalid='ignore'):
            w = (x + 1j * y) / denominator
        # Points at south pole map to infinity
        if np.isscalar(w):
            if np.abs(denominator) < 1e-10:
                w = np.inf * (1 + 0j)
        else:
            at_pole = np.abs(denominator) < 1e-10
            w = np.where(at_pole, np.inf * (1 + 0j), w)
        
    return w


def complex_to_sphere(w: np.ndarray, to_north: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply inverse stereographic projection from complex plane to sphere.
    
    Projects points from the complex plane to the unit sphere using
    inverse stereographic projection.
    
    Parameters
    ----------
    w : array_like
        Complex coordinates in the plane.
    to_north : bool, default=True
        If True, inverse of projection from north pole.
        If False, inverse of projection from south pole.
        
    Returns
    -------
    x, y, z : ndarray
        Coordinates on the unit sphere.
        
    Notes
    -----
    This is a convenience function that wraps the stereographic_projection
    function from core.functions for consistency with legacy API.
    """
    # Convert to arrays
    w = np.asarray(w)
    u = np.real(w)
    v = np.imag(w)
    
    # |w|^2
    w_squared = u**2 + v**2
    
    if to_north:
        # Inverse of projection from north pole
        x = 2 * u / (1 + w_squared)
        y = 2 * v / (1 + w_squared)
        z = (w_squared - 1) / (1 + w_squared)
    else:
        # Inverse of projection from south pole
        x = 2 * u / (1 + w_squared)
        y = 2 * v / (1 + w_squared)
        z = (1 - w_squared) / (1 + w_squared)
        
    return x, y, z


# Legacy API compatibility
inverse_stereographic = complex_to_sphere
stereographic_projection = sphere_to_complex