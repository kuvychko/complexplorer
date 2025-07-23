"""
Mesh utilities for complex function visualization on the Riemann sphere.

This module provides sphere meshing using rectangular (latitude-longitude) grid:
- Good rendering quality in PyVista due to structured grid format
- Smooth appearance with standard shading
- May have slight distortion near poles but generally produces good results
- Recommended for all visualizations
"""

import numpy as np
import pyvista as pv
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .domain import Domain


class RectangularSphereGenerator:
    """
    Generate sphere with rectangular (latitude-longitude) grid.
    
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
        
    def generate(self) -> pv.PolyData:
        """
        Generate the sphere mesh.
        
        Returns
        -------
        mesh : pv.PolyData
            PyVista mesh object representing the sphere.
        """
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


def stereographic_projection(x: np.ndarray, y: np.ndarray, z: np.ndarray, 
                           from_north: bool = True) -> np.ndarray:
    """
    Apply stereographic projection from sphere to complex plane.
    
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
    """
    if from_north:
        # Project from north pole
        denominator = 1 - z
        # Avoid division by zero at north pole
        denominator = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)
        w = (x + 1j * y) / denominator
    else:
        # Project from south pole  
        denominator = 1 + z
        # Avoid division by zero at south pole
        denominator = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)
        w = (x + 1j * y) / denominator
        
    return w


def inverse_stereographic(w: np.ndarray, to_north: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply inverse stereographic projection from complex plane to sphere.
    
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


class ModulusScaling:
    """
    Methods for scaling sphere radius based on function modulus.
    
    These methods map the modulus |f(z)| to a radius value, allowing
    the Riemann sphere to show both phase and magnitude information.
    """
    
    @staticmethod
    def constant(moduli: np.ndarray, radius: float = 1.0) -> np.ndarray:
        """Constant radius regardless of modulus."""
        return np.full_like(moduli, radius, dtype=float)
    
    @staticmethod
    def linear(moduli: np.ndarray, scale: float = 0.1) -> np.ndarray:
        """Linear scaling: r = 1 + scale * |f(z)|."""
        return 1.0 + scale * moduli
    
    @staticmethod
    def arctan(moduli: np.ndarray, r_min: float = 0.5, r_max: float = 1.5) -> np.ndarray:
        """
        Smooth scaling using arctangent.
        
        Maps [0, ∞) to [r_min, r_max] smoothly.
        """
        # Normalize modulus to [0, 1] using arctan
        normalized = (2/np.pi) * np.arctan(moduli)
        # Map to [r_min, r_max]
        return r_min + (r_max - r_min) * normalized
    
    @staticmethod  
    def logarithmic(moduli: np.ndarray, base: float = np.e, 
                    r_min: float = 0.5, r_max: float = 1.5) -> np.ndarray:
        """
        Logarithmic scaling for large dynamic range.
        
        Good for functions with exponential growth.
        """
        # Avoid log(0)
        safe_moduli = np.maximum(moduli, 1e-10)
        # Log scaling
        log_moduli = np.log(safe_moduli) / np.log(base)
        # Use sigmoid to map to [0, 1]
        normalized = 1 / (1 + np.exp(-log_moduli))
        # Map to [r_min, r_max]
        return r_min + (r_max - r_min) * normalized
    
    @staticmethod
    def linear_clamp(moduli: np.ndarray, m_max: float = 10,
                     r_min: float = 0.5, r_max: float = 1.5) -> np.ndarray:
        """
        Linear scaling with clamping.
        
        Linear up to m_max, then constant.
        """
        # Clamp moduli to [0, m_max]
        clamped = np.minimum(moduli, m_max)
        # Normalize to [0, 1]
        normalized = clamped / m_max
        # Map to [r_min, r_max]
        return r_min + (r_max - r_min) * normalized
    
    @staticmethod
    def power(moduli: np.ndarray, exponent: float = 0.5,
              r_min: float = 0.5, r_max: float = 1.5) -> np.ndarray:
        """
        Power scaling: r = r_min + (r_max - r_min) * (|f|/|f|_max)^exponent.
        
        Exponent < 1 compresses large values, > 1 expands them.
        """
        # Normalize by maximum modulus
        max_mod = np.max(moduli)
        if max_mod > 0:
            normalized = (moduli / max_mod) ** exponent
        else:
            normalized = np.zeros_like(moduli)
        # Map to [r_min, r_max]
        return r_min + (r_max - r_min) * normalized