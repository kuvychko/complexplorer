"""Mesh distortion utilities for modulus-based scaling.

This module provides shared functionality for distorting meshes based on
complex function moduli, used by both visualization and STL export.
"""

import numpy as np
import warnings
from typing import Tuple, Dict, Any, Optional, Callable

from ..core.scaling import ModulusScaling
from ..utils.validation import ValidationError


def apply_modulus_distortion(
    mesh_points: np.ndarray,
    moduli: np.ndarray,
    scaling_mode: str = 'arctan',
    scaling_params: Optional[Dict[str, Any]] = None,
    handle_infinities: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply modulus-based radial distortion to mesh points.
    
    This is the core function used by both Riemann sphere visualization
    and STL export to distort a sphere based on function moduli.
    
    Parameters
    ----------
    mesh_points : np.ndarray
        Original mesh points (N x 3).
    moduli : np.ndarray
        Modulus values |f(z)| for each point.
    scaling_mode : str, default='arctan'
        Scaling method name from ModulusScaling.
    scaling_params : dict, optional
        Parameters for the scaling method.
    handle_infinities : bool, default=True
        Replace infinite moduli with max finite value.
        
    Returns
    -------
    scaled_points : np.ndarray
        Distorted mesh points (N x 3).
    radii : np.ndarray
        Applied radius scaling factors.
    """
    if scaling_params is None:
        scaling_params = {}
    
    # Handle infinities in moduli
    moduli = np.asarray(moduli)
    finite_mask = np.isfinite(moduli)
    
    if handle_infinities and not np.all(finite_mask):
        # Replace infinities with max finite value
        if np.any(finite_mask):
            max_finite = np.max(moduli[finite_mask])
            moduli = moduli.copy()
            moduli[~finite_mask] = max_finite
        else:
            # All infinite - use 1.0
            moduli = np.ones_like(moduli)
    
    # Get scaling method
    if scaling_mode == 'custom':
        # Custom mode requires a function
        if 'scaling_func' not in scaling_params:
            raise ValidationError(
                "Custom mode requires 'scaling_func' in scaling_params"
            )
        radii = scaling_params['scaling_func'](moduli)
    else:
        # Use built-in scaling method
        scaling_method = getattr(ModulusScaling, scaling_mode, None)
        if scaling_method is None:
            raise ValidationError(
                f"Unknown scaling mode: {scaling_mode}. "
                f"Available: constant, linear, arctan, logarithmic, "
                f"linear_clamp, power, sigmoid, adaptive, hybrid, custom"
            )
        radii = scaling_method(moduli, **scaling_params)
    
    # Apply radial scaling
    scaled_points = mesh_points * radii[:, np.newaxis]
    
    return scaled_points, radii


def compute_riemann_sphere_distortion(
    sphere_mesh,
    func: Callable,
    scaling_mode: str = 'arctan',
    scaling_params: Optional[Dict[str, Any]] = None,
    from_north: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute distorted Riemann sphere from complex function.
    
    Parameters
    ----------
    sphere_mesh : mesh object
        Base sphere mesh with .points attribute.
    func : callable
        Complex function to evaluate.
    scaling_mode : str, default='arctan'
        Scaling method for modulus.
    scaling_params : dict, optional
        Parameters for scaling method.
    from_north : bool, default=True
        Use north pole stereographic projection.
        
    Returns
    -------
    scaled_points : np.ndarray
        Distorted mesh points.
    f_vals : np.ndarray
        Complex function values.
    radii : np.ndarray
        Applied scaling factors.
    """
    from ..core.functions import inverse_stereographic
    
    # Get sphere points
    points = sphere_mesh.points
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    # Apply stereographic projection to get complex values
    w = inverse_stereographic(x, y, z, project_from_north=from_north)
    
    # Evaluate function
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        f_vals = func(w)
    
    # Get moduli
    f_vals = np.asarray(f_vals)
    moduli = np.abs(f_vals)
    
    # Apply distortion
    scaled_points, radii = apply_modulus_distortion(
        points, moduli, scaling_mode, scaling_params
    )
    
    return scaled_points, f_vals, radii


def get_default_scaling_params(scaling_mode: str, for_stl: bool = False) -> Dict[str, Any]:
    """Get default parameters for a scaling mode.
    
    Parameters
    ----------
    scaling_mode : str
        Name of scaling method.
    for_stl : bool, default=False
        If True, return parameters suitable for STL export
        (tighter bounds for 3D printing).
        
    Returns
    -------
    dict
        Default parameters for the scaling method.
    """
    if for_stl:
        # Tighter bounds for 3D printing
        defaults = {
            'constant': {'radius': 1.0},
            'linear': {'scale': 0.1},
            'arctan': {'r_min': 0.5, 'r_max': 1.0},
            'logarithmic': {'base': np.e, 'r_min': 0.5, 'r_max': 1.0},
            'linear_clamp': {'m_max': 10, 'r_min': 0.5, 'r_max': 1.0},
            'power': {'exponent': 0.5, 'r_min': 0.5, 'r_max': 1.0},
            'sigmoid': {'steepness': 2.0, 'center': 1.0, 'r_min': 0.5, 'r_max': 1.0},
            'adaptive': {'low_percentile': 10, 'high_percentile': 90, 'r_min': 0.5, 'r_max': 1.0},
            'hybrid': {'transition': 1.0, 'r_min': 0.5, 'r_max': 1.0},
        }
    else:
        # Wider bounds for visualization
        defaults = {
            'constant': {'radius': 1.0},
            'linear': {'scale': 0.1},
            'arctan': {'r_min': 0.5, 'r_max': 1.5},
            'logarithmic': {'base': np.e, 'r_min': 0.5, 'r_max': 1.5},
            'linear_clamp': {'m_max': 10, 'r_min': 0.5, 'r_max': 1.5},
            'power': {'exponent': 0.5, 'r_min': 0.3, 'r_max': 1.5},
            'sigmoid': {'steepness': 2.0, 'center': 1.0, 'r_min': 0.3, 'r_max': 1.5},
            'adaptive': {'low_percentile': 10, 'high_percentile': 90, 'r_min': 0.3, 'r_max': 1.5},
            'hybrid': {'transition': 1.0, 'r_min': 0.3, 'r_max': 1.5},
        }
    
    return defaults.get(scaling_mode, {})


__all__ = [
    'apply_modulus_distortion',
    'compute_riemann_sphere_distortion', 
    'get_default_scaling_params'
]