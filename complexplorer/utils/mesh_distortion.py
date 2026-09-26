"""Mesh distortion utilities for modulus-based scaling.

This module provides shared functionality for distorting meshes based on
complex function moduli, used by both visualization and STL export.
"""

from typing import Any

import numpy as np

from ..core.scaling import apply_scaling_mode


def apply_modulus_distortion(
    mesh_points: np.ndarray,
    moduli: np.ndarray,
    scaling_mode: str = "arctan",
    scaling_params: dict[str, Any] | None = None,
    handle_infinities: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
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

    radii = apply_scaling_mode(moduli, scaling_mode, scaling_params)

    # Apply radial scaling
    scaled_points = mesh_points * radii[:, np.newaxis]

    return scaled_points, radii


def get_default_scaling_params(scaling_mode: str, for_stl: bool = False) -> dict[str, Any]:
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
        # Tighter bounds for 3D printing. The depth (r_min) is 0.2, matching every entry in
        # SCALING_PRESETS; the STL defaults alone used to say 0.5, which made the relief half as
        # deep as any named preset for no stated reason. A relief is a star-shaped solid about the
        # origin, so a depth of 0.2 is still 0.4 of the radius thick through the centre.
        defaults = {
            "constant": {"radius": 1.0},
            "linear": {"scale": 0.1},
            "arctan": {"r_min": 0.2, "r_max": 1.0},
            "logarithmic": {"base": np.e, "r_min": 0.2, "r_max": 1.0},
            "log_mixture": {"scale": 2.0, "boost": 3.0, "weight": 0.5, "r_min": 0.2, "r_max": 1.0},
            "linear_clamp": {"m_max": 10, "r_min": 0.2, "r_max": 1.0},
            "power": {"exponent": 0.5, "r_min": 0.2, "r_max": 1.0},
            "sigmoid": {"steepness": 2.0, "center": 1.0, "r_min": 0.2, "r_max": 1.0},
            "adaptive": {"low_percentile": 10, "high_percentile": 90, "r_min": 0.2, "r_max": 1.0},
            "hybrid": {"transition": 1.0, "r_min": 0.2, "r_max": 1.0},
        }
    else:
        # Wider bounds for visualization
        defaults = {
            "constant": {"radius": 1.0},
            "linear": {"scale": 0.1},
            "arctan": {"r_min": 0.5, "r_max": 1.5},
            "logarithmic": {"base": np.e, "r_min": 0.5, "r_max": 1.5},
            "log_mixture": {"scale": 2.0, "boost": 3.0, "weight": 0.5, "r_min": 0.3, "r_max": 1.5},
            "linear_clamp": {"m_max": 10, "r_min": 0.5, "r_max": 1.5},
            "power": {"exponent": 0.5, "r_min": 0.3, "r_max": 1.5},
            "sigmoid": {"steepness": 2.0, "center": 1.0, "r_min": 0.3, "r_max": 1.5},
            "adaptive": {"low_percentile": 10, "high_percentile": 90, "r_min": 0.3, "r_max": 1.5},
            "hybrid": {"transition": 1.0, "r_min": 0.3, "r_max": 1.5},
        }

    return defaults.get(scaling_mode, {})


__all__ = [
    "apply_modulus_distortion",
    "get_default_scaling_params",
]
