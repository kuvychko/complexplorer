"""Mathematical functions for complex visualization.

This module provides various mathematical functions used throughout
the library for complex function visualization.
"""

from typing import Union, Tuple, Optional
import numpy as np
from ..utils.validation import validate_array_shape


def phase(z: Union[complex, np.ndarray]) -> Union[float, np.ndarray]:
    """Calculate phase (argument) of complex values in [0, 2π).
    
    The phase is the angle of the complex number in polar form,
    mapped to the interval [0, 2π) for consistent coloring.
    
    Parameters
    ----------
    z : complex or np.ndarray
        Complex value(s).
        
    Returns
    -------
    float or np.ndarray
        Phase values in [0, 2π).
        
    Examples
    --------
    >>> phase(1+0j)
    0.0
    >>> phase(1j)
    1.5707963267948966  # π/2
    >>> phase(-1+0j)
    3.141592653589793   # π
    """
    # Get phase in [-π, π]
    phi = np.angle(z)
    
    # Convert to [0, 2π)
    if np.isscalar(phi):
        if phi < 0:
            phi = 2 * np.pi + phi
    else:
        phi = np.asarray(phi)
        mask = phi < 0
        phi[mask] = 2 * np.pi + phi[mask]
    
    return phi


def sawtooth(x: Union[float, np.ndarray], 
             period: float = 1.0) -> Union[float, np.ndarray]:
    """Generate sawtooth wave with values in [0, 1).
    
    Creates a periodic sawtooth function that maps input values
    to the interval [0, 1). Used for creating periodic patterns
    in enhanced phase portraits.
    
    Parameters
    ----------
    x : float or np.ndarray
        Input values.
    period : float, optional
        Period of the sawtooth wave.
        
    Returns
    -------
    float or np.ndarray
        Sawtooth values in [0, 1).
        
    Examples
    --------
    >>> sawtooth(0.5)
    0.5
    >>> sawtooth(1.5)
    0.5
    >>> sawtooth(2.3, period=2.0)
    0.15
    """
    return np.mod(x / period, 1.0)


def sawtooth_log(x: Union[float, np.ndarray],
                 base: float = np.e) -> Union[float, np.ndarray]:
    """Generate logarithmic sawtooth wave.
    
    Applies logarithm before creating sawtooth pattern.
    Useful for visualizing functions with wide range of moduli.
    
    Parameters
    ----------
    x : float or np.ndarray
        Input values (must be positive).
    base : float, optional
        Logarithm base.
        
    Returns
    -------
    float or np.ndarray
        Sawtooth values in [0, 1).
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        if base == np.e:
            log_x = np.log(x)
        else:
            log_x = np.log(x) / np.log(base)
    
    result = np.mod(log_x, 1.0)
    
    # Handle x=0 case
    if np.isscalar(x):
        if x == 0:
            result = 0.0
    else:
        result = np.asarray(result)
        result[x == 0] = 0.0
    
    return result


def sawtooth_legacy(x: Union[float, np.ndarray],
                    log_base: Optional[float] = None) -> Union[float, np.ndarray]:
    """Legacy sawtooth function for backward compatibility.
    
    This function uses ceil(x) - x formula which gives values
    in (0, 1] instead of [0, 1).
    
    Parameters
    ----------
    x : float or np.ndarray
        Input values.
    log_base : float, optional
        If provided, apply logarithm first.
        
    Returns
    -------
    float or np.ndarray
        Sawtooth values in (0, 1].
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        if log_base is not None:
            x = np.log(x) / np.log(log_base)
        return np.ceil(x) - x


def stereographic_projection(z: Union[complex, np.ndarray],
                           project_from_north: bool = False) -> np.ndarray:
    """Map complex plane to Riemann sphere via stereographic projection.
    
    The stereographic projection maps the complex plane to a sphere
    with the point at infinity mapped to one of the poles.
    
    Parameters
    ----------
    z : complex or np.ndarray
        Complex value(s) to project.
    project_from_north : bool, optional
        If True, project from north pole (infinity at north).
        If False, project from south pole (infinity at south).
        Default is False for consistent zero/pole visualization.
        
    Returns
    -------
    np.ndarray
        Array of shape (..., 3) with (x, y, z) coordinates on sphere.
        
    Notes
    -----
    The formulas for projection from south pole are:
    - x = 2Re(z) / (1 + |z|²)
    - y = 2Im(z) / (1 + |z|²)  
    - z = (|z|² - 1) / (1 + |z|²)
    
    Examples
    --------
    >>> stereographic_projection(0+0j)
    array([ 0.,  0., -1.])  # South pole
    >>> stereographic_projection(1+0j)
    array([ 1.,  0.,  0.])  # On equator
    """
    z = np.asarray(z)
    scalar_input = z.ndim == 0
    
    X = np.real(z)
    Y = np.imag(z)
    denominator = 1 + X**2 + Y**2
    
    x = 2 * X / denominator
    y = 2 * Y / denominator
    z_coord = (-1 + X**2 + Y**2) / denominator
    
    if project_from_north:
        z_coord = z_coord
    else:
        z_coord = -z_coord
    
    # Stack coordinates
    if scalar_input:
        return np.array([x, y, z_coord])
    else:
        return np.stack([x, y, z_coord], axis=-1)


def inverse_stereographic(x: Union[float, np.ndarray], 
                         y: Union[float, np.ndarray], 
                         z: Union[float, np.ndarray],
                         project_from_north: bool = False) -> Union[complex, np.ndarray]:
    """Inverse stereographic projection from sphere to complex plane.
    
    Maps points on the Riemann sphere back to the complex plane.
    
    Parameters
    ----------
    x, y, z : float or np.ndarray
        Coordinates on the unit sphere.
    project_from_north : bool, optional
        Must match the projection direction used.
        
    Returns
    -------
    complex or np.ndarray
        Complex values.
        
    Notes
    -----
    For projection from south pole:
    - Re(w) = x / (1 + z)
    - Im(w) = y / (1 + z)
    
    Points at the pole (z = -1 for south, z = 1 for north) 
    map to infinity.
    """
    # Convert to arrays for uniform handling
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    scalar_input = x.ndim == 0
    
    if not project_from_north:
        z = -z
    
    # Handle division by zero at pole
    with np.errstate(divide='ignore', invalid='ignore'):
        denominator = 1 - z
        real_part = x / denominator
        imag_part = y / denominator
    
    # Set infinities for points at pole
    if scalar_input:
        if np.abs(denominator) < 1e-10:
            real_part = np.inf
            imag_part = np.inf
    else:
        at_pole = np.abs(denominator) < 1e-10
        real_part = real_part.copy()
        imag_part = imag_part.copy()
        real_part[at_pole] = np.inf
        imag_part[at_pole] = np.inf
    
    result = real_part + 1j * imag_part
    return complex(result) if scalar_input else result


# Backward compatibility
stereographic = stereographic_projection