"""Mathematical functions for complex visualization.

This module provides various mathematical functions used throughout
the library for complex function visualization.
"""

from typing import Union, Tuple, Optional
import numpy as np
from complexplorer.utils.validation import validate_array_shape


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
    # Suppress warnings for edge cases
    with np.errstate(invalid='ignore'):
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
    
    # Suppress warnings for points at infinity
    with np.errstate(divide='ignore', invalid='ignore'):
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



def sigmoid(x: Union[float, np.ndarray],
            center: float = 0.0,
            scale: float = 1.0) -> Union[float, np.ndarray]:
    """Sigmoid function for smooth transitions.
    
    Maps input values to [0, 1] using a logistic function.
    Useful for smooth modulus-based transitions in colormaps.
    
    Parameters
    ----------
    x : float or np.ndarray
        Input values.
    center : float, optional
        Center point of the sigmoid.
    scale : float, optional
        Scale factor controlling steepness.
        
    Returns
    -------
    float or np.ndarray
        Sigmoid values in [0, 1].
        
    Examples
    --------
    >>> sigmoid(0.0)
    0.5
    >>> sigmoid(10.0)
    0.9999546...
    """
    z = (x - center) / scale
    return 1.0 / (1.0 + np.exp(-z))


def circular_interpolate(theta1: float, theta2: float, 
                        t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Interpolate between two angles on a circle.
    
    Performs shortest-path interpolation between two angles,
    properly handling the circular wraparound at 2π.
    
    Parameters
    ----------
    theta1 : float
        Start angle in radians.
    theta2 : float
        End angle in radians.
    t : float or np.ndarray
        Interpolation parameter(s) in [0, 1].
        
    Returns
    -------
    float or np.ndarray
        Interpolated angle(s) in [0, 2π).
        
    Examples
    --------
    >>> circular_interpolate(0, np.pi/2, 0.5)
    0.7853981...  # π/4
    >>> circular_interpolate(7*np.pi/4, np.pi/4, 0.5)
    0.0  # Correctly interpolates across 0
    """
    # Convert to complex for circular interpolation
    z1 = np.exp(1j * theta1)
    z2 = np.exp(1j * theta2)
    
    # Linear interpolation in complex plane
    z_interp = (1 - t) * z1 + t * z2
    
    # Extract angle and ensure [0, 2π)
    result = np.angle(z_interp)
    if np.isscalar(result):
        if result < 0:
            result = 2 * np.pi + result
    else:
        result = np.asarray(result)
        mask = result < 0
        result[mask] = 2 * np.pi + result[mask]
    
    return result


# Backward compatibility
stereographic = stereographic_projection