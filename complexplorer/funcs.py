import numpy as np

"""
Module containing various support functions.

Functions:
----------

- `phase`: return a phase of complex input mapped to [0, 2*pi) interval.

- `sawtooth`: return a sawtooth wave of input x.

- `stereographic`: return a (x,y,z) tuple corresponding to stereographic projection of complex input z.

"""

def phase(z: complex):
    """
    Return a phase of complex input mapped to [0, 2*pi) interval.
    
    Parameters:
    -----------
    z: complex
        Input complex number.

    Returns:
    --------
    float
        Phase of input z mapped to [0, 2*pi) interval.
    """

    phi = np.angle(z)
    # convert phase angles from [pi, -pi) which is the output of numpy.angle to [0, 2*pi)
    phi[phi<0] = 2*np.pi + phi[phi<0]
    return phi

def sawtooth(x, log_base=None):
    """
    Return a sawtooth wave of input x.

    Parameters:
    -----------
    x: np.ndarray
        Input array.
    log_base: float
        If not None, the input x is first converted to its logarithm in log_base.
        Default is None.

    Returns:
    --------
    np.ndarray
        Sawtooth wave of input x.
    """
    # setting numpy to ignore divide by zero and invalid input errors locally
    with np.errstate(divide='ignore', invalid='ignore'):
        if log_base is not None:
            x = np.log(x)
            x = x / np.log(log_base) # converting log from natural to log_base
        return np.ceil(x) - x

def stereographic(z, project_from_north=False):
    """
    Return a (x,y,z) tuple corresponding to stereographic projection of complex input z.

    Parameters:
    -----------
    z: complex
        Input complex number.
    project_from_north: bool
        If True, stereographic projection is performed from the north pole. 
        If False, stereographic projection is performed from the south pole.
        Default is False (this is non-standard, but helps with consistent 
        interpretation of zeros and poles via phase portrait color maps).

    Returns:
    --------
    (x,y,z): tuple
        Tuple of floats corresponding to stereographic projection of z.
    """
    X = np.real(z)
    Y = np.imag(z)
    x = 2*X / (1 + X**2 + Y**2)
    y = 2*Y / (1 + X**2 + Y**2)
    z = (-1 + X**2 + Y**2) / (1 + X**2 + Y**2)
    if project_from_north:
        return (x, y, z)
    else:
        return (x, y, -z)
