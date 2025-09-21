"""Mathematical special functions for complex visualization.

This module provides a collection of special mathematical functions
commonly used in complex analysis, with proper branch cut handling
and documentation of their mathematical properties.

Functions are provided as a convenience wrapper around scipy.special
when available, with fallback implementations for basic functions.
"""

from typing import Union, Optional
import numpy as np
import warnings


# Try to import scipy special functions
try:
    import scipy.special as sp
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not available. Some special functions will be limited or unavailable.")


def gamma(z: Union[complex, np.ndarray]) -> Union[complex, np.ndarray]:
    """Complex gamma function Γ(z).
    
    The gamma function extends the factorial to complex numbers:
    Γ(n) = (n-1)! for positive integers n.
    
    Parameters
    ----------
    z : complex or np.ndarray
        Complex argument(s).
        
    Returns
    -------
    complex or np.ndarray
        Gamma function value(s).
        
    Notes
    -----
    The gamma function has poles at non-positive integers.
    Branch cuts are handled automatically by the underlying implementation.
    """
    if HAS_SCIPY:
        return sp.gamma(z)
    else:
        raise NotImplementedError("Gamma function requires scipy")


def zeta(z: Union[complex, np.ndarray]) -> Union[complex, np.ndarray]:
    """Riemann zeta function ζ(z).
    
    The Riemann zeta function is defined as:
    ζ(s) = Σ(n=1 to ∞) 1/n^s for Re(s) > 1
    
    Extended to the entire complex plane by analytic continuation.
    
    Parameters
    ----------
    z : complex or np.ndarray
        Complex argument(s).
        
    Returns
    -------
    complex or np.ndarray
        Zeta function value(s).
        
    Notes
    -----
    The zeta function has a pole at z=1 and zeros at negative even integers
    (trivial zeros) and along the critical strip 0 < Re(z) < 1 (non-trivial zeros).
    """
    if HAS_SCIPY:
        return sp.zeta(z)
    else:
        raise NotImplementedError("Zeta function requires scipy")


def digamma(z: Union[complex, np.ndarray]) -> Union[complex, np.ndarray]:
    """Digamma (psi) function ψ(z).
    
    The digamma function is the logarithmic derivative of the gamma function:
    ψ(z) = d/dz ln(Γ(z)) = Γ'(z)/Γ(z)
    
    Parameters
    ----------
    z : complex or np.ndarray
        Complex argument(s).
        
    Returns
    -------
    complex or np.ndarray
        Digamma function value(s).
        
    Notes
    -----
    The digamma function has poles at non-positive integers.
    """
    if HAS_SCIPY:
        return sp.digamma(z)
    else:
        raise NotImplementedError("Digamma function requires scipy")


def erf(z: Union[complex, np.ndarray]) -> Union[complex, np.ndarray]:
    """Complex error function erf(z).
    
    The error function is related to the Gaussian distribution:
    erf(z) = (2/√π) ∫(0 to z) exp(-t²) dt
    
    Parameters
    ----------
    z : complex or np.ndarray
        Complex argument(s).
        
    Returns
    -------
    complex or np.ndarray
        Error function value(s).
    """
    if HAS_SCIPY:
        return sp.erf(z)
    else:
        # Basic approximation for demonstration
        # Not accurate for complex values far from real axis
        warnings.warn("Using approximate erf implementation. Install scipy for accurate results.")
        t = 1.0 / (1.0 + 0.5 * np.abs(z))
        tau = t * np.exp(-z * z - 1.26551223 +
                        t * (1.00002368 +
                        t * (0.37409196 +
                        t * (0.09678418 +
                        t * (-0.18628806 +
                        t * (0.27886807 +
                        t * (-1.13520398 +
                        t * (1.48851587 +
                        t * (-0.82215223 +
                        t * 0.17087277)))))))))
        return np.sign(z) * (1 - tau)


def airy_ai(z: Union[complex, np.ndarray]) -> Union[complex, np.ndarray]:
    """Airy function Ai(z).
    
    The Airy function Ai(z) is a solution to the Airy differential equation:
    d²w/dz² - zw = 0
    
    Parameters
    ----------
    z : complex or np.ndarray
        Complex argument(s).
        
    Returns
    -------
    complex or np.ndarray
        Airy Ai function value(s).
        
    Notes
    -----
    Ai(z) decays exponentially for positive real z and oscillates
    for negative real z.
    """
    if HAS_SCIPY:
        result = sp.airy(z)
        return result[0]  # Ai is the first component
    else:
        raise NotImplementedError("Airy functions require scipy")


def airy_bi(z: Union[complex, np.ndarray]) -> Union[complex, np.ndarray]:
    """Airy function Bi(z).
    
    The Airy function Bi(z) is the second solution to the Airy differential equation:
    d²w/dz² - zw = 0
    
    Parameters
    ----------
    z : complex or np.ndarray
        Complex argument(s).
        
    Returns
    -------
    complex or np.ndarray
        Airy Bi function value(s).
        
    Notes
    -----
    Bi(z) grows exponentially for positive real z and oscillates
    for negative real z.
    """
    if HAS_SCIPY:
        result = sp.airy(z)
        return result[2]  # Bi is the third component
    else:
        raise NotImplementedError("Airy functions require scipy")


def bessel_j(n: Union[int, float], z: Union[complex, np.ndarray]) -> Union[complex, np.ndarray]:
    """Bessel function of the first kind Jₙ(z).
    
    Parameters
    ----------
    n : int or float
        Order of the Bessel function.
    z : complex or np.ndarray
        Complex argument(s).
        
    Returns
    -------
    complex or np.ndarray
        Bessel J function value(s).
        
    Notes
    -----
    Bessel functions arise in problems with cylindrical symmetry.
    J₀(z) has zeros along the real axis at approximately 2.40, 5.52, 8.65, ...
    """
    if HAS_SCIPY:
        return sp.jv(n, z)
    else:
        raise NotImplementedError("Bessel functions require scipy")


def bessel_y(n: Union[int, float], z: Union[complex, np.ndarray]) -> Union[complex, np.ndarray]:
    """Bessel function of the second kind Yₙ(z).
    
    Parameters
    ----------
    n : int or float
        Order of the Bessel function.
    z : complex or np.ndarray
        Complex argument(s).
        
    Returns
    -------
    complex or np.ndarray
        Bessel Y function value(s).
        
    Notes
    -----
    Also called Neumann functions. They have a singularity at z=0.
    """
    if HAS_SCIPY:
        return sp.yv(n, z)
    else:
        raise NotImplementedError("Bessel functions require scipy")


def elliptic_k(m: Union[complex, np.ndarray]) -> Union[complex, np.ndarray]:
    """Complete elliptic integral of the first kind K(m).
    
    K(m) = ∫(0 to π/2) 1/√(1 - m sin²θ) dθ
    
    Parameters
    ----------
    m : complex or np.ndarray
        Parameter m (not the modulus k where m = k²).
        
    Returns
    -------
    complex or np.ndarray
        Elliptic integral K(m).
        
    Notes
    -----
    The function has a logarithmic singularity at m=1.
    """
    if HAS_SCIPY:
        return sp.ellipk(m)
    else:
        raise NotImplementedError("Elliptic integrals require scipy")


def elliptic_e(m: Union[complex, np.ndarray]) -> Union[complex, np.ndarray]:
    """Complete elliptic integral of the second kind E(m).
    
    E(m) = ∫(0 to π/2) √(1 - m sin²θ) dθ
    
    Parameters
    ----------
    m : complex or np.ndarray
        Parameter m (not the modulus k where m = k²).
        
    Returns
    -------
    complex or np.ndarray
        Elliptic integral E(m).
    """
    if HAS_SCIPY:
        return sp.ellipe(m)
    else:
        raise NotImplementedError("Elliptic integrals require scipy")


def jacobi_elliptic(u: Union[complex, np.ndarray], 
                   m: Union[float, complex]) -> tuple:
    """Jacobi elliptic functions sn(u,m), cn(u,m), dn(u,m).
    
    Parameters
    ----------
    u : complex or np.ndarray
        Argument.
    m : float or complex
        Parameter.
        
    Returns
    -------
    sn, cn, dn : tuple of complex or np.ndarray
        Jacobi elliptic function values.
        
    Notes
    -----
    These are doubly periodic functions that generalize trigonometric functions.
    When m=0: sn(u,0)=sin(u), cn(u,0)=cos(u), dn(u,0)=1
    When m=1: sn(u,1)=tanh(u), cn(u,1)=sech(u), dn(u,1)=sech(u)
    """
    if HAS_SCIPY:
        return sp.ellipj(u, m)
    else:
        raise NotImplementedError("Jacobi elliptic functions require scipy")


def lambert_w(z: Union[complex, np.ndarray], 
              k: int = 0) -> Union[complex, np.ndarray]:
    """Lambert W function (product logarithm).
    
    The Lambert W function is the inverse of f(w) = w*exp(w).
    It satisfies: W(z)*exp(W(z)) = z
    
    Parameters
    ----------
    z : complex or np.ndarray
        Complex argument(s).
    k : int, optional
        Branch index (default 0 for principal branch).
        
    Returns
    -------
    complex or np.ndarray
        Lambert W function value(s).
        
    Notes
    -----
    The function has a branch point at z = -1/e.
    The principal branch W₀(z) is real for z ≥ -1/e.
    """
    if HAS_SCIPY:
        return sp.lambertw(z, k)
    else:
        raise NotImplementedError("Lambert W function requires scipy")


def get_special_function(name: str):
    """Get a special function by name.
    
    Parameters
    ----------
    name : str
        Name of the function (e.g., 'gamma', 'zeta', 'erf').
        
    Returns
    -------
    callable
        The requested special function.
        
    Raises
    ------
    ValueError
        If function name is not recognized.
        
    Examples
    --------
    >>> func = get_special_function('gamma')
    >>> value = func(2.5)
    """
    functions = {
        'gamma': gamma,
        'zeta': zeta,
        'digamma': digamma,
        'psi': digamma,  # Alias
        'erf': erf,
        'airy_ai': airy_ai,
        'airy_bi': airy_bi,
        'elliptic_k': elliptic_k,
        'elliptic_e': elliptic_e,
        'lambert_w': lambert_w,
    }
    
    if name.lower() in functions:
        return functions[name.lower()]
    else:
        available = ', '.join(sorted(functions.keys()))
        raise ValueError(f"Unknown function: {name}. Available: {available}")


# Create convenience namespace for common functions
class special:
    """Namespace for special functions (similar to scipy.special)."""
    gamma = gamma
    zeta = zeta
    digamma = digamma
    psi = digamma  # Alias
    erf = erf
    airy_ai = airy_ai
    airy_bi = airy_bi
    bessel_j = bessel_j
    bessel_y = bessel_y
    elliptic_k = elliptic_k
    elliptic_e = elliptic_e
    jacobi_elliptic = jacobi_elliptic
    lambert_w = lambert_w


__all__ = [
    'gamma', 'zeta', 'digamma', 'erf',
    'airy_ai', 'airy_bi', 'bessel_j', 'bessel_y',
    'elliptic_k', 'elliptic_e', 'jacobi_elliptic',
    'lambert_w', 'get_special_function', 'special',
    'HAS_SCIPY'
]