"""Modulus scaling methods for complex function visualization.

This module provides various methods to map the modulus |f(z)| to a radius value,
allowing visualizations to show both phase and magnitude information effectively.
"""

import numpy as np
from typing import Callable

__all__ = ['ModulusScaling', 'SCALING_PRESETS', 'get_scaling_preset']


class ModulusScaling:
    """Collection of modulus scaling methods for visualization.
    
    These methods map the modulus |f(z)| to a radius value, allowing
    visualizations to show both phase and magnitude information.
    """
    
    @staticmethod
    def constant(moduli: np.ndarray, radius: float = 1.0) -> np.ndarray:
        """Constant radius regardless of modulus.
        
        Parameters
        ----------
        moduli : np.ndarray
            Modulus values |f(z)|.
        radius : float, default=1.0
            Constant radius value.
            
        Returns
        -------
        np.ndarray
            Array of constant radius values.
        """
        return np.full_like(moduli, radius, dtype=float)
    
    @staticmethod
    def linear(moduli: np.ndarray, scale: float = 0.1) -> np.ndarray:
        """Linear scaling: r = 1 + scale * |f(z)|.
        
        Parameters
        ----------
        moduli : np.ndarray
            Modulus values |f(z)|.
        scale : float, default=0.1
            Scaling factor.
            
        Returns
        -------
        np.ndarray
            Linearly scaled radius values.
        """
        return 1.0 + scale * moduli
    
    @staticmethod
    def arctan(moduli: np.ndarray, r_min: float = 0.5, r_max: float = 1.5) -> np.ndarray:
        """Smooth scaling using arctangent.
        
        Maps [0, ∞) to [r_min, r_max] smoothly.
        
        Parameters
        ----------
        moduli : np.ndarray
            Modulus values |f(z)|.
        r_min : float, default=0.5
            Minimum radius value.
        r_max : float, default=1.5
            Maximum radius value.
            
        Returns
        -------
        np.ndarray
            Smoothly scaled radius values.
        """
        # Normalize modulus to [0, 1] using arctan
        normalized = (2/np.pi) * np.arctan(moduli)
        # Map to [r_min, r_max]
        return r_min + (r_max - r_min) * normalized
    
    @staticmethod  
    def logarithmic(moduli: np.ndarray, base: float = np.e, 
                    r_min: float = 0.5, r_max: float = 1.5) -> np.ndarray:
        """Logarithmic scaling for large dynamic range.
        
        Good for functions with exponential growth.
        
        Parameters
        ----------
        moduli : np.ndarray
            Modulus values |f(z)|.
        base : float, default=e
            Logarithm base.
        r_min : float, default=0.5
            Minimum radius value.
        r_max : float, default=1.5
            Maximum radius value.
            
        Returns
        -------
        np.ndarray
            Logarithmically scaled radius values.
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
        """Linear scaling with clamping.
        
        Linear up to m_max, then constant.
        
        Parameters
        ----------
        moduli : np.ndarray
            Modulus values |f(z)|.
        m_max : float, default=10
            Maximum modulus value before clamping.
        r_min : float, default=0.5
            Minimum radius value.
        r_max : float, default=1.5
            Maximum radius value.
            
        Returns
        -------
        np.ndarray
            Linearly scaled and clamped radius values.
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
        """Power scaling: r = r_min + (r_max - r_min) * (|f|/|f|_max)^exponent.
        
        Exponent < 1 compresses large values, > 1 expands them.
        
        Parameters
        ----------
        moduli : np.ndarray
            Modulus values |f(z)|.
        exponent : float, default=0.5
            Power exponent.
        r_min : float, default=0.5
            Minimum radius value.
        r_max : float, default=1.5
            Maximum radius value.
            
        Returns
        -------
        np.ndarray
            Power-scaled radius values.
        """
        # Normalize by maximum modulus
        max_mod = np.max(moduli)
        if max_mod > 0:
            normalized = (moduli / max_mod) ** exponent
        else:
            normalized = np.zeros_like(moduli)
        # Map to [r_min, r_max]
        return r_min + (r_max - r_min) * normalized
    
    @staticmethod
    def custom(moduli: np.ndarray, scaling_func: Callable[[np.ndarray], np.ndarray],
               r_min: float = 0.5, r_max: float = 1.5) -> np.ndarray:
        """Custom scaling function.
        
        Parameters
        ----------
        moduli : np.ndarray
            Modulus values |f(z)|.
        scaling_func : callable
            User-defined function that maps moduli to [0, 1].
        r_min : float, default=0.5
            Minimum radius value.
        r_max : float, default=1.5
            Maximum radius value.
            
        Returns
        -------
        np.ndarray
            Custom-scaled radius values.
        """
        # Apply custom function and clip to [0, 1]
        normalized = np.clip(scaling_func(moduli), 0, 1)
        # Map to [r_min, r_max]
        return r_min + (r_max - r_min) * normalized
    
    @staticmethod
    def sigmoid(moduli: np.ndarray, steepness: float = 2.0, center: float = 1.0,
                r_min: float = 0.5, r_max: float = 1.5) -> np.ndarray:
        """Sigmoid (S-curve) scaling.
        
        Provides smooth transition with adjustable steepness and center.
        Good general-purpose scaling for most functions.
        
        Parameters
        ----------
        moduli : np.ndarray
            Modulus values |f(z)|.
        steepness : float, default=2.0
            Controls transition sharpness (higher = steeper).
        center : float, default=1.0
            Center of transition (where r ≈ (r_min + r_max) / 2).
        r_min : float, default=0.5
            Minimum radius value.
        r_max : float, default=1.5
            Maximum radius value.
            
        Returns
        -------
        np.ndarray
            Sigmoid-scaled radius values.
        """
        # Sigmoid function
        normalized = 1 / (1 + np.exp(-steepness * (moduli - center)))
        # Map to [r_min, r_max]
        return r_min + (r_max - r_min) * normalized
    
    @staticmethod
    def adaptive(moduli: np.ndarray, low_percentile: float = 10, high_percentile: float = 90,
                 r_min: float = 0.5, r_max: float = 1.5) -> np.ndarray:
        """Adaptive percentile-based scaling.
        
        Automatically adjusts to data range, ignoring outliers.
        Excellent for unknown functions or those with extreme values.
        
        Parameters
        ----------
        moduli : np.ndarray
            Modulus values |f(z)|.
        low_percentile : float, default=10
            Lower percentile for mapping to r_min.
        high_percentile : float, default=90
            Upper percentile for mapping to r_max.
        r_min : float, default=0.5
            Minimum radius value.
        r_max : float, default=1.5
            Maximum radius value.
            
        Returns
        -------
        np.ndarray
            Adaptively scaled radius values.
        """
        # Get finite values only
        finite_moduli = moduli[np.isfinite(moduli)]
        if len(finite_moduli) == 0:
            return np.full_like(moduli, r_min)
        
        # Calculate percentiles
        p_low = np.percentile(finite_moduli, low_percentile)
        p_high = np.percentile(finite_moduli, high_percentile)
        
        # Handle edge case where all values are similar
        if p_high <= p_low:
            return np.full_like(moduli, (r_min + r_max) / 2)
        
        # Normalize to [0, 1] based on percentiles
        normalized = np.clip((moduli - p_low) / (p_high - p_low), 0, 1)
        
        # Map to [r_min, r_max]
        return r_min + (r_max - r_min) * normalized
    
    @staticmethod
    def hybrid(moduli: np.ndarray, transition: float = 1.0,
               r_min: float = 0.5, r_max: float = 1.5) -> np.ndarray:
        """Hybrid linear-logarithmic scaling.
        
        Linear for |f| < transition, logarithmic for larger values.
        Ideal for functions with detailed behavior near zero.
        
        Parameters
        ----------
        moduli : np.ndarray
            Modulus values |f(z)|.
        transition : float, default=1.0
            Transition point between linear and logarithmic.
        r_min : float, default=0.5
            Minimum radius value.
        r_max : float, default=1.5
            Maximum radius value.
            
        Returns
        -------
        np.ndarray
            Hybrid-scaled radius values.
        """
        normalized = np.zeros_like(moduli)
        
        # Linear part: [0, transition] -> [0, 0.5]
        small_mask = moduli <= transition
        if np.any(small_mask):
            normalized[small_mask] = 0.5 * moduli[small_mask] / transition
        
        # Logarithmic part: (transition, ∞) -> (0.5, 1]
        large_mask = ~small_mask
        if np.any(large_mask):
            with np.errstate(divide='ignore', invalid='ignore'):
                log_values = np.log(moduli[large_mask] / transition)
                # Use tanh to compress to (0.5, 1]
                normalized[large_mask] = 0.5 + 0.5 * np.tanh(log_values)
        
        # Map to [r_min, r_max]
        return r_min + (r_max - r_min) * normalized


# Scaling presets for common use cases
SCALING_PRESETS = {
    'balanced': {
        'method': 'sigmoid',
        'params': {'steepness': 2.0, 'center': 1.0, 'r_min': 0.2, 'r_max': 1.0},
        'description': 'General purpose sigmoid scaling for balanced visualization'
    },
    'detail_near_zero': {
        'method': 'hybrid',
        'params': {'transition': 0.5, 'r_min': 0.2, 'r_max': 1.0},
        'description': 'Emphasizes small values with hybrid linear-log scaling'
    },
    'auto': {
        'method': 'adaptive',
        'params': {'low_percentile': 5, 'high_percentile': 95, 'r_min': 0.2, 'r_max': 1.0},
        'description': 'Adaptive scaling that automatically adjusts to function range'
    },
    'high_contrast': {
        'method': 'sigmoid',
        'params': {'steepness': 5.0, 'center': 1.0, 'r_min': 0.1, 'r_max': 1.0},
        'description': 'High contrast sigmoid with steep transition'
    },
    'poles_emphasis': {
        'method': 'power',
        'params': {'exponent': 0.3, 'r_min': 0.2, 'r_max': 1.0},
        'description': 'Emphasizes pole behavior with power scaling'
    }
}


def get_scaling_preset(name: str) -> dict:
    """Get a predefined scaling configuration.
    
    Parameters
    ----------
    name : str
        Name of the preset. Available presets:
        - 'balanced': General purpose sigmoid scaling
        - 'detail_near_zero': Emphasizes small values
        - 'auto': Adaptive scaling for unknown functions
        - 'high_contrast': High contrast with steep transition
        - 'poles_emphasis': Emphasizes pole behavior
        
    Returns
    -------
    dict
        Dictionary with 'method' and 'params' keys.
        
    Raises
    ------
    ValueError
        If preset name is not recognized.
    """
    if name not in SCALING_PRESETS:
        available = ', '.join(SCALING_PRESETS.keys())
        raise ValueError(f"Unknown preset: {name}. Available presets: {available}")
    
    preset = SCALING_PRESETS[name]
    return {
        'method': preset['method'],
        'params': preset['params'].copy()
    }