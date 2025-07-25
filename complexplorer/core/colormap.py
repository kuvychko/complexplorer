"""Color mapping functionality for complex visualization.

This module provides various colormaps for domain coloring of complex functions.
Each colormap converts complex values to colors using different techniques.

The module includes:
- Base Colormap class
- Phase colormap (regular and enhanced)
- Chessboard patterns (Cartesian and polar)
- Logarithmic rings
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union
import numpy as np
import matplotlib.colors as mcolors
from ..utils.validation import ValidationError
from .functions import phase as phase_func, sawtooth, sawtooth_log


# Default color for out-of-domain points
OUT_OF_DOMAIN_COLOR_HSV = (0.0, 0.01, 0.9)  # Light gray


class Colormap(ABC):
    """Abstract base class for complex-to-color mappings.
    
    A colormap defines how complex values are mapped to colors.
    Subclasses must implement the hsv_tuple method.
    
    Parameters
    ----------
    out_of_domain_hsv : tuple[float, float, float], optional
        HSV color for points outside the domain.
    """
    
    def __init__(self, 
                 out_of_domain_hsv: Tuple[float, float, float] = OUT_OF_DOMAIN_COLOR_HSV):
        """Initialize colormap with out-of-domain color."""
        self.out_of_domain_hsv = out_of_domain_hsv
    
    @abstractmethod
    def hsv_tuple(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert complex values to HSV components.
        
        Parameters
        ----------
        z : np.ndarray
            Complex values.
            
        Returns
        -------
        H, S, V : tuple of np.ndarray
            Hue, saturation, and value arrays (each in [0, 1]).
        """
        pass
    
    def hsv(self, z: np.ndarray, outmask: Optional[np.ndarray] = None) -> np.ndarray:
        """Convert complex values to HSV array.
        
        Parameters
        ----------
        z : np.ndarray
            Complex values.
        outmask : np.ndarray, optional
            Boolean mask (True for out-of-domain points).
            
        Returns
        -------
        np.ndarray
            HSV values with shape (*z.shape, 3).
        """
        z = np.asarray(z)
        H, S, V = self.hsv_tuple(z)
        
        # Apply out-of-domain coloring
        if outmask is not None:
            H = H.copy()
            S = S.copy()
            V = V.copy()
            H[outmask] = self.out_of_domain_hsv[0]
            S[outmask] = self.out_of_domain_hsv[1]
            V[outmask] = self.out_of_domain_hsv[2]
        
        # Stack along last axis
        if z.ndim == 0:
            # Scalar case
            return np.array([H, S, V])
        else:
            # Use stack instead of dstack to preserve shape
            return np.stack((H, S, V), axis=-1)
    
    def rgb(self, z: np.ndarray, outmask: Optional[np.ndarray] = None) -> np.ndarray:
        """Convert complex values to RGB array.
        
        Parameters
        ----------
        z : np.ndarray
            Complex values.
        outmask : np.ndarray, optional
            Boolean mask (True for out-of-domain points).
            
        Returns
        -------
        np.ndarray
            RGB values with shape (*z.shape, 3).
        """
        hsv = self.hsv(z, outmask)
        return mcolors.hsv_to_rgb(hsv)


class Phase(Colormap):
    """Phase colormap with optional enhancement.
    
    Maps complex phase to hue. Can create enhanced phase portraits
    by modulating saturation/value based on phase sectors and/or
    modulus contours.
    
    Parameters
    ----------
    n_phi : int, optional
        Number of phase sectors for enhancement.
    r_linear_step : float, optional
        Linear modulus step for contours.
    r_log_base : float, optional
        Logarithmic base for modulus contours.
    v_base : float, optional
        Base value (brightness), in [0, 1).
    auto_scale_r : bool, optional
        Auto-calculate r_linear_step for square cells.
    scale_radius : float, optional
        Reference radius for auto-scaling.
    out_of_domain_hsv : tuple, optional
        Color for out-of-domain points.
    """
    
    def __init__(self,
                 n_phi: Optional[int] = None,
                 r_linear_step: Optional[float] = None,
                 r_log_base: Optional[float] = None,
                 v_base: float = 0.5,
                 auto_scale_r: bool = False,
                 scale_radius: float = 1.0,
                 out_of_domain_hsv: Tuple[float, float, float] = OUT_OF_DOMAIN_COLOR_HSV):
        """Initialize phase colormap."""
        super().__init__(out_of_domain_hsv)
        
        # Validate v_base
        if not 0 <= v_base < 1:
            raise ValidationError("v_base must be in [0, 1)")
        
        # Handle auto-scaling
        if auto_scale_r:
            if n_phi is None:
                raise ValidationError("auto_scale_r=True requires n_phi to be specified")
            if r_linear_step is not None:
                raise ValidationError("Cannot specify both auto_scale_r=True and r_linear_step")
            # Calculate r_linear_step for visually square cells
            r_linear_step = 2 * np.pi / n_phi * scale_radius
        
        self.n_phi = n_phi
        self.phi = np.pi / n_phi if n_phi is not None else None
        self.r_linear_step = r_linear_step
        self.r_log_base = r_log_base
        self.v_base = v_base
        self.auto_scale_r = auto_scale_r
        self.scale_radius = scale_radius
    
    def hsv_tuple(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert complex values to HSV components."""
        # Phase determines hue
        phi = phase_func(z)
        H = phi / (2 * np.pi)  # Map [0, 2π] to [0, 1]
        
        # Full saturation by default
        S = np.ones_like(z, dtype=float)
        
        # Value modulation
        if self.phi is not None:
            # Phase-based modulation
            V_phi = sawtooth(phi, self.phi)
        else:
            V_phi = np.ones_like(z, dtype=float)
        
        # Modulus-based modulation
        r = np.abs(z)
        if self.r_linear_step and self.r_log_base is None:
            V_r = sawtooth(r, self.r_linear_step)
        elif self.r_linear_step is None and self.r_log_base:
            V_r = sawtooth_log(r, self.r_log_base)
        elif self.r_linear_step and self.r_log_base:
            V_r = sawtooth_log(r / self.r_linear_step, self.r_log_base)
        else:
            V_r = np.ones_like(z, dtype=float)
        
        # Combine value modulations
        V_scaler = 1 - self.v_base
        V = (V_phi + V_r) * V_scaler / 2 + self.v_base
        
        return H, S, V


class Chessboard(Colormap):
    """Cartesian chessboard pattern.
    
    Creates a black and white chessboard pattern aligned with
    real and imaginary axes.
    
    Parameters
    ----------
    spacing : float, optional
        Size of each square.
    center : complex, optional
        Center of the pattern.
    out_of_domain_hsv : tuple, optional
        Color for out-of-domain points.
    """
    
    def __init__(self,
                 spacing: float = 1.0,
                 center: complex = 0+0j,
                 out_of_domain_hsv: Tuple[float, float, float] = OUT_OF_DOMAIN_COLOR_HSV):
        """Initialize chessboard colormap."""
        super().__init__(out_of_domain_hsv)
        self.spacing = spacing
        self.center = center
    
    def hsv_tuple(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert complex values to HSV components."""
        # No hue or saturation (grayscale)
        H = np.zeros_like(z, dtype=float)
        S = np.zeros_like(z, dtype=float)
        
        # Shift and scale
        z_shifted = (z - self.center) / self.spacing
        
        # Check which square each point is in
        real_idx = np.floor(np.real(z_shifted)).astype(int)
        imag_idx = np.floor(np.imag(z_shifted)).astype(int)
        
        # Chessboard pattern: white if indices have same parity
        V = ((real_idx + imag_idx) % 2 == 0).astype(float)
        
        return H, S, V


class PolarChessboard(Colormap):
    """Polar chessboard pattern.
    
    Creates a black and white pattern in polar coordinates,
    with sectors in phase and rings in modulus.
    
    Parameters
    ----------
    n_phi : int, optional
        Number of phase sectors.
    spacing : float, optional
        Radial spacing between rings.
    r_log : float, optional
        Logarithmic base for radial spacing.
    out_of_domain_hsv : tuple, optional
        Color for out-of-domain points.
    """
    
    def __init__(self,
                 n_phi: int = 6,
                 spacing: float = 1.0,
                 r_log: Optional[float] = None,
                 out_of_domain_hsv: Tuple[float, float, float] = OUT_OF_DOMAIN_COLOR_HSV):
        """Initialize polar chessboard."""
        super().__init__(out_of_domain_hsv)
        self.n_phi = n_phi
        self.phi = np.pi / n_phi
        self.spacing = spacing
        self.r_log = r_log
    
    def hsv_tuple(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert complex values to HSV components."""
        # No hue or saturation (grayscale)
        H = np.zeros_like(z, dtype=float)
        S = np.zeros_like(z, dtype=float)
        
        # Phase sectors
        angle = np.angle(z)
        angle_idx = np.floor((angle + np.pi) / self.phi).astype(int)
        
        # Radial rings
        r = np.abs(z) / self.spacing
        if self.r_log is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                r = np.log(r) / np.log(self.r_log)
        r_idx = np.floor(r).astype(int)
        
        # Chessboard pattern
        V = ((angle_idx + r_idx) % 2 == 0).astype(float)
        
        return H, S, V


class LogRings(Colormap):
    """Logarithmic black and white rings.
    
    Creates concentric rings with logarithmic spacing.
    
    Parameters
    ----------
    log_spacing : float, optional
        Logarithmic spacing parameter.
    out_of_domain_hsv : tuple, optional
        Color for out-of-domain points.
    """
    
    def __init__(self,
                 log_spacing: float = 0.2,
                 out_of_domain_hsv: Tuple[float, float, float] = OUT_OF_DOMAIN_COLOR_HSV):
        """Initialize logarithmic rings."""
        super().__init__(out_of_domain_hsv)
        self.log_spacing = log_spacing
    
    def hsv_tuple(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert complex values to HSV components."""
        # No hue or saturation (grayscale)
        H = np.zeros_like(z, dtype=float)
        S = np.zeros_like(z, dtype=float)
        
        # Logarithmic rings
        with np.errstate(divide='ignore', invalid='ignore'):
            r_log = np.log(np.abs(z)) / self.log_spacing
            # Alternate black and white
            V = (np.floor(r_log) % 2 == 0).astype(float)
        
        # Handle r=0 (log undefined)
        V[np.abs(z) == 0] = 1.0  # White at origin
        
        return H, S, V


# Backward compatibility aliases
Cmap = Colormap  # Keep old name available