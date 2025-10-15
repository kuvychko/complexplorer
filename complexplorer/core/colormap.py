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
from complexplorer.utils.validation import ValidationError
from complexplorer.core.functions import (
    phase as phase_func, sawtooth, sawtooth_log,
    sigmoid, circular_interpolate
)
from complexplorer.core.color_utils import (
    oklch_to_srgb, hsl_to_rgb, cubehelix,
    interpolate_hue, clip_to_gamut
)
from complexplorer.core import constants


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
        if outmask is not None and z.ndim > 0:
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


class BasePhasePortrait(Colormap):
    """Base class for phase portrait colormaps with shared modulation logic.

    This class provides common functionality for colormaps that use:
    - Phase sector enhancement (sawtooth patterns in phase)
    - Modulus-based modulation (linear or logarithmic contours)
    - Auto-scaling for visually square cells

    Subclasses should implement _compute_colors() to define their specific
    color mapping strategy.

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
        """Initialize base phase portrait colormap."""
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

    def _compute_phase_modulation(self, phi: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Compute phase-based value modulation (sawtooth sectors).

        Parameters
        ----------
        phi : np.ndarray
            Phase values in [0, 2π].
        z : np.ndarray
            Complex values (for shape).

        Returns
        -------
        np.ndarray
            Phase modulation in [0, 1].
        """
        if self.phi is not None:
            return sawtooth(phi, self.phi)
        else:
            return np.ones_like(z, dtype=float)

    def _compute_modulus_modulation(self, r: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Compute modulus-based value modulation.

        Parameters
        ----------
        r : np.ndarray
            Modulus values |z|.
        z : np.ndarray
            Complex values (for shape).

        Returns
        -------
        np.ndarray
            Modulus modulation in [0, 1].
        """
        if self.r_linear_step and self.r_log_base is None:
            return sawtooth(r, self.r_linear_step)
        elif self.r_linear_step is None and self.r_log_base:
            return sawtooth_log(r, self.r_log_base)
        elif self.r_linear_step and self.r_log_base:
            return sawtooth_log(r / self.r_linear_step, self.r_log_base)
        else:
            return np.ones_like(z, dtype=float)

    def _combine_modulations(self, V_phi: np.ndarray, V_r: np.ndarray) -> np.ndarray:
        """Combine phase and modulus modulations into final value.

        Parameters
        ----------
        V_phi : np.ndarray
            Phase modulation in [0, 1].
        V_r : np.ndarray
            Modulus modulation in [0, 1].

        Returns
        -------
        np.ndarray
            Combined modulation mapped to [v_base, 1].
        """
        V_scaler = 1 - self.v_base
        return (V_phi + V_r) * V_scaler / 2 + self.v_base

    @abstractmethod
    def _compute_colors(self, z: np.ndarray, phi: np.ndarray, r: np.ndarray,
                       V_phi: np.ndarray, V_r: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute HSV colors from complex values and modulations.

        This method must be implemented by subclasses to define their
        specific color mapping strategy.

        Parameters
        ----------
        z : np.ndarray
            Complex values.
        phi : np.ndarray
            Phase values in [0, 2π].
        r : np.ndarray
            Modulus values |z|.
        V_phi : np.ndarray
            Phase modulation in [0, 1].
        V_r : np.ndarray
            Modulus modulation in [0, 1].

        Returns
        -------
        H, S, V : tuple of np.ndarray
            Hue, saturation, and value arrays.
        """
        pass

    def hsv_tuple(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert complex values to HSV components.

        This method handles the shared modulation logic and delegates
        color computation to _compute_colors().
        """
        # Compute phase and modulus
        phi = phase_func(z)
        r = np.abs(z)

        # Compute modulations
        V_phi = self._compute_phase_modulation(phi, z)
        V_r = self._compute_modulus_modulation(r, z)

        # Delegate to subclass for color computation
        return self._compute_colors(z, phi, r, V_phi, V_r)


class Phase(BasePhasePortrait):
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
    emphasize_unit_circle : bool, optional
        If True, emphasize the unit circle |z|=1.
    unit_circle_strength : float, optional
        Strength of unit circle emphasis (0 to 1).
    unit_circle_color : tuple, optional
        Optional HSV color to blend at unit circle. If None, just brightens.
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
                 emphasize_unit_circle: bool = False,
                 unit_circle_strength: float = 0.3,
                 unit_circle_color: Optional[Tuple[float, float, float]] = None,
                 out_of_domain_hsv: Tuple[float, float, float] = OUT_OF_DOMAIN_COLOR_HSV):
        """Initialize phase colormap."""
        super().__init__(n_phi, r_linear_step, r_log_base, v_base,
                        auto_scale_r, scale_radius, out_of_domain_hsv)

        # Validate unit circle parameters
        if not 0 <= unit_circle_strength <= 1:
            raise ValidationError("unit_circle_strength must be in [0, 1]")

        self.emphasize_unit_circle = emphasize_unit_circle
        self.unit_circle_strength = unit_circle_strength
        self.unit_circle_color = unit_circle_color

    def _compute_colors(self, z: np.ndarray, phi: np.ndarray, r: np.ndarray,
                       V_phi: np.ndarray, V_r: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute HSV colors for Phase colormap."""
        # Phase determines hue
        H = phi / (2 * np.pi)  # Map [0, 2π] to [0, 1]

        # Full saturation by default
        S = np.ones_like(z, dtype=float)

        # Unit circle emphasis
        if self.emphasize_unit_circle:
            # Create emphasis near |z| = 1
            dist_from_unit = np.abs(r - 1.0)
            # Gaussian-like emphasis
            unit_emphasis = np.exp(-10 * dist_from_unit**2)

            if self.unit_circle_color is not None:
                # Blend towards specified color
                H_unit, S_unit, V_unit = self.unit_circle_color
                H = H * (1 - self.unit_circle_strength * unit_emphasis) + H_unit * self.unit_circle_strength * unit_emphasis
                S = S * (1 - self.unit_circle_strength * unit_emphasis) + S_unit * self.unit_circle_strength * unit_emphasis
            else:
                # Just boost brightness near unit circle
                V_r = V_r * (1 + self.unit_circle_strength * unit_emphasis)
                V_r = np.clip(V_r, 0, 1)

        # Combine value modulations
        V = self._combine_modulations(V_phi, V_r)

        return H, S, V


class OklabPhase(BasePhasePortrait):
    """Pure OKLAB phase colormap with optional enhancement.
    
    Implements a perceptually uniform phase portrait using the OKLAB
    color space. Maps complex phase directly to OKLAB hue angle while
    maintaining consistent lightness and chroma. Supports enhanced
    phase/modulus visualization through sawtooth modulation.
    
    Parameters
    ----------
    n_phi : int, optional
        Number of phase sectors for enhancement.
    r_linear_step : float, optional
        Linear modulus step for contours.
    r_log_base : float, optional
        Logarithmic base for modulus contours.
    auto_scale_r : bool, optional
        Auto-calculate r_linear_step for square cells.
    scale_radius : float, optional
        Reference radius for auto-scaling.
    enhanced : bool, optional
        If True, use sawtooth modulation for better structure visibility.
        If False (default), use smooth OKLAB colors similar to cplot.
    L : float, optional
        Base lightness in OKLAB space (0 to 1). Default is 0.7.
    C : float, optional
        Chroma (saturation) in OKLAB space (typically 0 to 0.4). Default is 0.35.
    v_base : float, optional
        Base value for enhanced mode, in [0, 1). Default is 0.5.
    emphasize_unit_circle : bool, optional
        If True, emphasize the unit circle |z|=1.
    unit_circle_strength : float, optional
        Strength of unit circle emphasis (0 to 1).
    phase_offset : float, optional
        Phase rotation offset in radians. Default is 0.8936868*π to match cplot's
        color mapping (green for arg=0, blue for arg=π/2, orange for arg=-π/2, pink for arg=π).
    out_of_domain_hsv : tuple, optional
        Color for out-of-domain points.
        
    Notes
    -----
    The OKLAB color space provides perceptually uniform color gradients,
    meaning equal steps in the color values correspond to equal perceptual
    differences. This is particularly useful for accurate interpretation
    of complex function behavior.
    
    When enhanced=True, the colormap uses sawtooth functions to create
    discontinuous edges at phase and modulus boundaries, dramatically
    improving the visibility of mathematical structures.
    """
    
    def __init__(self,
                 n_phi: Optional[int] = None,
                 r_linear_step: Optional[float] = None,
                 r_log_base: Optional[float] = None,
                 auto_scale_r: bool = False,
                 scale_radius: float = 1.0,
                 enhanced: bool = False,
                 L: float = 0.7,
                 C: float = 0.35,
                 v_base: float = 0.5,
                 emphasize_unit_circle: bool = False,
                 unit_circle_strength: float = 0.3,
                 phase_offset: float = 0.8936868 * np.pi,
                 out_of_domain_hsv: Tuple[float, float, float] = OUT_OF_DOMAIN_COLOR_HSV):
        """Initialize OKLAB phase colormap."""
        super().__init__(n_phi, r_linear_step, r_log_base, v_base,
                        auto_scale_r, scale_radius, out_of_domain_hsv)

        # Validate OklabPhase-specific parameters
        if not 0 <= L <= 1:
            raise ValidationError("L (lightness) must be in [0, 1]")
        if not 0 <= C <= 0.5:
            raise ValidationError("C (chroma) must be in [0, 0.5]")
        if not 0 <= unit_circle_strength <= 1:
            raise ValidationError("unit_circle_strength must be in [0, 1]")

        self.enhanced = enhanced
        self.L = L
        self.C = C
        self.emphasize_unit_circle = emphasize_unit_circle
        self.unit_circle_strength = unit_circle_strength
        self.phase_offset = phase_offset
    
    def _oklab_to_rgb(self, L: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert OKLAB to RGB directly (not via OkLCh).
        
        This is the direct OKLAB to RGB conversion, maintaining
        the cylindrical nature of the color space.
        """
        # OkLab to linear RGB
        l_ = L + 0.3963377774 * a + 0.2158037573 * b
        m_ = L - 0.1055613458 * a - 0.0638541728 * b
        s_ = L - 0.0894841775 * a - 1.2914855480 * b
        
        l = l_ * l_ * l_
        m = m_ * m_ * m_
        s = s_ * s_ * s_
        
        r_linear = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
        g_linear = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
        b_linear = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
        
        # Linear to sRGB (gamma correction)
        def linear_to_srgb(c: np.ndarray) -> np.ndarray:
            c_safe = np.maximum(c, 0)
            return np.where(c_safe <= 0.0031308,
                           12.92 * c_safe,
                           1.055 * np.power(c_safe, 1/2.4) - 0.055)
        
        R = linear_to_srgb(r_linear)
        G = linear_to_srgb(g_linear)
        B = linear_to_srgb(b_linear)
        
        # Clip to valid range
        R = np.clip(R, 0, 1)
        G = np.clip(G, 0, 1)
        B = np.clip(B, 0, 1)
        
        return R, G, B
    
    def _compute_colors(self, z: np.ndarray, phi: np.ndarray, r: np.ndarray,
                       V_phi: np.ndarray, V_r: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute HSV colors for OklabPhase colormap."""
        if self.enhanced:
            # Enhanced mode with sawtooth modulation (complexplorer style)

            # Unit circle emphasis
            if self.emphasize_unit_circle:
                # Create emphasis near |z| = 1
                dist_from_unit = np.abs(r - 1.0)
                # Gaussian-like emphasis
                unit_emphasis = np.exp(-10 * dist_from_unit**2)
                # Boost brightness near unit circle
                V_r = V_r * (1 + self.unit_circle_strength * unit_emphasis)
                V_r = np.clip(V_r, 0, 1)

            # Combine modulations for enhanced visibility
            L_modulated = self._combine_modulations(V_phi, V_r)

            # Apply to OKLAB lightness
            L_final = self.L * L_modulated

        else:
            # Smooth mode (cplot-like)
            L_final = self.L

        # Convert phase to OKLAB a, b components
        # OKLAB uses a cylindrical representation where:
        # a = chroma * cos(hue), b = chroma * sin(hue)
        # Apply phase offset to match cplot's color mapping
        # (green for arg=0, blue for arg=pi/2, etc.)
        phi_shifted = phi + self.phase_offset
        a = self.C * np.cos(phi_shifted)
        b = self.C * np.sin(phi_shifted)

        # Convert OKLAB to RGB
        R, G, B = self._oklab_to_rgb(L_final, a, b)

        # Convert RGB to HSV for compatibility with base class
        rgb_array = np.stack([R, G, B], axis=-1)
        hsv_array = mcolors.rgb_to_hsv(rgb_array)

        return hsv_array[..., 0], hsv_array[..., 1], hsv_array[..., 2]


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

        # Validate parameters
        if spacing <= 0:
            from complexplorer.exceptions import ColormapError
            raise ColormapError("spacing must be positive")

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
        # Suppress warnings for NaN/inf values
        with np.errstate(invalid='ignore'):
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

        # Validate parameters
        if n_phi <= 0:
            from complexplorer.exceptions import ColormapError
            raise ColormapError("n_phi must be positive")
        if spacing <= 0:
            from complexplorer.exceptions import ColormapError
            raise ColormapError("spacing must be positive")
        if r_log is not None and r_log <= 0:
            from complexplorer.exceptions import ColormapError
            raise ColormapError("r_log must be positive")

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

        # Validate parameters
        if log_spacing <= 0:
            from complexplorer.exceptions import ColormapError
            raise ColormapError("log_spacing must be positive")

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


class PerceptualPastel(BasePhasePortrait):
    """Perceptually uniform pastel colormap using OkLCh color space.
    
    Creates elegant, non-fluorescent colors with uniform perceived lightness
    as hue cycles. Phase determines hue, modulus creates lightness bands.
    
    Parameters
    ----------
    n_phi : int, optional
        Number of phase sectors for enhanced phase portrait.
    r_linear_step : float, optional
        Linear modulus step for contours.
    r_log_base : float, optional
        Logarithmic base for modulus contours.
    L_center : float, optional
        Center lightness value in [0, 1]. Default is 0.55.
    L_range : float, optional
        Lightness variation range. Default is 0.3.
    C : float, optional
        Chroma (saturation) in [0, 0.4]. Default is 0.10 for pastels.
    v_base : float, optional
        Base value (brightness) for phase sectors, in [0, 1). Default is 0.5.
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
                 L_center: float = 0.55,
                 L_range: float = 0.3,
                 C: float = 0.1,
                 v_base: float = 0.5,
                 auto_scale_r: bool = False,
                 scale_radius: float = 1.0,
                 out_of_domain_hsv: Tuple[float, float, float] = OUT_OF_DOMAIN_COLOR_HSV):
        """Initialize perceptual pastel colormap."""
        super().__init__(n_phi, r_linear_step, r_log_base, v_base,
                        auto_scale_r, scale_radius, out_of_domain_hsv)

        # Validate parameters
        if not 0 <= L_center <= 1:
            from complexplorer.exceptions import ColormapError
            raise ColormapError("L_center must be in [0, 1]")
        if not 0 <= L_range <= 1:
            from complexplorer.exceptions import ColormapError
            raise ColormapError("L_range must be in [0, 1]")
        if not 0 <= C <= 0.5:
            from complexplorer.exceptions import ColormapError
            raise ColormapError("C (chroma) must be in [0, 0.5]")

        self.L_center = L_center
        self.L_range = L_range
        self.C = C
    
    def _compute_colors(self, z: np.ndarray, phi: np.ndarray, r: np.ndarray,
                       V_phi: np.ndarray, V_r: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute HSV colors for PerceptualPastel colormap."""
        # Phase to hue (in degrees for OkLCh)
        H_deg = phi * 180 / np.pi  # Convert to degrees

        # Combine phase sectors with modulus bands
        L_mod = self._combine_modulations(V_phi, V_r)
        # Map L_mod (which ranges v_base to 1.0) to lightness range
        # Normalize to [0, 1] first: (L_mod - v_base) / (1 - v_base)
        L_normalized = (L_mod - self.v_base) / (1 - self.v_base) if self.v_base < 1 else L_mod
        L = self.L_center + self.L_range * (L_normalized - 0.5)

        # Convert OkLCh to RGB
        R, G, B = oklch_to_srgb(L, self.C, H_deg)

        # Convert RGB to HSV for compatibility with base class
        hsv = mcolors.rgb_to_hsv(np.stack([R, G, B], axis=-1))
        return hsv[..., 0], hsv[..., 1], hsv[..., 2]


class AnalogousWedge(BasePhasePortrait):
    """Analogous color scheme with compressed hue range.
    
    Maps phase to a wedge of the color wheel (20-50% range) for
    harmonious color schemes while preserving phase winding information.
    
    Parameters
    ----------
    n_phi : int, optional
        Number of phase sectors for enhanced phase portrait.
    r_linear_step : float, optional
        Linear modulus step for contours.
    r_log_base : float, optional
        Logarithmic base for modulus contours.
    H_center : float, optional
        Center hue in [0, 1]. Default is 0.55 (cyan-ish).
    H_wedge : float, optional
        Hue range as fraction of color wheel in [0.2, 0.5]. Default is 0.2.
    S : float, optional
        Saturation in [0, 1]. Default is 0.35 (muted).
    V_base : float, optional
        Base value (brightness) in [0, 1]. Default is 0.55.
    V_range : float, optional
        Value modulation range. Default is 0.35.
    auto_scale_r : bool, optional
        Auto-calculate r_linear_step for square cells.
    scale_radius : float, optional
        Reference radius for auto-scaling.
    use_sigmoid : bool, optional
        Use sigmoid for smooth modulus mapping. Default is True.
    out_of_domain_hsv : tuple, optional
        Color for out-of-domain points.
    """
    
    def __init__(self,
                 n_phi: Optional[int] = None,
                 r_linear_step: Optional[float] = None,
                 r_log_base: Optional[float] = None,
                 H_center: float = 0.55,
                 H_wedge: float = 0.2,
                 S: float = 0.35,
                 V_base: float = 0.55,
                 V_range: float = 0.35,
                 auto_scale_r: bool = False,
                 scale_radius: float = 1.0,
                 use_sigmoid: bool = True,
                 out_of_domain_hsv: Tuple[float, float, float] = OUT_OF_DOMAIN_COLOR_HSV):
        """Initialize analogous wedge colormap."""
        # Note: V_base parameter maps to v_base in BasePhasePortrait
        super().__init__(n_phi, r_linear_step, r_log_base, V_base,
                        auto_scale_r, scale_radius, out_of_domain_hsv)

        # Validate parameters
        if not 0 <= H_center <= 1:
            from complexplorer.exceptions import ColormapError
            raise ColormapError("H_center must be in [0, 1]")
        if not 0.2 <= H_wedge <= 0.5:
            from complexplorer.exceptions import ColormapError
            raise ColormapError("H_wedge must be in [0.2, 0.5]")
        if not 0 <= S <= 1:
            from complexplorer.exceptions import ColormapError
            raise ColormapError("S (saturation) must be in [0, 1]")
        if not 0 <= V_range <= 1:
            from complexplorer.exceptions import ColormapError
            raise ColormapError("V_range must be in [0, 1]")

        self.H_center = H_center
        self.H_wedge = H_wedge  # No longer need clamping since we validate
        self.S = S
        self.V_base = V_base  # Store original parameter for backwards compatibility
        self.V_range = V_range
        self.use_sigmoid = use_sigmoid

    def _compute_modulus_modulation(self, r: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Compute modulus-based value modulation with sigmoid/tanh option."""
        if self.r_linear_step and self.r_log_base is None:
            return sawtooth(r, self.r_linear_step)
        elif self.r_linear_step is None and self.r_log_base:
            return sawtooth_log(r, self.r_log_base)
        elif self.r_linear_step and self.r_log_base:
            return sawtooth_log(r / self.r_linear_step, self.r_log_base)
        else:
            # Use sigmoid or tanh for smooth modulus mapping
            if self.use_sigmoid:
                with np.errstate(divide='ignore', invalid='ignore'):
                    return sigmoid(np.log(r), center=0, scale=2)
            else:
                return np.tanh(r / 2)  # Maps [0, ∞) to [0, 1)

    def _compute_colors(self, z: np.ndarray, phi: np.ndarray, r: np.ndarray,
                       V_phi: np.ndarray, V_r: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute HSV colors for AnalogousWedge colormap."""
        # Phase to compressed hue range
        H = self.H_center + self.H_wedge * (phi / (2 * np.pi) - 0.5)
        H = np.mod(H, 1.0)  # Wrap to [0, 1]

        # Fixed saturation
        S = np.full_like(z, self.S, dtype=float)

        # Combine phase sectors with modulus
        V = self._combine_modulations(V_phi, V_r)
        V = np.clip(V, 0, 1)

        return H, S, V


class DivergingWarmCool(BasePhasePortrait):
    """Diverging warm-cool colormap based on phase sign.
    
    Positive phases lean toward warm colors, negative toward cool.
    Creates refined, cartographic appearance with natural emphasis
    on real/imaginary axes.
    
    Parameters
    ----------
    n_phi : int, optional
        Number of phase sectors for enhanced phase portrait.
    r_linear_step : float, optional
        Period for linear modulus rings. Default is None (no rings).
    r_log_base : float, optional
        Base for logarithmic modulus rings. Default is None (no rings).
    auto_scale_r : bool, optional
        Auto-calculate r_linear_step from n_phi. Default is False.
    scale_radius : float, optional
        Scale factor for auto-calculated r_linear_step. Default is 1.0.
    H_warm : float, optional
        Warm anchor hue in degrees. Default is 30 (amber).
    H_cool : float, optional
        Cool anchor hue in degrees. Default is 220 (indigo).
    L_center : float, optional
        Center lightness in [0, 1]. Default is 0.5.
    L_range : float, optional
        Lightness modulation range. Default is 0.3.
    C_min : float, optional
        Minimum chroma. Default is 0.04.
    C_max : float, optional
        Maximum chroma. Default is 0.14.
    v_base : float, optional
        Base value for phase sectors, in [0, 1). Default is 0.5.
    use_oklch : bool, optional
        Use OkLCh color space for perceptual uniformity. Default is True.
    out_of_domain_hsv : tuple, optional
        Color for out-of-domain points.
    """
    
    def __init__(self,
                 n_phi: Optional[int] = None,
                 r_linear_step: Optional[float] = None,
                 r_log_base: Optional[float] = None,
                 auto_scale_r: bool = False,
                 scale_radius: float = 1.0,
                 H_warm: float = 30,
                 H_cool: float = 220,
                 L_center: float = 0.5,
                 L_range: float = 0.3,
                 C_min: float = 0.04,
                 C_max: float = 0.14,
                 v_base: float = 0.5,
                 use_oklch: bool = True,
                 out_of_domain_hsv: Tuple[float, float, float] = OUT_OF_DOMAIN_COLOR_HSV):
        """Initialize diverging warm-cool colormap."""
        super().__init__(n_phi, r_linear_step, r_log_base, v_base,
                        auto_scale_r, scale_radius, out_of_domain_hsv)

        # Validate parameters
        if not 0 <= L_center <= 1:
            from complexplorer.exceptions import ColormapError
            raise ColormapError("L_center must be in [0, 1]")
        if not 0 <= L_range <= 1:
            from complexplorer.exceptions import ColormapError
            raise ColormapError("L_range must be in [0, 1]")
        if not 0 <= C_min <= 0.5:
            from complexplorer.exceptions import ColormapError
            raise ColormapError("C_min must be in [0, 0.5]")
        if not 0 <= C_max <= 0.5:
            from complexplorer.exceptions import ColormapError
            raise ColormapError("C_max must be in [0, 0.5]")
        if C_min > C_max:
            from complexplorer.exceptions import ColormapError
            raise ColormapError("C_min must be less than or equal to C_max")

        self.H_warm = H_warm
        self.H_cool = H_cool
        self.L_center = L_center
        self.L_range = L_range
        self.C_min = C_min
        self.C_max = C_max
        self.use_oklch = use_oklch
    
    def _compute_modulus_modulation(self, r: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Compute modulus-based value modulation with sigmoid default."""
        if self.r_linear_step is not None:
            return sawtooth(r, self.r_linear_step)
        elif self.r_log_base is not None:
            return sawtooth_log(r, self.r_log_base)
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                return sigmoid(np.log(r), center=0, scale=2)

    def _compute_colors(self, z: np.ndarray, phi: np.ndarray, r: np.ndarray,
                       V_phi: np.ndarray, V_r: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute HSV colors for DivergingWarmCool colormap."""
        # Map phase to warm-cool interpolation parameter
        a = np.sin(phi)  # in [-1, 1]
        t = (a + 1) / 2  # in [0, 1]

        if self.use_oklch:
            # Interpolate hue in OkLCh space
            H_deg = (1 - t) * self.H_cool + t * self.H_warm

            # Combine phase sectors with modulus
            L_mod = self._combine_modulations(V_phi, V_r)
            # Map L_mod (which ranges v_base to 1.0) to lightness range
            L_normalized = (L_mod - self.v_base) / (1 - self.v_base) if self.v_base < 1 else L_mod
            L = self.L_center + self.L_range * (L_normalized - 0.5)

            # Vary chroma based on phase extremity
            C = self.C_min + (self.C_max - self.C_min) * np.abs(a)

            # Convert to RGB
            R, G, B = oklch_to_srgb(L, C, H_deg)

            # Convert RGB to HSV
            hsv = mcolors.rgb_to_hsv(np.stack([R, G, B], axis=-1))
            return hsv[..., 0], hsv[..., 1], hsv[..., 2]
        else:
            # Simple HSV interpolation
            H = interpolate_hue(self.H_cool / 360, self.H_warm / 360, t)

            # Combine phase sectors with modulus
            V = self._combine_modulations(V_phi, V_r)

            # Vary saturation based on phase extremity
            S = self.C_min + (self.C_max - self.C_min) * np.abs(a)

            return H, S, V


class Isoluminant(BasePhasePortrait):
    """Isoluminant colormap with optional contour lines.
    
    Maintains constant lightness with hue encoding phase only.
    Optionally overlays thin contour lines to show modulus information.
    
    Parameters
    ----------
    n_phi : int, optional
        Number of phase sectors for enhanced phase portrait.
    r_linear_step : float, optional
        Period for linear modulus rings. Default is None (no rings).
    r_log_base : float, optional
        Base for logarithmic modulus rings. Default is None (no rings).
    auto_scale_r : bool, optional
        Auto-calculate r_linear_step from n_phi. Default is False.
    scale_radius : float, optional
        Scale factor for auto-calculated r_linear_step. Default is 1.0.
    L : float, optional
        Constant lightness in [0, 1]. Default is 0.6.
    C_min : float, optional
        Minimum chroma. Default is 0.12.
    C_max : float, optional
        Maximum chroma. Default is 0.18.
    v_base : float, optional
        Base value for phase sectors, in [0, 1). Default is 0.5.
    show_contours : bool, optional
        Show modulus contour lines. Default is True.
    contour_period : float, optional
        Period for contour lines. Default is 1.0.
    contour_width : float, optional
        Width parameter for contours. Default is 0.05.
    use_oklch : bool, optional
        Use OkLCh for perceptual uniformity. Default is True.
    out_of_domain_hsv : tuple, optional
        Color for out-of-domain points.
    """
    
    def __init__(self,
                 n_phi: Optional[int] = None,
                 r_linear_step: Optional[float] = None,
                 r_log_base: Optional[float] = None,
                 auto_scale_r: bool = False,
                 scale_radius: float = 1.0,
                 L: float = 0.6,
                 C_min: float = 0.12,
                 C_max: float = 0.18,
                 v_base: float = 0.5,
                 show_contours: bool = True,
                 contour_period: float = 1.0,
                 contour_width: float = 0.05,
                 use_oklch: bool = True,
                 out_of_domain_hsv: Tuple[float, float, float] = OUT_OF_DOMAIN_COLOR_HSV):
        """Initialize isoluminant colormap."""
        super().__init__(n_phi, r_linear_step, r_log_base, v_base,
                        auto_scale_r, scale_radius, out_of_domain_hsv)

        # Validate parameters
        if not 0 <= L <= 1:
            from complexplorer.exceptions import ColormapError
            raise ColormapError("L (lightness) must be in [0, 1]")
        if not 0 <= C_min <= 0.5:
            from complexplorer.exceptions import ColormapError
            raise ColormapError("C_min must be in [0, 0.5]")
        if not 0 <= C_max <= 0.5:
            from complexplorer.exceptions import ColormapError
            raise ColormapError("C_max must be in [0, 0.5]")
        if C_min > C_max:
            from complexplorer.exceptions import ColormapError
            raise ColormapError("C_min must be less than or equal to C_max")
        if contour_period <= 0:
            from complexplorer.exceptions import ColormapError
            raise ColormapError("contour_period must be positive")
        if contour_width <= 0:
            from complexplorer.exceptions import ColormapError
            raise ColormapError("contour_width must be positive")

        self.L = L
        self.C_min = C_min
        self.C_max = C_max
        self.show_contours = show_contours
        self.contour_period = contour_period
        self.contour_width = contour_width
        self.use_oklch = use_oklch

    def _compute_modulus_modulation(self, r: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Compute modulus-based value modulation (special handling for show_contours)."""
        # When show_contours is True, don't use modulus modulation
        # (contours are applied separately in _compute_colors)
        if not self.show_contours:
            if self.r_linear_step is not None:
                return sawtooth(r, self.r_linear_step)
            elif self.r_log_base is not None:
                return sawtooth_log(r, self.r_log_base)
            else:
                return np.ones_like(z, dtype=float)
        else:
            return np.ones_like(z, dtype=float)

    def _compute_colors(self, z: np.ndarray, phi: np.ndarray, r: np.ndarray,
                       V_phi: np.ndarray, V_r: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute HSV colors for Isoluminant colormap."""
        if self.use_oklch:
            # Phase to hue in degrees
            H_deg = phi * 180 / np.pi

            # Base lightness modulated by phase sectors and modulus
            V_scaler = 1 - self.v_base
            if not self.show_contours:
                # Include modulus modulation
                L_mod = (V_phi + V_r) * V_scaler / 2 + self.v_base
                # Map L_mod (which ranges v_base to 1.0) to subtle lightness variation
                L_normalized = (L_mod - self.v_base) / (1 - self.v_base) if self.v_base < 1 else L_mod
                L = self.L + (L_normalized - 0.5) * 0.2  # Subtle modulation
            else:
                L_base = self.L + (V_phi - 0.5) * V_scaler * 0.2  # Subtle phase modulation
                L = np.full_like(z, L_base, dtype=float)

            if self.show_contours:
                with np.errstate(divide='ignore', invalid='ignore'):
                    rho = np.log(r) / np.log(np.e)
                T_rho = np.mod(rho / self.contour_period, 1.0)
                # Gaussian-like contour lines
                contour = 1 - np.exp(-np.power(T_rho / self.contour_width, 2))
                L = L * (0.8 + 0.2 * contour)  # Darken at contours

            # Vary chroma slightly with modulus
            C = self.C_min + (self.C_max - self.C_min) * np.tanh(r / 2)

            # Convert to RGB
            R, G, B = oklch_to_srgb(L, C, H_deg)

            # Convert RGB to HSV
            hsv = mcolors.rgb_to_hsv(np.stack([R, G, B], axis=-1))
            return hsv[..., 0], hsv[..., 1], hsv[..., 2]
        else:
            # Simple HSV version
            H = phi / (2 * np.pi)
            S = np.full_like(z, 0.5, dtype=float)

            # Base value modulated by phase sectors and modulus
            V_scaler = 1 - self.v_base
            if not self.show_contours:
                # Include modulus modulation
                V_mod = (V_phi + V_r) * V_scaler / 2 + self.v_base
                V = self.L + (V_mod - 0.75) * 0.2  # Subtle modulation
            else:
                V = np.full_like(z, self.L, dtype=float)

            if self.show_contours:
                with np.errstate(divide='ignore', invalid='ignore'):
                    rho = np.log(r) / np.log(np.e)
                T_rho = np.mod(rho / self.contour_period, 1.0)
                contour = 1 - np.exp(-np.power(T_rho / self.contour_width, 2))
                V = V * (0.8 + 0.2 * contour)

            return H, S, V


class CubehelixPhase(BasePhasePortrait):
    """Cubehelix colormap driven by complex phase.
    
    Uses Dave Green's cubehelix color scheme which maintains
    monotonic perceived brightness and prints well in grayscale.
    
    Parameters
    ----------
    n_phi : int, optional
        Number of phase sectors for enhanced phase portrait.
    r_linear_step : float, optional
        Period for linear modulus rings. Default is None (no rings).
    r_log_base : float, optional
        Base for logarithmic modulus rings. Default is None (no rings).
    auto_scale_r : bool, optional
        Auto-calculate r_linear_step from n_phi. Default is False.
    scale_radius : float, optional
        Scale factor for auto-calculated r_linear_step. Default is 1.0.
    start : float, optional
        Starting color (1=red, 2=green, 3=blue). Default is 0.5.
    rotations : float, optional
        Number of rotations through color space. Default is -1.5.
    saturation : float, optional
        Color saturation in [0, 1]. Default is 0.8.
    L_min : float, optional
        Minimum lightness. Default is 0.15.
    L_max : float, optional
        Maximum lightness. Default is 0.85.
    v_base : float, optional
        Base value for phase sectors, in [0, 1). Default is 0.5.
    gamma : float, optional
        Gamma correction factor. Default is 1.0.
    modulate_with_r : bool, optional
        Modulate lightness with modulus. Default is True.
    out_of_domain_hsv : tuple, optional
        Color for out-of-domain points.
    """
    
    def __init__(self,
                 n_phi: Optional[int] = None,
                 r_linear_step: Optional[float] = None,
                 r_log_base: Optional[float] = None,
                 auto_scale_r: bool = False,
                 scale_radius: float = 1.0,
                 start: float = 0.5,
                 rotations: float = -1.5,
                 saturation: float = 0.8,
                 L_min: float = 0.15,
                 L_max: float = 0.85,
                 v_base: float = 0.5,
                 gamma: float = 1.0,
                 modulate_with_r: bool = True,
                 out_of_domain_hsv: Tuple[float, float, float] = OUT_OF_DOMAIN_COLOR_HSV):
        """Initialize cubehelix phase colormap."""
        super().__init__(n_phi, r_linear_step, r_log_base, v_base,
                        auto_scale_r, scale_radius, out_of_domain_hsv)

        # Validate parameters
        if not 0 <= saturation <= 1:
            from complexplorer.exceptions import ColormapError
            raise ColormapError("saturation must be in [0, 1]")
        if not 0 <= L_min <= 1:
            from complexplorer.exceptions import ColormapError
            raise ColormapError("L_min must be in [0, 1]")
        if not 0 <= L_max <= 1:
            from complexplorer.exceptions import ColormapError
            raise ColormapError("L_max must be in [0, 1]")
        if L_min > L_max:
            from complexplorer.exceptions import ColormapError
            raise ColormapError("L_min must be less than or equal to L_max")
        if gamma <= 0:
            from complexplorer.exceptions import ColormapError
            raise ColormapError("gamma must be positive")

        self.start = start
        self.rotations = rotations
        self.saturation = saturation
        self.L_min = L_min
        self.L_max = L_max
        self.gamma = gamma
        self.modulate_with_r = modulate_with_r

    def _compute_modulus_modulation(self, r: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Compute modulus-based value modulation with sigmoid default."""
        if self.r_linear_step is not None:
            return sawtooth(r, self.r_linear_step)
        elif self.r_log_base is not None:
            return sawtooth_log(r, self.r_log_base)
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                return sigmoid(np.log(r), center=0, scale=2)

    def _compute_colors(self, z: np.ndarray, phi: np.ndarray, r: np.ndarray,
                       V_phi: np.ndarray, V_r: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute HSV colors for CubehelixPhase colormap."""
        # Phase determines position along helix
        h = phi / (2 * np.pi)  # Normalize to [0, 1]

        if self.modulate_with_r:
            # Combine phase sectors and modulus
            combined_mod = self._combine_modulations(V_phi, V_r)
            h_effective = h * (self.L_min + (self.L_max - self.L_min) * combined_mod)
        else:
            # Apply phase sectors to base scaling
            V_scaler = 1 - self.v_base
            h_mod = h * ((V_phi * V_scaler) + self.v_base)
            h_effective = self.L_min + h_mod * (self.L_max - self.L_min)

        # Generate cubehelix colors
        R, G, B = cubehelix(h_effective, s=self.saturation, r=self.rotations, gamma=self.gamma)

        # Convert RGB to HSV
        hsv = mcolors.rgb_to_hsv(np.stack([R, G, B], axis=-1))
        return hsv[..., 0], hsv[..., 1], hsv[..., 2]


class InkPaper(BasePhasePortrait):
    """Nearly monochrome colormap with subtle phase tints.
    
    Creates a classy, etching-like appearance that's almost grayscale
    with just enough color to read phase information.
    
    Parameters
    ----------
    n_phi : int, optional
        Number of phase sectors for enhanced phase portrait.
    r_linear_step : float, optional
        Period for linear modulus rings. Default is None (no rings).
    r_log_base : float, optional
        Base for logarithmic modulus rings. Default is None (no rings).
    auto_scale_r : bool, optional
        Auto-calculate r_linear_step from n_phi. Default is False.
    scale_radius : float, optional
        Scale factor for auto-calculated r_linear_step. Default is 1.0.
    L_min : float, optional
        Minimum lightness. Default is 0.35.
    L_max : float, optional
        Maximum lightness. Default is 0.85.
    C_min : float, optional
        Minimum chroma (very low for near-monochrome). Default is 0.02.
    C_max : float, optional
        Maximum chroma. Default is 0.06.
    v_base : float, optional
        Base value for phase sectors, in [0, 1). Default is 0.5.
    add_phase_stripes : bool, optional
        Add subtle phase stripes. Default is False.
    stripe_count : int, optional
        Number of phase stripes if enabled. Default is 8.
    stripe_amplitude : float, optional
        Amplitude of phase stripes. Default is 0.03.
    use_oklch : bool, optional
        Use OkLCh color space. Default is True.
    out_of_domain_hsv : tuple, optional
        Color for out-of-domain points.
    """
    
    def __init__(self,
                 n_phi: Optional[int] = None,
                 r_linear_step: Optional[float] = None,
                 r_log_base: Optional[float] = None,
                 auto_scale_r: bool = False,
                 scale_radius: float = 1.0,
                 L_min: float = 0.35,
                 L_max: float = 0.85,
                 C_min: float = 0.02,
                 C_max: float = 0.06,
                 v_base: float = 0.5,
                 add_phase_stripes: bool = False,
                 stripe_count: int = 8,
                 stripe_amplitude: float = 0.03,
                 use_oklch: bool = True,
                 out_of_domain_hsv: Tuple[float, float, float] = OUT_OF_DOMAIN_COLOR_HSV):
        """Initialize ink & paper colormap."""
        super().__init__(n_phi, r_linear_step, r_log_base, v_base,
                        auto_scale_r, scale_radius, out_of_domain_hsv)

        # Validate parameters
        if not 0 <= L_min <= 1:
            from complexplorer.exceptions import ColormapError
            raise ColormapError("L_min must be in [0, 1]")
        if not 0 <= L_max <= 1:
            from complexplorer.exceptions import ColormapError
            raise ColormapError("L_max must be in [0, 1]")
        if L_min > L_max:
            from complexplorer.exceptions import ColormapError
            raise ColormapError("L_min must be less than or equal to L_max")
        if not 0 <= C_min <= 0.5:
            from complexplorer.exceptions import ColormapError
            raise ColormapError("C_min must be in [0, 0.5]")
        if not 0 <= C_max <= 0.5:
            from complexplorer.exceptions import ColormapError
            raise ColormapError("C_max must be in [0, 0.5]")
        if C_min > C_max:
            from complexplorer.exceptions import ColormapError
            raise ColormapError("C_min must be less than or equal to C_max")
        if stripe_count <= 0:
            from complexplorer.exceptions import ColormapError
            raise ColormapError("stripe_count must be positive")
        if not 0 <= stripe_amplitude <= 1:
            from complexplorer.exceptions import ColormapError
            raise ColormapError("stripe_amplitude must be in [0, 1]")

        self.L_min = L_min
        self.L_max = L_max
        self.C_min = C_min
        self.C_max = C_max
        self.add_phase_stripes = add_phase_stripes
        self.stripe_count = stripe_count
        self.stripe_amplitude = stripe_amplitude
        self.use_oklch = use_oklch

    def _compute_modulus_modulation(self, r: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Compute modulus-based value modulation with sigmoid default."""
        if self.r_linear_step is not None:
            return sawtooth(r, self.r_linear_step)
        elif self.r_log_base is not None:
            return sawtooth_log(r, self.r_log_base)
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                return sigmoid(np.log(r), center=0, scale=2)

    def _compute_colors(self, z: np.ndarray, phi: np.ndarray, r: np.ndarray,
                       V_phi: np.ndarray, V_r: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute HSV colors for InkPaper colormap."""
        if self.use_oklch:
            # Phase to hue (degrees)
            H_deg = phi * 180 / np.pi

            # Combine phase sectors and modulus for lightness modulation
            if self.phi is not None:
                # Combine with phase sectors
                combined_mod = self._combine_modulations(V_phi, V_r)
            else:
                combined_mod = V_r
            L = self.L_min + (self.L_max - self.L_min) * combined_mod

            # Add phase stripes if requested
            if self.add_phase_stripes:
                L = L + self.stripe_amplitude * np.cos(self.stripe_count * phi)
                L = np.clip(L, 0, 1)

            # Very low chroma for near-monochrome look
            C = self.C_min + (self.C_max - self.C_min) * np.tanh(r)

            # Convert to RGB
            R, G, B = oklch_to_srgb(L, C, H_deg)

            # Convert RGB to HSV
            hsv = mcolors.rgb_to_hsv(np.stack([R, G, B], axis=-1))
            return hsv[..., 0], hsv[..., 1], hsv[..., 2]
        else:
            # Simple HSV version
            H = phi / (2 * np.pi)

            # Very low saturation
            S = np.full_like(z, 0.1, dtype=float)

            # Combine phase sectors and modulus for value modulation
            if self.phi is not None:
                # Combine with phase sectors
                combined_mod = self._combine_modulations(V_phi, V_r)
            else:
                combined_mod = V_r
            V = self.L_min + (self.L_max - self.L_min) * combined_mod

            if self.add_phase_stripes:
                V = V + self.stripe_amplitude * np.cos(self.stripe_count * phi)
                V = np.clip(V, 0, 1)

            return H, S, V


class EarthTopographic(BasePhasePortrait):
    """Earth-tone topographic colormap.
    
    Creates a terrain-inspired aesthetic where modulus appears as
    elevation with phase providing subtle earth-tone tints.
    
    Parameters
    ----------
    n_phi : int, optional
        Number of phase sectors for enhanced phase portrait.
    r_linear_step : float, optional
        Period for linear modulus rings. Default is None (no rings).
    r_log_base : float, optional
        Base for logarithmic modulus rings. Default is None (no rings).
    auto_scale_r : bool, optional
        Auto-calculate r_linear_step from n_phi. Default is False.
    scale_radius : float, optional
        Scale factor for auto-calculated r_linear_step. Default is 1.0.
    L_min : float, optional
        Minimum lightness. Default is 0.4.
    L_max : float, optional
        Maximum lightness. Default is 0.8.
    H_water : float, optional
        Water hue in degrees (bluish). Default is 200.
    H_land : float, optional
        Land hue in degrees (brownish). Default is 30.
    C_min : float, optional
        Minimum chroma. Default is 0.05.
    C_max : float, optional
        Maximum chroma. Default is 0.12.
    add_hillshade : bool, optional
        Add hillshade effect to modulus. Default is True.
    hillshade_amplitude : float, optional
        Amplitude of hillshade effect. Default is 0.07.
    v_base : float, optional
        Base value for phase sectors, in [0, 1). Default is 0.5.
    use_oklch : bool, optional
        Use OkLCh color space. Default is True.
    out_of_domain_hsv : tuple, optional
        Color for out-of-domain points.
    """
    
    def __init__(self,
                 n_phi: Optional[int] = None,
                 r_linear_step: Optional[float] = None,
                 r_log_base: Optional[float] = None,
                 auto_scale_r: bool = False,
                 scale_radius: float = 1.0,
                 L_min: float = 0.4,
                 L_max: float = 0.8,
                 H_water: float = 200,
                 H_land: float = 30,
                 C_min: float = 0.05,
                 C_max: float = 0.12,
                 add_hillshade: bool = True,
                 hillshade_amplitude: float = 0.07,
                 v_base: float = 0.5,
                 use_oklch: bool = True,
                 out_of_domain_hsv: Tuple[float, float, float] = OUT_OF_DOMAIN_COLOR_HSV):
        """Initialize earth topographic colormap."""
        super().__init__(n_phi, r_linear_step, r_log_base, v_base,
                        auto_scale_r, scale_radius, out_of_domain_hsv)

        # Validate parameters
        if not 0 <= L_min <= 1:
            from complexplorer.exceptions import ColormapError
            raise ColormapError("L_min must be in [0, 1]")
        if not 0 <= L_max <= 1:
            from complexplorer.exceptions import ColormapError
            raise ColormapError("L_max must be in [0, 1]")
        if L_min > L_max:
            from complexplorer.exceptions import ColormapError
            raise ColormapError("L_min must be less than or equal to L_max")
        if not 0 <= C_min <= 0.5:
            from complexplorer.exceptions import ColormapError
            raise ColormapError("C_min must be in [0, 0.5]")
        if not 0 <= C_max <= 0.5:
            from complexplorer.exceptions import ColormapError
            raise ColormapError("C_max must be in [0, 0.5]")
        if C_min > C_max:
            from complexplorer.exceptions import ColormapError
            raise ColormapError("C_min must be less than or equal to C_max")
        if not 0 <= hillshade_amplitude <= 1:
            from complexplorer.exceptions import ColormapError
            raise ColormapError("hillshade_amplitude must be in [0, 1]")

        self.L_min = L_min
        self.L_max = L_max
        self.H_water = H_water
        self.H_land = H_land
        self.C_min = C_min
        self.C_max = C_max
        self.add_hillshade = add_hillshade
        self.hillshade_amplitude = hillshade_amplitude
        self.use_oklch = use_oklch

    def _compute_modulus_modulation(self, r: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Compute modulus-based value modulation with sigmoid default."""
        if self.r_linear_step is not None:
            return sawtooth(r, self.r_linear_step)
        elif self.r_log_base is not None:
            return sawtooth_log(r, self.r_log_base)
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                return sigmoid(np.log(r), center=0, scale=2)

    def _compute_colors(self, z: np.ndarray, phi: np.ndarray, r: np.ndarray,
                       V_phi: np.ndarray, V_r: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute HSV colors for EarthTopographic colormap."""
        if self.use_oklch:
            # Map phase to water/land hues
            t = (np.sin(phi) + 1) / 2  # Map to [0, 1]
            H_deg = (1 - t) * self.H_water + t * self.H_land

            # Base lightness from modulus
            L_base = self.L_min + (self.L_max - self.L_min) * V_r

            # Combine phase sectors with modulus
            if self.phi is not None:
                # Mix phase and modulus modulation for clear phase sectors
                combined_mod = self._combine_modulations(V_phi, V_r)
                L = self.L_min + (self.L_max - self.L_min) * combined_mod
            else:
                L = L_base

            # Add hillshade effect
            if self.add_hillshade:
                with np.errstate(divide='ignore', invalid='ignore'):
                    rho = np.log(r) / np.log(np.e)
                T_rho = np.mod(rho, 1.0)
                hillshade = np.cos(2 * np.pi * T_rho)
                L = L + self.hillshade_amplitude * hillshade
                L = np.clip(L, 0, 1)

            # Subtle earth-tone chroma
            C = self.C_min + (self.C_max - self.C_min) * np.tanh(r / 2)

            # Convert to RGB
            R, G, B = oklch_to_srgb(L, C, H_deg)

            # Convert RGB to HSV
            hsv = mcolors.rgb_to_hsv(np.stack([R, G, B], axis=-1))
            return hsv[..., 0], hsv[..., 1], hsv[..., 2]
        else:
            # Simple HSV version
            t = (np.sin(phi) + 1) / 2
            H = interpolate_hue(self.H_water / 360, self.H_land / 360, t)

            S = np.full_like(z, 0.3, dtype=float)

            # Base value from modulus
            V_base = self.L_min + (self.L_max - self.L_min) * V_r

            # Combine phase sectors with modulus
            if self.phi is not None:
                # Mix phase and modulus modulation for clear phase sectors
                combined_mod = self._combine_modulations(V_phi, V_r)
                V = self.L_min + (self.L_max - self.L_min) * combined_mod
            else:
                V = V_base

            if self.add_hillshade:
                with np.errstate(divide='ignore', invalid='ignore'):
                    rho = np.log(r) / np.log(np.e)
                T_rho = np.mod(rho, 1.0)
                hillshade = np.cos(2 * np.pi * T_rho)
                V = V + self.hillshade_amplitude * hillshade
                V = np.clip(V, 0, 1)

            return H, S, V


class FourQuadrant(BasePhasePortrait):
    """Four-quadrant colormap with smooth circular interpolation.
    
    Maps the four principal phase angles to four color anchors
    and smoothly interpolates between them on the circle.
    
    Parameters
    ----------
    n_phi : int, optional
        Number of phase sectors for enhanced phase portrait.
    r_linear_step : float, optional
        Period for linear modulus rings. Default is None (no rings).
    r_log_base : float, optional
        Base for logarithmic modulus rings. Default is None (no rings).
    auto_scale_r : bool, optional
        Auto-calculate r_linear_step from n_phi. Default is False.
    scale_radius : float, optional
        Scale factor for auto-calculated r_linear_step. Default is 1.0.
    H_anchors : tuple, optional
        Four hue anchors in degrees for 0°, 90°, 180°, 270°.
        Default is (10, 120, 210, 300) for red, green, cyan, magenta.
    C : float, optional
        Chroma (saturation) in [0, 0.4]. Default is 0.10.
    L_min : float, optional
        Minimum lightness. Default is 0.4.
    L_max : float, optional
        Maximum lightness. Default is 0.8.
    use_oklch : bool, optional
        Use OkLCh for perceptual uniformity. Default is True.
    smooth_interpolation : bool, optional
        Use smooth spline interpolation. Default is True.
    v_base : float, optional
        Base value for phase sectors, in [0, 1). Default is 0.5.
    out_of_domain_hsv : tuple, optional
        Color for out-of-domain points.
    """
    
    def __init__(self,
                 n_phi: Optional[int] = None,
                 r_linear_step: Optional[float] = None,
                 r_log_base: Optional[float] = None,
                 auto_scale_r: bool = False,
                 scale_radius: float = 1.0,
                 H_anchors: Tuple[float, float, float, float] = (10, 120, 210, 300),
                 C: float = 0.10,
                 L_min: float = 0.4,
                 L_max: float = 0.8,
                 use_oklch: bool = True,
                 smooth_interpolation: bool = True,
                 v_base: float = 0.5,
                 out_of_domain_hsv: Tuple[float, float, float] = OUT_OF_DOMAIN_COLOR_HSV):
        """Initialize four-quadrant colormap."""
        super().__init__(n_phi, r_linear_step, r_log_base, v_base,
                        auto_scale_r, scale_radius, out_of_domain_hsv)

        # Validate parameters
        if len(H_anchors) != 4:
            from complexplorer.exceptions import ColormapError
            raise ColormapError("H_anchors must have exactly 4 elements")
        if not 0 <= C <= 0.5:
            from complexplorer.exceptions import ColormapError
            raise ColormapError("C (chroma) must be in [0, 0.5]")
        if not 0 <= L_min <= 1:
            from complexplorer.exceptions import ColormapError
            raise ColormapError("L_min must be in [0, 1]")
        if not 0 <= L_max <= 1:
            from complexplorer.exceptions import ColormapError
            raise ColormapError("L_max must be in [0, 1]")
        if L_min > L_max:
            from complexplorer.exceptions import ColormapError
            raise ColormapError("L_min must be less than or equal to L_max")

        self.H_anchors = H_anchors
        self.C = C
        self.L_min = L_min
        self.L_max = L_max
        self.use_oklch = use_oklch
        self.smooth_interpolation = smooth_interpolation

    def _compute_modulus_modulation(self, r: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Compute modulus-based value modulation with sigmoid default."""
        if self.r_linear_step is not None:
            return sawtooth(r, self.r_linear_step)
        elif self.r_log_base is not None:
            return sawtooth_log(r, self.r_log_base)
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                return sigmoid(np.log(r), center=0, scale=2)

    def _compute_colors(self, z: np.ndarray, phi: np.ndarray, r: np.ndarray,
                       V_phi: np.ndarray, V_r: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute HSV colors for FourQuadrant colormap."""
        # Map phase to hue via 4-point interpolation
        phi_norm = phi / (2 * np.pi)  # Normalize to [0, 1]

        # Find which quadrant we're in
        quadrant = np.floor(phi_norm * 4).astype(int) % 4
        t_local = (phi_norm * 4) % 1  # Local interpolation parameter

        # Initialize hue array
        shape = np.broadcast(z).shape
        H_deg = np.zeros(shape)

        # Interpolate within each quadrant
        for q in range(4):
            mask = (quadrant == q)
            if np.any(mask):
                h1 = self.H_anchors[q]
                h2 = self.H_anchors[(q + 1) % 4]

                if self.smooth_interpolation:
                    # Smooth interpolation using raised cosine
                    t_smooth = 0.5 - 0.5 * np.cos(np.pi * t_local[mask])
                    H_deg[mask] = h1 + (h2 - h1) * t_smooth
                else:
                    # Linear interpolation
                    H_deg[mask] = h1 + (h2 - h1) * t_local[mask]

        if self.use_oklch:
            # Base lightness from modulus
            L_base = self.L_min + (self.L_max - self.L_min) * V_r

            # Combine phase sectors with modulus
            if self.phi is not None:
                # Mix phase and modulus modulation for clear phase sectors
                combined_mod = self._combine_modulations(V_phi, V_r)
                L = self.L_min + (self.L_max - self.L_min) * combined_mod
            else:
                L = L_base

            # Convert to RGB
            R, G, B = oklch_to_srgb(L, self.C, H_deg)

            # Convert RGB to HSV
            hsv = mcolors.rgb_to_hsv(np.stack([R, G, B], axis=-1))
            return hsv[..., 0], hsv[..., 1], hsv[..., 2]
        else:
            # Simple HSV version
            H = H_deg / 360  # Convert to [0, 1]
            S = np.full_like(z, 0.5, dtype=float)

            # Base value from modulus
            V_base = self.L_min + (self.L_max - self.L_min) * V_r

            # Combine phase sectors with modulus
            if self.phi is not None:
                # Mix phase and modulus modulation for clear phase sectors
                combined_mod = self._combine_modulations(V_phi, V_r)
                V = self.L_min + (self.L_max - self.L_min) * combined_mod
            else:
                V = V_base

            return H, S, V