"""Color space conversion utilities for advanced colormaps.

This module provides conversion functions between various color spaces
used by the new colormap families, including OkLCh, HSL, and Cubehelix.
"""

from typing import Union, Tuple
import numpy as np


def oklch_to_srgb(L: Union[float, np.ndarray],
                   C: Union[float, np.ndarray],
                   H: Union[float, np.ndarray],
                   clip_method: str = 'adaptive_chroma') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert OkLCh color to sRGB with perceptually-aware gamut mapping.

    OkLCh is a perceptually uniform color space based on OkLab.
    L = lightness [0, 1], C = chroma [0, ~0.4], H = hue [0, 360].

    When colors fall outside the sRGB gamut, this function uses adaptive
    chroma reduction to preserve hue and lightness, minimizing visual artifacts.

    Parameters
    ----------
    L : float or np.ndarray
        Lightness in [0, 1].
    C : float or np.ndarray
        Chroma (saturation) typically in [0, 0.4].
    H : float or np.ndarray
        Hue in degrees [0, 360].
    clip_method : str, optional
        Gamut mapping method:
        - 'adaptive_chroma': Reduce chroma to fit in gamut (default, best quality)
        - 'simple': Direct clipping (faster but may cause hue shifts)

    Returns
    -------
    R, G, B : tuple of np.ndarray
        RGB values in [0, 1].

    Notes
    -----
    Based on Björn Ottosson's OkLab color space (2020).
    https://bottosson.github.io/posts/oklab/

    The adaptive_chroma method performs binary search to find the maximum
    chroma that fits within the sRGB gamut, preserving hue and lightness.
    """
    L = np.asarray(L)
    C = np.asarray(C)
    H = np.asarray(H)

    def oklch_to_linear_rgb(L_val, C_val, H_val):
        """Helper to convert OkLCh to linear RGB."""
        # Convert to OkLab
        H_rad = np.deg2rad(H_val)
        a = C_val * np.cos(H_rad)
        b = C_val * np.sin(H_rad)

        # OkLab to linear RGB
        l_ = L_val + 0.3963377774 * a + 0.2158037573 * b
        m_ = L_val - 0.1055613458 * a - 0.0638541728 * b
        s_ = L_val - 0.0894841775 * a - 1.2914855480 * b

        l = l_ * l_ * l_
        m = m_ * m_ * m_
        s = s_ * s_ * s_

        r_linear = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
        g_linear = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
        b_linear = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s

        return r_linear, g_linear, b_linear

    def linear_to_srgb(c: np.ndarray) -> np.ndarray:
        """Convert linear RGB to sRGB with gamma correction."""
        c_safe = np.maximum(c, 0)
        return np.where(c_safe <= 0.0031308,
                        12.92 * c_safe,
                        1.055 * np.power(c_safe, 1/2.4) - 0.055)

    if clip_method == 'adaptive_chroma':
        # Adaptive chroma reduction for out-of-gamut colors
        r_linear, g_linear, b_linear = oklch_to_linear_rgb(L, C, H)

        # Check if color is out of gamut
        out_of_gamut = (r_linear < 0) | (r_linear > 1) | \
                       (g_linear < 0) | (g_linear > 1) | \
                       (b_linear < 0) | (b_linear > 1)

        # For out-of-gamut colors, reduce chroma using binary search
        if np.any(out_of_gamut):
            C_adjusted = C.copy()

            # Binary search for maximum chroma that fits in gamut
            C_low = np.zeros_like(C)
            C_high = C.copy()

            for _ in range(15):  # 15 iterations gives ~0.003% precision
                C_mid = (C_low + C_high) / 2
                r_test, g_test, b_test = oklch_to_linear_rgb(L, C_mid, H)

                in_gamut = (r_test >= 0) & (r_test <= 1) & \
                          (g_test >= 0) & (g_test <= 1) & \
                          (b_test >= 0) & (b_test <= 1)

                # Update search bounds
                C_low = np.where(in_gamut & out_of_gamut, C_mid, C_low)
                C_high = np.where(~in_gamut & out_of_gamut, C_mid, C_high)

            # Use the adjusted chroma for out-of-gamut colors
            C_adjusted = np.where(out_of_gamut, C_low, C)

            # Recompute with adjusted chroma
            r_linear, g_linear, b_linear = oklch_to_linear_rgb(L, C_adjusted, H)

        # Convert to sRGB
        R = linear_to_srgb(r_linear)
        G = linear_to_srgb(g_linear)
        B = linear_to_srgb(b_linear)

        # Final safety clip (should rarely be needed)
        R = np.clip(R, 0, 1)
        G = np.clip(G, 0, 1)
        B = np.clip(B, 0, 1)

    elif clip_method == 'simple':
        # Simple clipping (faster but may cause hue shifts)
        r_linear, g_linear, b_linear = oklch_to_linear_rgb(L, C, H)

        R = linear_to_srgb(r_linear)
        G = linear_to_srgb(g_linear)
        B = linear_to_srgb(b_linear)

        R = np.clip(R, 0, 1)
        G = np.clip(G, 0, 1)
        B = np.clip(B, 0, 1)

    else:
        raise ValueError(f"Unknown clip_method: {clip_method}. "
                        "Use 'adaptive_chroma' or 'simple'.")

    return R, G, B


def hsl_to_rgb(H: Union[float, np.ndarray],
               S: Union[float, np.ndarray],
               L: Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert HSL color to RGB.
    
    Parameters
    ----------
    H : float or np.ndarray
        Hue in [0, 1] (0=red, 1/3=green, 2/3=blue).
    S : float or np.ndarray
        Saturation in [0, 1].
    L : float or np.ndarray
        Lightness in [0, 1].
        
    Returns
    -------
    R, G, B : tuple of np.ndarray
        RGB values in [0, 1].
    """
    H = np.asarray(H)
    S = np.asarray(S)
    L = np.asarray(L)
    
    C = (1 - np.abs(2 * L - 1)) * S
    X = C * (1 - np.abs((H * 6) % 2 - 1))
    m = L - C / 2
    
    # Determine RGB based on hue sector
    H6 = H * 6
    sector = np.floor(H6).astype(int) % 6
    
    # Initialize RGB arrays
    shape = np.broadcast(H, S, L).shape
    R = np.zeros(shape)
    G = np.zeros(shape)
    B = np.zeros(shape)
    
    # Sector 0: 0° - 60°
    mask = (sector == 0)
    R[mask] = C[mask]
    G[mask] = X[mask]
    
    # Sector 1: 60° - 120°
    mask = (sector == 1)
    R[mask] = X[mask]
    G[mask] = C[mask]
    
    # Sector 2: 120° - 180°
    mask = (sector == 2)
    G[mask] = C[mask]
    B[mask] = X[mask]
    
    # Sector 3: 180° - 240°
    mask = (sector == 3)
    G[mask] = X[mask]
    B[mask] = C[mask]
    
    # Sector 4: 240° - 300°
    mask = (sector == 4)
    R[mask] = X[mask]
    B[mask] = C[mask]
    
    # Sector 5: 300° - 360°
    mask = (sector == 5)
    R[mask] = C[mask]
    B[mask] = X[mask]
    
    # Add lightness component
    R = R + m
    G = G + m
    B = B + m
    
    return R, G, B


def cubehelix(h: Union[float, np.ndarray],
              s: float = 0.5,
              r: float = -1.5,
              gamma: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate colors using Dave Green's cubehelix scheme.
    
    Cubehelix produces a rainbow colormap with monotonic lightness.
    It's designed to be perceived as increasing in intensity and
    prints well in grayscale.
    
    Parameters
    ----------
    h : float or np.ndarray
        Position along helix in [0, 1].
    s : float, optional
        Saturation (0=grayscale, 1=saturated).
    r : float, optional
        Number of rotations through color space.
    gamma : float, optional
        Gamma correction factor.
        
    Returns
    -------
    R, G, B : tuple of np.ndarray
        RGB values in [0, 1].
        
    References
    ----------
    Green, D. A., 2011, "A colour scheme for the display of 
    astronomical intensity images", Bull. Astr. Soc. India, 39, 289.
    """
    h = np.asarray(h)
    
    # Apply gamma
    h_gamma = np.power(h, gamma)
    
    # Calculate angle and amplitude
    angle = 2 * np.pi * (s / 3 + r * h)
    amp = s * h_gamma * (1 - h_gamma) / 2
    
    # RGB weights
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    R = h_gamma + amp * (-0.14861 * cos_a + 1.78277 * sin_a)
    G = h_gamma + amp * (-0.29227 * cos_a - 0.90649 * sin_a)
    B = h_gamma + amp * (+1.97294 * cos_a)
    
    # Clip to valid range
    R = np.clip(R, 0, 1)
    G = np.clip(G, 0, 1)
    B = np.clip(B, 0, 1)
    
    return R, G, B


def interpolate_hue(h1: float, h2: float, 
                    t: Union[float, np.ndarray],
                    direction: str = 'shortest') -> Union[float, np.ndarray]:
    """Interpolate between two hue values.
    
    Parameters
    ----------
    h1, h2 : float
        Hue values in [0, 1].
    t : float or np.ndarray
        Interpolation parameter(s) in [0, 1].
    direction : str, optional
        'shortest', 'longest', 'clockwise', or 'counter-clockwise'.
        
    Returns
    -------
    float or np.ndarray
        Interpolated hue in [0, 1].
    """
    t = np.asarray(t)
    
    if direction == 'shortest':
        # Find shortest path around circle
        diff = h2 - h1
        if diff > 0.5:
            diff = diff - 1
        elif diff < -0.5:
            diff = diff + 1
        h = h1 + t * diff
        
    elif direction == 'longest':
        # Find longest path around circle
        diff = h2 - h1
        if -0.5 <= diff <= 0.5:
            if diff >= 0:
                diff = diff - 1
            else:
                diff = diff + 1
        h = h1 + t * diff
        
    elif direction == 'clockwise':
        diff = h2 - h1
        if diff < 0:
            diff = diff + 1
        h = h1 + t * diff
        
    elif direction == 'counter-clockwise':
        diff = h2 - h1
        if diff > 0:
            diff = diff - 1
        h = h1 + t * diff
        
    else:
        raise ValueError(f"Unknown direction: {direction}")
    
    # Ensure result is in [0, 1]
    return np.mod(h, 1.0)


def clip_to_gamut(R: np.ndarray, G: np.ndarray, B: np.ndarray,
                  preserve: str = 'hue') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clip RGB values to valid gamut while preserving appearance.
    
    Parameters
    ----------
    R, G, B : np.ndarray
        RGB values (may be outside [0, 1]).
    preserve : str, optional
        What to preserve: 'hue', 'lightness', or 'chroma'.
        
    Returns
    -------
    R, G, B : tuple of np.ndarray
        Clipped RGB values in [0, 1].
    """
    if preserve == 'hue':
        # Simple clipping preserves hue reasonably well
        R = np.clip(R, 0, 1)
        G = np.clip(G, 0, 1)
        B = np.clip(B, 0, 1)
        
    elif preserve == 'lightness':
        # Scale all channels equally to preserve ratios
        max_val = np.maximum(np.maximum(R, G), B)
        scale = np.where(max_val > 1, 1 / max_val, 1)
        R = R * scale
        G = G * scale
        B = B * scale
        
        # Also handle negative values
        min_val = np.minimum(np.minimum(R, G), B)
        offset = np.where(min_val < 0, -min_val, 0)
        R = R + offset
        G = G + offset
        B = B + offset
        
        # Final clip for safety
        R = np.clip(R, 0, 1)
        G = np.clip(G, 0, 1)
        B = np.clip(B, 0, 1)
        
    elif preserve == 'chroma':
        # Desaturate toward gray to keep in gamut
        gray = 0.299 * R + 0.587 * G + 0.114 * B
        
        # Find how much we need to desaturate
        t = np.ones_like(R)
        
        # Check upper bound
        for channel in [R, G, B]:
            mask = channel > 1
            t_needed = (1 - gray[mask]) / (channel[mask] - gray[mask])
            t[mask] = np.minimum(t[mask], t_needed)
        
        # Check lower bound
        for channel in [R, G, B]:
            mask = channel < 0
            t_needed = gray[mask] / (gray[mask] - channel[mask])
            t[mask] = np.minimum(t[mask], t_needed)
        
        # Apply desaturation
        R = gray + t * (R - gray)
        G = gray + t * (G - gray)
        B = gray + t * (B - gray)
        
    else:
        raise ValueError(f"Unknown preserve mode: {preserve}")
    
    return R, G, B