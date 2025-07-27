"""Direct texture computation based on colormap patterns.

This module computes textures directly from the same mathematical
patterns that colormaps use, avoiding unreliable gradient detection.
"""

from typing import Tuple, Optional
import numpy as np
from ..core.colormap import (
    Colormap, Phase, Chessboard, PolarChessboard, LogRings
)
from ..core.functions import phase as phase_func, sawtooth, sawtooth_log


def compute_texture_direct(
    f_values: np.ndarray,
    cmap: Colormap,
    mode: str = 'ridges'
) -> np.ndarray:
    """Compute texture directly from colormap patterns.
    
    Instead of detecting edges from HSV values, this computes
    boundaries directly using the same logic as the colormaps.
    
    Parameters
    ----------
    f_values : ndarray
        Complex function values at mesh points.
    cmap : Colormap
        Colormap instance.
    mode : str
        Texture mode: 'ridges', 'grooves', or 'binary'.
        
    Returns
    -------
    ndarray
        Displacement values in range [-1, 1].
    """
    if isinstance(cmap, Chessboard):
        return _chessboard_texture(f_values, cmap, mode)
    elif isinstance(cmap, PolarChessboard):
        return _polar_chessboard_texture(f_values, cmap, mode)
    elif isinstance(cmap, LogRings):
        return _log_rings_texture(f_values, cmap, mode)
    elif isinstance(cmap, Phase):
        return _phase_texture(f_values, cmap, mode)
    else:
        # Unknown colormap - fall back to zero texture
        return np.zeros_like(f_values, dtype=float)


def _chessboard_texture(
    z: np.ndarray,
    cmap: Chessboard,
    mode: str
) -> np.ndarray:
    """Compute texture for Chessboard colormap."""
    # Shift and scale
    z_shifted = (z - cmap.center) / cmap.spacing
    
    # Get indices
    with np.errstate(invalid='ignore'):
        real_idx = np.floor(np.real(z_shifted))
        imag_idx = np.floor(np.imag(z_shifted))
    
    if mode == 'binary':
        # Direct height mapping
        is_white = ((real_idx + imag_idx) % 2 == 0)
        return np.where(is_white, 1.0, -1.0)
    
    else:  # ridges or grooves
        # Compute boundaries
        # Boundaries occur at integer values of real_idx and imag_idx
        real_frac = np.real(z_shifted) - real_idx
        imag_frac = np.imag(z_shifted) - imag_idx
        
        # Distance to nearest boundary
        real_dist = np.minimum(real_frac, 1 - real_frac)
        imag_dist = np.minimum(imag_frac, 1 - imag_frac)
        min_dist = np.minimum(real_dist, imag_dist)
        
        # Create ridges/grooves at boundaries
        boundary_width = 0.1  # Fraction of square size
        is_boundary = min_dist < boundary_width
        
        if mode == 'ridges':
            return is_boundary.astype(float)
        else:  # grooves
            return -is_boundary.astype(float)


def _polar_chessboard_texture(
    z: np.ndarray,
    cmap: PolarChessboard,
    mode: str
) -> np.ndarray:
    """Compute texture for PolarChessboard colormap."""
    # Phase sectors
    angle = np.angle(z)
    angle_normalized = (angle + np.pi) / cmap.phi
    angle_idx = np.floor(angle_normalized)
    
    # Radial rings
    r = np.abs(z) / cmap.spacing
    if cmap.r_log is not None:
        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.log(r) / np.log(cmap.r_log)
    r_idx = np.floor(r)
    
    if mode == 'binary':
        # Direct height mapping
        is_white = ((angle_idx + r_idx) % 2 == 0)
        return np.where(is_white, 1.0, -1.0)
    
    else:  # ridges or grooves
        # Angular boundaries
        angle_frac = angle_normalized - angle_idx
        angle_dist = np.minimum(angle_frac, 1 - angle_frac)
        
        # Radial boundaries
        r_frac = r - r_idx
        r_dist = np.minimum(r_frac, 1 - r_frac)
        
        # Combined boundary detection
        min_dist = np.minimum(angle_dist, r_dist)
        boundary_width = 0.1
        is_boundary = min_dist < boundary_width
        
        if mode == 'ridges':
            return is_boundary.astype(float)
        else:  # grooves
            return -is_boundary.astype(float)


def _log_rings_texture(
    z: np.ndarray,
    cmap: LogRings,
    mode: str
) -> np.ndarray:
    """Compute texture for LogRings colormap."""
    r = np.abs(z)
    
    # Logarithmic rings
    with np.errstate(divide='ignore', invalid='ignore'):
        log_r = np.log(r) / cmap.log_spacing
    
    # Handle origin
    log_r = np.where(r == 0, 0, log_r)
    
    # Get ring index
    ring_idx = np.floor(log_r)
    
    if mode == 'binary':
        # Alternating rings
        is_white = (ring_idx % 2 == 0)
        return np.where(is_white, 1.0, -1.0)
    
    else:  # ridges or grooves
        # Distance to ring boundary
        ring_frac = log_r - ring_idx
        ring_dist = np.minimum(ring_frac, 1 - ring_frac)
        
        boundary_width = 0.1
        is_boundary = ring_dist < boundary_width
        
        if mode == 'ridges':
            return is_boundary.astype(float)
        else:  # grooves
            return -is_boundary.astype(float)


def _phase_texture(
    z: np.ndarray,
    cmap: Phase,
    mode: str
) -> np.ndarray:
    """Compute texture for Phase colormap with enhancements."""
    displacement = np.zeros_like(z, dtype=float)
    
    # Phase-based boundaries
    if cmap.n_phi is not None:
        phi = phase_func(z)  # Returns values in [0, 2π)
        # Phase sectors - boundaries are evenly spaced in [0, 2π)
        # For n_phi sectors, boundaries are at k * 2π/n_phi for k = 0, 1, ..., n_phi-1
        sector_width = 2 * np.pi / cmap.n_phi
        
        # Find minimum distance to any sector boundary
        min_dist = np.full_like(phi, np.inf)
        for k in range(cmap.n_phi):
            boundary = k * sector_width
            # Calculate circular distance
            dist = np.abs(phi - boundary)
            # Handle wraparound at 0/2π
            dist = np.minimum(dist, 2*np.pi - dist)
            min_dist = np.minimum(min_dist, dist)
        
        # Create texture at boundaries
        boundary_width = sector_width * 0.1  # 10% of sector width
        phase_boundary = min_dist < boundary_width
        
        if mode == 'ridges':
            displacement = np.maximum(displacement, phase_boundary.astype(float))
        elif mode == 'grooves':
            displacement = np.minimum(displacement, -phase_boundary.astype(float))
    
    # Modulus-based boundaries
    if cmap.r_linear_step is not None:
        r = np.abs(z)
        # Linear rings at multiples of r_linear_step
        r_normalized = r / cmap.r_linear_step
        r_idx = np.floor(r_normalized)
        r_frac = r_normalized - r_idx
        
        # Distance to nearest ring
        r_dist = np.minimum(r_frac, 1 - r_frac)
        boundary_width = 0.1
        modulus_boundary = r_dist < boundary_width
        
        if mode == 'ridges':
            displacement = np.maximum(displacement, modulus_boundary.astype(float))
        elif mode == 'grooves':
            displacement = np.minimum(displacement, -modulus_boundary.astype(float))
    
    elif cmap.r_log_base is not None:
        r = np.abs(z)
        # Logarithmic rings
        with np.errstate(divide='ignore', invalid='ignore'):
            log_r = np.log(r) / np.log(cmap.r_log_base)
        log_r = np.where(r == 0, 0, log_r)
        
        r_idx = np.floor(log_r)
        r_frac = log_r - r_idx
        r_dist = np.minimum(r_frac, 1 - r_frac)
        
        boundary_width = 0.1
        modulus_boundary = r_dist < boundary_width
        
        if mode == 'ridges':
            displacement = np.maximum(displacement, modulus_boundary.astype(float))
        elif mode == 'grooves':
            displacement = np.minimum(displacement, -modulus_boundary.astype(float))
    
    # For binary mode with enhanced phase, create a pattern
    if mode == 'binary' and (cmap.n_phi or cmap.r_linear_step or cmap.r_log_base):
        # Create alternating pattern based on phase sectors and modulus rings
        pattern = np.zeros_like(z, dtype=float)
        
        if cmap.n_phi:
            phi = phase_func(z)  # [0, 2π)
            sector_width = 2 * np.pi / cmap.n_phi
            phi_idx = np.floor(phi / sector_width).astype(int)
            pattern += phi_idx
        
        if cmap.r_linear_step:
            r = np.abs(z)
            r_idx = np.floor(r / cmap.r_linear_step).astype(int)
            pattern += r_idx
        
        displacement = np.where((pattern % 2) == 0, 1.0, -1.0)
    
    return displacement