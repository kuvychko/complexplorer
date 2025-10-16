"""Color export utilities for complex visualization.

This module provides functions to directly export colors from complex values
using any colormap, useful for custom visualizations or data export.
"""

from typing import Union, Tuple, Optional
import numpy as np
from complexplorer.core.colormap import Colormap, Phase


def get_color(z: Union[complex, np.ndarray], 
              cmap: Optional[Colormap] = None,
              format: str = 'rgb') -> Union[Tuple[float, float, float], np.ndarray]:
    """Get color(s) for complex value(s) using specified colormap.
    
    Parameters
    ----------
    z : complex or np.ndarray
        Complex value(s) to convert to colors.
    cmap : Colormap, optional
        Colormap to use. Defaults to Phase colormap.
    format : str, optional
        Output format: 'rgb', 'hsv', or 'hex'. Default is 'rgb'.
        
    Returns
    -------
    tuple or np.ndarray
        For scalar input: (R, G, B) tuple or (H, S, V) tuple in [0, 1].
        For array input: Array of colors with shape (*z.shape, 3).
        For hex format: Hex string(s) like '#RRGGBB'.
        
    Examples
    --------
    >>> import complexplorer as cp
    >>> import numpy as np
    >>> 
    >>> # Get color for a single complex number
    >>> color = cp.get_color(1 + 1j)
    >>> print(f"RGB: {color}")
    >>> 
    >>> # Get colors for an array
    >>> z = np.array([1, 1j, -1, -1j])
    >>> colors = cp.get_color(z)
    >>> 
    >>> # Use custom colormap
    >>> cmap = cp.OklabPhase(phase_sectors=8, enhanced=True)
    >>> color = cp.get_color(np.exp(1j * np.pi/4), cmap=cmap)
    >>> 
    >>> # Get hex colors
    >>> hex_color = cp.get_color(1j, format='hex')
    >>> print(f"Hex: {hex_color}")
    """
    # Default colormap
    if cmap is None:
        cmap = Phase(phase_sectors=6, auto_scale_r=True)
    
    # Convert to numpy array for uniform handling
    z_array = np.asarray(z)
    is_scalar = z_array.ndim == 0
    
    # Get colors based on format
    if format.lower() == 'hsv':
        colors = cmap.hsv(z_array)
    elif format.lower() == 'rgb':
        colors = cmap.rgb(z_array)
    elif format.lower() == 'hex':
        rgb = cmap.rgb(z_array)
        if is_scalar:
            # Convert single RGB to hex
            r, g, b = (int(255 * c) for c in rgb)
            return f"#{r:02x}{g:02x}{b:02x}"
        else:
            # Convert array of RGB to hex strings
            hex_colors = []
            rgb_flat = rgb.reshape(-1, 3)
            for color in rgb_flat:
                r, g, b = (int(255 * c) for c in color)
                hex_colors.append(f"#{r:02x}{g:02x}{b:02x}")
            return np.array(hex_colors).reshape(z_array.shape)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'rgb', 'hsv', or 'hex'.")
    
    # Return tuple for scalar input
    if is_scalar:
        return tuple(colors)
    
    return colors


def get_color_array(z: np.ndarray,
                   cmap: Optional[Colormap] = None) -> np.ndarray:
    """Get RGB color array for complex array (optimized version).
    
    This is an optimized version for large arrays, returning RGB values
    directly as a numpy array without conversion overhead.
    
    Parameters
    ----------
    z : np.ndarray
        Complex array to convert.
    cmap : Colormap, optional
        Colormap to use. Defaults to Phase colormap.
        
    Returns
    -------
    np.ndarray
        RGB array with shape (*z.shape, 3), values in [0, 1].
        
    Examples
    --------
    >>> import complexplorer as cp
    >>> import numpy as np
    >>> 
    >>> # Create a grid of complex numbers
    >>> x = np.linspace(-2, 2, 100)
    >>> y = np.linspace(-2, 2, 100)
    >>> X, Y = np.meshgrid(x, y)
    >>> Z = X + 1j * Y
    >>> 
    >>> # Get colors for the entire grid
    >>> colors = cp.get_color_array(Z)
    >>> print(f"Color array shape: {colors.shape}")  # (100, 100, 3)
    """
    if cmap is None:
        cmap = Phase(phase_sectors=6, auto_scale_r=True)
    
    return cmap.rgb(z)


def interpolate_colormap(z: Union[complex, np.ndarray],
                        cmap1: Colormap,
                        cmap2: Colormap,
                        t: float = 0.5) -> np.ndarray:
    """Interpolate between two colormaps.
    
    Useful for creating smooth transitions between different
    visualization styles.
    
    Parameters
    ----------
    z : complex or np.ndarray
        Complex value(s) to visualize.
    cmap1 : Colormap
        First colormap.
    cmap2 : Colormap
        Second colormap.
    t : float
        Interpolation parameter in [0, 1].
        0 gives cmap1, 1 gives cmap2.
        
    Returns
    -------
    np.ndarray
        Interpolated RGB colors.
        
    Examples
    --------
    >>> import complexplorer as cp
    >>> 
    >>> # Blend between Phase and OklabPhase
    >>> cmap1 = cp.Phase(phase_sectors=6)
    >>> cmap2 = cp.OklabPhase(phase_sectors=6, enhanced=False)
    >>> 
    >>> # Get 50% blend
    >>> color = cp.interpolate_colormap(1 + 1j, cmap1, cmap2, t=0.5)
    """
    if not 0 <= t <= 1:
        raise ValueError("Interpolation parameter t must be in [0, 1]")
    
    rgb1 = cmap1.rgb(z)
    rgb2 = cmap2.rgb(z)
    
    return (1 - t) * rgb1 + t * rgb2


__all__ = ['get_color', 'get_color_array', 'interpolate_colormap']