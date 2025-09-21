"""High-level API for common complexplorer workflows.

This module provides convenient functions for typical use cases.
"""

from typing import Callable, Optional, Any
import numpy as np

from complexplorer.core.domain import Domain, Rectangle
from complexplorer.core.colormap import Colormap, Phase
from complexplorer.core.scaling import ModulusScaling, get_scaling_preset

# Import plotting functions
from complexplorer.plotting.matplotlib.plot_2d import plot as plot_2d
from complexplorer.plotting.matplotlib.plot_3d import plot_landscape as plot_3d_landscape
from complexplorer.plotting.matplotlib.plot_3d import riemann as plot_riemann

# Try to import PyVista functions
try:
    from complexplorer.plotting.pyvista.plot_3d import plot_landscape_pv
    from complexplorer.plotting.pyvista.riemann import riemann_pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False


def show(func: Callable[[complex], complex],
         x_range: tuple = (-2, 2, 500),
         y_range: Optional[tuple] = None,
         **kwargs) -> Any:
    """Quick plot function for simple use cases, similar to cplot.
    
    Parameters
    ----------
    func : callable
        Complex function to visualize
    x_range : tuple
        (min, max) or (min, max, resolution) for real axis
    y_range : tuple, optional
        (min, max) or (min, max, resolution) for imaginary axis. 
        If None, uses x_range.
    **kwargs
        Additional arguments passed to plot()
    
    Returns
    -------
    Any
        Matplotlib axes or PyVista plotter depending on mode.
        
    Examples
    --------
    >>> import complexplorer as cp
    >>> cp.show(lambda z: z**2)
    >>> cp.show(lambda z: 1/z, (-3, 3, 500), (-3, 3, 500))
    >>> cp.show(lambda z: np.sin(z), (-5, 5), (-5, 5))
    """
    # Handle y_range default
    if y_range is None:
        y_range = x_range
    
    # Parse ranges - support both (min, max) and (min, max, resolution)
    if len(x_range) == 2:
        x_min, x_max = x_range
        x_res = 500
    elif len(x_range) == 3:
        x_min, x_max, x_res = x_range
    else:
        raise ValueError("x_range must be (min, max) or (min, max, resolution)")
    
    if len(y_range) == 2:
        y_min, y_max = y_range
        y_res = 500
    elif len(y_range) == 3:
        y_min, y_max, y_res = y_range
    else:
        raise ValueError("y_range must be (min, max) or (min, max, resolution)")
    
    # Create Rectangle domain from ranges
    center = complex((x_min + x_max) / 2, (y_min + y_max) / 2)
    re_length = abs(x_max - x_min)
    im_length = abs(y_max - y_min)
    domain = Rectangle(re_length, im_length, center=center, square=False)
    
    # Use the maximum resolution
    resolution = max(x_res, y_res) if isinstance(x_res, int) and isinstance(y_res, int) else 500
    
    # Set default colormap if not provided (with better defaults for initial experience)
    if 'cmap' not in kwargs:
        kwargs['cmap'] = Phase(n_phi=6, auto_scale_r=True, scale_radius=0.8)
    
    # Get mode (default to 2d for simplicity)
    mode = kwargs.pop('mode', '2d')
    
    # Pass resolution if not in kwargs
    if 'resolution' not in kwargs:
        kwargs['resolution'] = resolution
    
    # Call the main plot function
    return plot(func, domain, mode, **kwargs)


def plot(func: Callable[[complex], complex], 
         domain: Optional[Domain] = None,
         mode: str = '2d',
         **kwargs) -> Any:
    """Visualize a complex function.
    
    Parameters
    ----------
    func : callable
        Complex function to visualize
    domain : Domain, optional
        Domain to plot. Defaults to Rectangle(4, 4)
    mode : str
        Plot mode: '2d', '3d', 'riemann'
    **kwargs
        Additional arguments passed to plotting function
        
    Returns
    -------
    Any
        Matplotlib axes or PyVista plotter depending on mode and backend.
    """
    if domain is None:
        domain = Rectangle(4, 4)
    
    if 'cmap' not in kwargs:
        kwargs['cmap'] = Phase(n_phi=6, auto_scale_r=True, scale_radius=0.8)
    
    if mode == '2d':
        return plot_2d(domain, func, **kwargs)
    elif mode == '3d':
        if HAS_PYVISTA and kwargs.get('backend', 'matplotlib') == 'pyvista':
            return plot_landscape_pv(domain, func, **kwargs)
        else:
            return plot_3d_landscape(domain, func=func, **kwargs)
    elif mode == 'riemann':
        if HAS_PYVISTA and kwargs.get('backend', 'matplotlib') == 'pyvista':
            return riemann_pv(func, **kwargs)
        else:
            return plot_riemann(func, **kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}")


# Removed analyze_function - incomplete implementation
# Users can use plot() directly with appropriate colormap


# Removed create_animation - not implemented
# Will be added in a future release when properly implemented


# Removed compare_functions - not implemented  
# Will be added in a future release when properly implemented


# Preset configurations for common use cases
def publication_preset() -> dict:
    """Get settings for publication-quality figures.
    
    Returns
    -------
    dict
        Configuration with high-resolution enhanced phase portrait.
    """
    return {
        'cmap': Phase(n_phi=12, auto_scale_r=True, scale_radius=0.8),
        'resolution': 800
    }


def interactive_preset() -> dict:
    """Get settings for interactive exploration.
    
    Returns
    -------
    dict
        Configuration optimized for speed and interactivity.
    """
    return {
        'cmap': Phase(n_phi=6, auto_scale_r=True, scale_radius=0.8),
        'resolution': 500
    }


def high_contrast_preset() -> dict:
    """Get settings for high contrast visualization.
    
    Returns
    -------
    dict
        Configuration with many phase sectors for maximum contrast.
    """
    return {
        'cmap': Phase(n_phi=16, auto_scale_r=True, scale_radius=0.5),
        'resolution': 600
    }


# Removed redundant aliases - one clear function name is better


__all__ = [
    'show',
    'plot',
    'publication_preset',
    'interactive_preset',
    'high_contrast_preset'
]