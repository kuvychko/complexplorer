"""
PyVista-based 3D visualization functions for complex functions.

This module provides high-performance, interactive 3D visualizations using PyVista
as an alternative to the matplotlib-based plots_3d module. All functions support
both interactive (default) and static modes.

Functions:
----------
- `plot_landscape_pv`: Plot a complex function as a 3D landscape using PyVista
- `pair_plot_landscape_pv`: Plot domain and codomain landscapes side-by-side
- `riemann_pv`: Plot a complex function on the Riemann sphere (requires mesh_utils)
"""

import numpy as np
import pyvista as pv
from typing import Callable, Optional, Union, Tuple
import warnings

try:
    from matplotlib.colors import hsv_to_rgb
except ImportError:
    warnings.warn("matplotlib not available, some color functions may be limited")
    hsv_to_rgb = None

from complexplorer.cmap import Phase, Cmap
from complexplorer.domain import Domain


def _ensure_pyvista_setup():
    """Ensure PyVista is properly configured for the current environment."""
    backend = pv.global_theme.jupyter_backend
    if backend is None:
        # Not in Jupyter, use default
        pass
    elif backend != 'trame':
        warnings.warn(
            f"PyVista backend is '{backend}', but 'trame' is recommended for "
            "interactive Jupyter visualizations. Set with: pv.set_jupyter_backend('trame')"
        )


def _create_complex_surface(
    domain: Optional[Domain],
    func: Optional[Callable],
    z: Optional[np.ndarray] = None,
    f: Optional[np.ndarray] = None,
    n: int = 100,
    cmap: Optional[Cmap] = None,
    z_scale: float = 1.0,
    log_z: bool = False,
    z_max: Optional[float] = None
) -> Tuple[pv.StructuredGrid, np.ndarray]:
    """
    Create a PyVista mesh for a complex function surface.
    
    Returns:
        grid: PyVista StructuredGrid
        rgb_colors: RGB color array
    """
    # Validate inputs
    if domain is None and z is None:
        raise ValueError('both domain and z parameters cannot be None')
    if f is None and func is None:
        raise ValueError('both f and func parameters cannot be None')
    
    # Get domain mesh
    if z is None:
        z = domain.mesh(n)
        mask = domain.outmask(n)
    else:
        mask = None
    
    # Evaluate function
    if f is None:
        f = func(z)
    
    # Ensure f is array
    f = np.asarray(f)
    if f.ndim == 0:  # scalar case
        f = np.full_like(z, f)
    
    # Apply mask
    if mask is not None:
        f[mask] = np.nan
    
    # Calculate height (magnitude)
    magnitude = np.abs(f)
    if z_max is not None:
        magnitude = np.clip(magnitude, 0, z_max)
    
    if log_z:
        with np.errstate(divide='ignore', invalid='ignore'):
            height = np.log1p(magnitude) * z_scale
    else:
        height = magnitude * z_scale
    
    # Create structured grid
    X = np.real(z)
    Y = np.imag(z)
    Z = height
    
    # Handle NaN values for masked regions
    if mask is not None:
        Z[mask] = np.nan
    
    grid = pv.StructuredGrid(X, Y, Z)
    
    # Get colors from colormap
    if cmap is None:
        cmap = Phase(6, 0.6)
    
    rgb = cmap.rgb(f, outmask=mask)
    
    # Flatten and add to grid
    rgb_flat = rgb.reshape(-1, 3)
    grid["RGB"] = rgb_flat
    
    # Also store magnitude and phase as scalars
    grid["magnitude"] = magnitude.ravel()
    grid["phase"] = np.angle(f).ravel()
    
    return grid, rgb


def plot_landscape_pv(
    domain: Optional[Domain] = None,
    func: Optional[Callable] = None,
    z: Optional[np.ndarray] = None,
    f: Optional[np.ndarray] = None,
    n: int = 100,
    cmap: Optional[Cmap] = None,
    interactive: bool = True,
    camera_position: Union[str, Tuple] = 'iso',
    show_edges: bool = False,
    edge_color: str = 'gray',
    z_scale: float = 1.0,
    log_z: bool = False,
    z_max: Optional[float] = None,
    window_size: Tuple[int, int] = (800, 600),
    title: Optional[str] = None,
    filename: Optional[str] = None,
    return_plotter: bool = False,
    **kwargs
) -> Optional[pv.Plotter]:
    """
    Plot a complex function as a 3D landscape using PyVista.
    
    This function provides high-performance, interactive 3D visualization with
    accurate per-vertex coloring (no interpolation artifacts).
    
    Parameters
    ----------
    domain : Domain, optional
        Domain object. If None, z must be provided.
    func : Callable, optional
        Complex function. If None, f must be provided.
    z : np.ndarray, optional
        2D array of complex numbers (domain). If None, domain must be provided.
    f : np.ndarray, optional
        2D array of complex numbers (codomain). If None, func must be provided.
    n : int, default=100
        Number of points along the longest side of the domain mesh.
    cmap : Cmap, optional
        Color map to use. Default is Phase(6, 0.6).
    interactive : bool, default=True
        If True, show interactive widget. If False, render static image.
    camera_position : str or tuple, default='iso'
        Camera position: 'iso', 'xy', 'xz', 'yz', or custom (position, focal_point, up).
    show_edges : bool, default=False
        If True, show mesh edges.
    edge_color : str, default='gray'
        Color of mesh edges.
    z_scale : float, default=1.0
        Scaling factor for height.
    log_z : bool, default=False
        If True, use logarithmic scaling for height.
    z_max : float, optional
        Maximum value for height clipping.
    window_size : tuple, default=(800, 600)
        Window size in pixels.
    title : str, optional
        Title for the plot.
    filename : str, optional
        Save plot to file (PNG, PDF, etc.).
    return_plotter : bool, default=False
        If True, return the plotter object for further customization.
    **kwargs
        Additional arguments passed to plotter.add_mesh().
    
    Returns
    -------
    plotter : pv.Plotter, optional
        Only returned if return_plotter=True.
    
    Examples
    --------
    >>> import complexplorer as cp
    >>> domain = cp.Rectangle(4, 4)
    >>> cp.plot_landscape_pv(domain, lambda z: (z-1)/(z**2+z+1))
    """
    _ensure_pyvista_setup()
    
    # Create mesh
    grid, _ = _create_complex_surface(
        domain, func, z, f, n, cmap, z_scale, log_z, z_max
    )
    
    # Create plotter
    plotter = pv.Plotter(
        off_screen=not interactive,
        window_size=window_size
    )
    
    # Add mesh
    mesh_kwargs = {
        'rgb': True,
        'show_edges': show_edges,
        'edge_color': edge_color,
        'smooth_shading': True,
        'lighting': True,
    }
    mesh_kwargs.update(kwargs)
    
    actor = plotter.add_mesh(grid, **mesh_kwargs)
    
    # Set camera
    plotter.camera_position = camera_position
    
    # Add axes
    plotter.show_axes()
    
    # Add title
    if title:
        plotter.add_text(title, font_size=14)
    
    # Show or save
    if interactive:
        plotter.show()
    else:
        if filename:
            if filename.endswith(('.pdf', '.svg', '.eps')):
                plotter.save_graphic(filename)
            else:
                plotter.screenshot(filename)
        else:
            plotter.show(screenshot=True)
    
    if return_plotter:
        return plotter
    else:
        plotter.close()


def pair_plot_landscape_pv(
    domain: Optional[Domain] = None,
    func: Optional[Callable] = None,
    z: Optional[np.ndarray] = None,
    f: Optional[np.ndarray] = None,
    n: int = 100,
    cmap: Optional[Cmap] = None,
    interactive: bool = True,
    camera_position: Union[str, Tuple] = 'iso',
    show_edges: bool = False,
    edge_color: str = 'gray',
    z_scale: float = 1.0,
    log_z: bool = False,
    z_max: Optional[float] = None,
    window_size: Tuple[int, int] = (1600, 600),
    title: Optional[str] = None,
    filename: Optional[str] = None,
    link_views: bool = True,
    **kwargs
) -> Optional[pv.Plotter]:
    """
    Plot domain and codomain landscapes side-by-side using PyVista.
    
    This creates two 3D surfaces: one showing the identity function on the domain
    (left) and one showing the complex function (right).
    
    Parameters
    ----------
    domain : Domain, optional
        Domain object. If None, z must be provided.
    func : Callable, optional
        Complex function. If None, f must be provided.
    z : np.ndarray, optional
        2D array of complex numbers (domain).
    f : np.ndarray, optional
        2D array of complex numbers (codomain).
    n : int, default=100
        Number of points along the longest side.
    cmap : Cmap, optional
        Color map to use. Default is Phase(6, 0.5).
    interactive : bool, default=True
        If True, show interactive widget.
    camera_position : str or tuple, default='iso'
        Camera position for both views.
    show_edges : bool, default=False
        If True, show mesh edges.
    edge_color : str, default='gray'
        Color of mesh edges.
    z_scale : float, default=1.0
        Scaling factor for height.
    log_z : bool, default=False
        If True, use logarithmic scaling.
    z_max : float, optional
        Maximum value for height clipping.
    window_size : tuple, default=(1600, 600)
        Window size in pixels.
    title : str, optional
        Title for the plot.
    filename : str, optional
        Save plot to file.
    link_views : bool, default=True
        If True, synchronize camera movement between subplots.
    **kwargs
        Additional arguments passed to plotter.add_mesh().
    
    Returns
    -------
    plotter : pv.Plotter, optional
        Only returned if interactive=True and user wants to customize further.
    """
    _ensure_pyvista_setup()
    
    # Default colormap for pair plots
    if cmap is None:
        cmap = Phase(6, 0.5)
    
    # Create meshes
    grid_domain, _ = _create_complex_surface(
        domain, lambda x: x, z, z, n, cmap, z_scale, log_z, z_max
    )
    grid_codomain, _ = _create_complex_surface(
        domain, func, z, f, n, cmap, z_scale, log_z, z_max
    )
    
    # Create plotter with subplots
    plotter = pv.Plotter(
        shape=(1, 2),
        off_screen=not interactive,
        window_size=window_size
    )
    
    # Plot domain (identity)
    plotter.subplot(0, 0)
    plotter.add_mesh(
        grid_domain,
        rgb=True,
        show_edges=show_edges,
        edge_color=edge_color,
        smooth_shading=True,
        **kwargs
    )
    plotter.add_text("Domain", font_size=12)
    plotter.show_axes()
    plotter.camera_position = camera_position
    
    # Plot codomain (function)
    plotter.subplot(0, 1)
    plotter.add_mesh(
        grid_codomain,
        rgb=True,
        show_edges=show_edges,
        edge_color=edge_color,
        smooth_shading=True,
        **kwargs
    )
    plotter.add_text("Codomain", font_size=12)
    plotter.show_axes()
    plotter.camera_position = camera_position
    
    # Link views if requested
    if link_views:
        plotter.link_views()
    
    # Add main title
    if title:
        plotter.add_text(title, font='arial', font_size=16, 
                        position='upper_edge', color='black')
    
    # Show or save
    if interactive:
        plotter.show()
        return plotter
    else:
        if filename:
            if filename.endswith(('.pdf', '.svg', '.eps')):
                plotter.save_graphic(filename)
            else:
                plotter.screenshot(filename)
        else:
            plotter.show(screenshot=True)
        plotter.close()


# Placeholder for riemann_pv - will be implemented after icosahedral meshing
def riemann_pv(
    func: Callable,
    n_subdivisions: int = 4,
    cmap: Optional[Cmap] = None,
    scaling: str = 'arctan',
    scaling_params: Optional[dict] = None,
    project_from_north: bool = True,
    interactive: bool = True,
    show_grid: bool = True,
    show_axes: bool = True,
    colorbar: bool = True,
    **kwargs
):
    """
    Plot a complex function on the Riemann sphere using PyVista.
    
    NOTE: This function requires the mesh_utils module with icosahedral
    sphere generation, which will be implemented in Phase 2.
    
    Parameters
    ----------
    func : Callable
        Complex function to visualize.
    n_subdivisions : int, default=4
        Number of icosphere subdivisions (controls resolution).
    cmap : Cmap, optional
        Color map to use. Default is Phase(6, 0.6).
    scaling : str, default='arctan'
        Modulus scaling method: 'arctan', 'logarithmic', 'linear_clamp', 'custom'.
    scaling_params : dict, optional
        Parameters for the scaling function.
    project_from_north : bool, default=True
        If True, use stereographic projection from north pole.
    interactive : bool, default=True
        If True, show interactive widget.
    show_grid : bool, default=True
        If True, show latitude/longitude grid lines.
    show_axes : bool, default=True
        If True, show coordinate axes.
    colorbar : bool, default=True
        If True, show color scale bar.
    **kwargs
        Additional arguments for visualization.
    
    Raises
    ------
    NotImplementedError
        This function requires icosahedral meshing (Phase 2).
    """
    raise NotImplementedError(
        "riemann_pv requires icosahedral sphere meshing. "
        "This will be implemented in Phase 2 of the PyVista integration. "
        "See docs/icosphere_technical_spec.md for details."
    )