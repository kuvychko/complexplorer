"""3D plotting functions using PyVista.

This module provides high-performance, interactive 3D visualizations
using PyVista as an alternative to matplotlib-based plots.
"""

from typing import Optional, Callable, Union, Tuple
import numpy as np
import warnings

from complexplorer.core.domain import Domain
from complexplorer.core.colormap import Colormap, Phase
from complexplorer.core.scaling import ModulusScaling
from complexplorer.utils.validation import ValidationError
from complexplorer.utils.mesh_distortion import get_default_scaling_params
from complexplorer.plotting.validation import (
    validate_modulus_mode, validate_z_max
)
from complexplorer.plotting.pyvista.utils import (
    check_pyvista_available, handle_export, add_axes_widget,
    ensure_pyvista_setup, get_camera_position
)

# Import PyVista if available
try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    pv = None


def create_complex_surface(
    domain: Optional[Domain],
    func: Optional[Callable],
    z: Optional[np.ndarray] = None,
    f: Optional[np.ndarray] = None,
    resolution: int = 100,
    cmap: Optional[Colormap] = None,
    z_scale: float = 1.0,
    log_z: bool = False,
    z_max: Optional[float] = None,
    modulus_mode: str = 'none',
    modulus_params: Optional[dict] = None
) -> Tuple['pv.StructuredGrid', np.ndarray]:
    """Create a PyVista mesh for a complex function surface.
    
    Parameters
    ----------
    domain : Domain, optional
        Domain object.
    func : callable, optional
        Complex function.
    z : ndarray, optional
        Domain mesh.
    f : ndarray, optional
        Function values.
    resolution : int, optional
        Resolution.
    cmap : Colormap, optional
        Colormap for coloring.
    z_scale : float, optional
        Height scaling factor.
    log_z : bool, optional
        Use logarithmic height.
    z_max : float, optional
        Maximum height value.
    modulus_mode : str, optional
        How to scale the height based on modulus.
    modulus_params : dict, optional
        Parameters for modulus scaling method.
        
    Returns
    -------
    grid : pv.StructuredGrid
        PyVista mesh.
    rgb_colors : ndarray
        RGB color array.
    """
    # Validate inputs
    if domain is None and z is None:
        raise ValidationError('Either domain or z must be provided')
    if f is None and func is None:
        raise ValidationError('Either f or func must be provided')
    
    # Get domain mesh
    if z is None:
        z = domain.mesh(resolution)
        mask = domain.outmask(resolution)
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
    
    # Apply modulus scaling if requested
    if modulus_mode != 'none':
        if modulus_params is None:
            modulus_params = get_default_scaling_params(modulus_mode)

        # Validate modulus mode and params
        validate_modulus_mode(modulus_mode, modulus_params)

        if modulus_mode == 'custom':
            magnitude = modulus_params['scaling_func'](magnitude)
        else:
            scaling_method = getattr(ModulusScaling, modulus_mode)
            magnitude = scaling_method(magnitude, **modulus_params)

    # Apply z_max clipping after scaling
    validate_z_max(z_max)
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
        cmap = Phase(phase_sectors=6, v_base=0.6)
    
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
    resolution: int = 100,
    cmap: Optional[Colormap] = None,
    interactive: bool = True,
    notebook: Optional[bool] = None,
    camera_position: Union[str, Tuple] = 'iso',
    show_edges: bool = False,
    edge_color: str = 'gray',
    z_scale: float = 1.0,
    log_z: bool = False,
    z_max: Optional[float] = None,
    modulus_mode: str = 'none',
    modulus_params: Optional[dict] = None,
    window_size: Tuple[int, int] = (800, 600),
    title: Optional[str] = None,
    filename: Optional[str] = None,
    return_plotter: bool = False,
    show_orientation: bool = True,
    **kwargs
) -> Optional['pv.Plotter']:
    """Plot complex function as 3D landscape using PyVista.
    
    This function provides high-performance, interactive 3D visualization
    with accurate per-vertex coloring (no interpolation artifacts).
    
    Parameters
    ----------
    domain : Domain, optional
        Domain object. If None, z must be provided.
    func : callable, optional
        Complex function. If None, f must be provided.
    z : ndarray, optional
        2D array of complex domain values.
    f : ndarray, optional
        2D array of complex codomain values.
    resolution : int, optional
        Resolution (number of points along longest edge).
    cmap : Colormap, optional
        Colormap to use. Defaults to enhanced phase portrait.
    interactive : bool, optional
        If True, show interactive widget. If False, render static.
    notebook : bool, optional
        If True, render inline in Jupyter. If None, auto-detect.
    camera_position : str or tuple, optional
        Camera position: 'iso', 'xy', 'xz', 'yz', or custom.
    show_edges : bool, optional
        If True, show mesh edges.
    edge_color : str, optional
        Color of mesh edges.
    z_scale : float, optional
        Scaling factor for height.
    log_z : bool, optional
        If True, use logarithmic scaling for height.
    z_max : float, optional
        Maximum value for height clipping.
    modulus_mode : str, optional
        How to scale the height based on modulus.
        See plot_landscape for available modes.
    modulus_params : dict, optional
        Parameters for modulus scaling method.
    window_size : tuple, optional
        Window size in pixels.
    title : str, optional
        Title for the plot.
    filename : str, optional
        Save plot to file. Supported formats:
        - Static images: .png, .jpg, .jpeg
        - Vector graphics: .pdf, .svg, .eps
        - Interactive HTML: .html (requires trame)
    return_plotter : bool, optional
        If True, return the plotter object.
    show_orientation : bool, optional
        If True, show orientation widget.
    **kwargs
        Additional arguments passed to pv.Plotter.
        
    Returns
    -------
    pv.Plotter or None
        The plotter object if return_plotter=True.
        
    Examples
    --------
    >>> # Interactive visualization
    >>> domain = Rectangle(4, 4)
    >>> plot_landscape_pv(domain, lambda z: z**2, resolution=150)
    
    >>> # Save static image
    >>> plot_landscape_pv(domain, lambda z: 1/z, 
    ...                   interactive=False, filename='poles.png')
    """
    check_pyvista_available()
    ensure_pyvista_setup()
    
    # Create surface mesh
    grid, rgb = create_complex_surface(
        domain, func, z, f, resolution, cmap, z_scale, log_z, z_max,
        modulus_mode, modulus_params
    )
    
    # Create plotter
    plotter_kwargs = {
        'window_size': window_size,
        'off_screen': not interactive,
    }
    if notebook is not None:
        plotter_kwargs['notebook'] = notebook
    
    # Add any user-provided kwargs, filtering out our function parameters
    # that might have been accidentally passed as kwargs
    filtered_kwargs = {k: v for k, v in kwargs.items() 
                      if k not in {'resolution', 'n', 'domain', 'func', 'z', 'f',
                                   'cmap', 'interactive', 'camera_position', 
                                   'show_edges', 'edge_color', 'z_scale', 'log_z',
                                   'z_max', 'modulus_mode', 'modulus_params',
                                   'title', 'filename', 'return_plotter',
                                   'show_orientation', 'show'}}
    plotter_kwargs.update(filtered_kwargs)
    
    plotter = pv.Plotter(**plotter_kwargs)
    
    # Add the surface
    actor = plotter.add_mesh(
        grid,
        scalars="RGB",
        rgb=True,
        show_edges=show_edges,
        edge_color=edge_color,
        smooth_shading=True,
        specular=0.5,
        specular_power=15,
        diffuse=0.7,
        ambient=0.3,
    )
    
    # Set camera
    plotter.camera_position = get_camera_position(camera_position)
    
    # Add title
    if title:
        plotter.add_text(title, position='upper_edge', font_size=14)
    
    # Add orientation widget
    if show_orientation:
        add_axes_widget(plotter, labels=('Re', 'Im', '|f|'))
    
    # Handle export
    if filename:
        if interactive:
            # For interactive mode, we'll export after showing
            plotter.show()
            handle_export(plotter, filename, interactive)
        else:
            # For static mode, export directly
            handle_export(plotter, filename, interactive)
    elif interactive:
        plotter.show()
    
    if return_plotter:
        return plotter


def pair_plot_landscape_pv(
    domain: Optional[Domain] = None,
    func: Optional[Callable] = None,
    z: Optional[np.ndarray] = None,
    f: Optional[np.ndarray] = None,
    resolution: int = 100,
    cmap: Optional[Colormap] = None,
    interactive: bool = True,
    notebook: Optional[bool] = None,
    camera_position: Union[str, Tuple] = 'iso',
    z_scale: float = 1.0,
    log_z: bool = False,
    z_max: Optional[float] = None,
    modulus_mode: str = 'none',
    modulus_params: Optional[dict] = None,
    window_size: Tuple[int, int] = (1200, 600),
    title: Optional[str] = None,
    filename: Optional[str] = None,
    return_plotter: bool = False,
    **kwargs
) -> Optional['pv.Plotter']:
    """Plot domain and codomain landscapes side-by-side using PyVista.
    
    Parameters
    ----------
    domain : Domain, optional
        Domain object. If None, z must be provided.
    func : callable, optional
        Complex function. If None, f must be provided.
    z : ndarray, optional
        2D array of complex domain values.
    f : ndarray, optional
        2D array of complex codomain values.
    resolution : int, optional
        Resolution.
    cmap : Colormap, optional
        Colormap to use.
    interactive : bool, optional
        If True, show interactive widget.
    notebook : bool, optional
        If True, render inline in Jupyter.
    camera_position : str or tuple, optional
        Camera position for both views.
    z_scale : float, optional
        Scaling factor for height.
    log_z : bool, optional
        Use logarithmic scaling.
    z_max : float, optional
        Maximum height value.
    modulus_mode : str, optional
        How to scale the height based on modulus.
    modulus_params : dict, optional
        Parameters for modulus scaling method.
    window_size : tuple, optional
        Window size in pixels.
    title : str, optional
        Overall title.
    filename : str, optional
        Save plot to file.
    return_plotter : bool, optional
        If True, return the plotter object.
        
    Returns
    -------
    pv.Plotter or None
        The plotter object if return_plotter=True.
    """
    check_pyvista_available()
    ensure_pyvista_setup()
    
    # Create plotter with two viewports
    plotter_kwargs = {
        'window_size': window_size,
        'off_screen': not interactive,
        'shape': (1, 2),
    }
    if notebook is not None:
        plotter_kwargs['notebook'] = notebook
    
    # Filter kwargs to avoid passing our function parameters to PyVista
    filtered_kwargs = {k: v for k, v in kwargs.items() 
                      if k not in {'resolution', 'n', 'domain', 'func', 'z', 'f',
                                   'cmap', 'interactive', 'camera_position', 
                                   'show_edges', 'edge_color', 'z_scale', 'log_z',
                                   'z_max', 'modulus_mode', 'modulus_params',
                                   'title', 'filename', 'return_plotter',
                                   'show_orientation', 'show'}}
    plotter_kwargs.update(filtered_kwargs)
    
    plotter = pv.Plotter(**plotter_kwargs)
    
    # Left subplot: Domain (identity function)
    plotter.subplot(0, 0)
    grid_domain, _ = create_complex_surface(
        domain, lambda x: x, z, z, resolution, cmap, z_scale, log_z, z_max,
        modulus_mode, modulus_params
    )
    plotter.add_mesh(
        grid_domain,
        scalars="RGB",
        rgb=True,
        smooth_shading=True,
        specular=0.5,
        specular_power=15,
    )
    plotter.add_text("Domain z", position='upper_edge')
    add_axes_widget(plotter, labels=('Re', 'Im', '|z|'))
    plotter.camera_position = get_camera_position(camera_position)
    
    # Right subplot: Codomain
    plotter.subplot(0, 1)
    grid_codomain, _ = create_complex_surface(
        domain, func, z, f, resolution, cmap, z_scale, log_z, z_max,
        modulus_mode, modulus_params
    )
    plotter.add_mesh(
        grid_codomain,
        scalars="RGB",
        rgb=True,
        smooth_shading=True,
        specular=0.5,
        specular_power=15,
    )
    # Use title as codomain label if provided, otherwise default
    codomain_label = title if title else "Codomain f(z)"
    plotter.add_text(codomain_label, position='upper_edge')
    add_axes_widget(plotter, labels=('Re', 'Im', '|f|'))
    plotter.camera_position = get_camera_position(camera_position)
    
    # Link cameras for synchronized interaction
    if interactive:
        plotter.link_views()
    
    # Handle export/display
    if filename:
        if interactive:
            plotter.show()
            handle_export(plotter, filename, interactive)
        else:
            handle_export(plotter, filename, interactive)
    elif interactive:
        plotter.show()
    
    if return_plotter:
        return plotter