"""Riemann sphere visualization using PyVista.

This module provides high-performance visualization of complex functions
on the Riemann sphere with various modulus scaling options.
"""

from typing import Optional, Callable, Union, Tuple, TYPE_CHECKING
import numpy as np
import warnings

from complexplorer.core.colormap import Colormap, Phase
from complexplorer.core.scaling import ModulusScaling
from complexplorer.utils.validation import ValidationError
from complexplorer.utils.mesh import RectangularSphereGenerator
from complexplorer.utils.mesh_distortion import apply_modulus_distortion, get_default_scaling_params
from complexplorer.plotting.pyvista.utils import (
    check_pyvista_available, handle_export, add_axes_widget,
    ensure_pyvista_setup, get_camera_position
)

if TYPE_CHECKING:
    from complexplorer.core.domain import Domain

# Import PyVista if available
try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    pv = None


def riemann_pv(
    func: Callable,
    resolution: int = 100,
    cmap: Optional[Colormap] = None,
    domain: Optional['Domain'] = None,
    interactive: bool = True,
    notebook: Optional[bool] = None,
    camera_position: Union[str, Tuple] = (2.5, 2.5, 2.5),
    window_size: Tuple[int, int] = (800, 800),
    title: Optional[str] = None,
    filename: Optional[str] = None,
    show_orientation: bool = True,
    show_grid: bool = False,
    modulus_mode: str = 'constant',
    modulus_params: Optional[dict] = None,
    return_plotter: bool = False,
    **kwargs
) -> Optional['pv.Plotter']:
    """Plot complex function on the Riemann sphere using PyVista.
    
    This function provides high-performance, interactive visualization
    of complex functions on the Riemann sphere with various options
    for incorporating magnitude information.
    
    Uses stereographic projection from the north pole (standard convention):
    - Origin (z=0) maps to south pole
    - Infinity maps to north pole
    - Unit circle maps to equator
    
    Parameters
    ----------
    func : callable
        Complex function to visualize.
    resolution : int, optional
        Resolution (number of divisions in each direction).
    cmap : Colormap, optional
        Colormap for coloring. Defaults to Phase(6, 0.6).
    domain : Domain, optional
        If provided, only show sphere points mapping to this domain.
    interactive : bool, optional
        If True, show interactive widget.
    notebook : bool, optional
        If True, render inline in Jupyter.
    camera_position : str or tuple, optional
        Camera position.
    window_size : tuple, optional
        Window size in pixels.
    title : str, optional
        Title for the plot.
    filename : str, optional
        Save plot to file.
    show_orientation : bool, optional
        If True, show orientation axes.
    show_grid : bool, optional
        If True, show latitude/longitude grid.
    modulus_mode : str, optional
        How to incorporate magnitude:
        - 'constant': Unit sphere (phase only)
        - 'linear': Linear scaling
        - 'arctan': Smooth bounded scaling
        - 'logarithmic': Log scaling
        - 'linear_clamp': Linear with clamping
        - 'power': Power scaling
        - 'sigmoid': S-curve scaling
        - 'adaptive': Percentile-based
        - 'hybrid': Linear near zero, log for large
        - 'custom': User-defined function
    modulus_params : dict, optional
        Parameters for modulus scaling method.
    return_plotter : bool, optional
        If True, return the plotter object.
    **kwargs
        Additional arguments passed to pv.Plotter.
        
    Returns
    -------
    pv.Plotter or None
        The plotter object if return_plotter=True.
        
    Examples
    --------
    >>> # Basic visualization
    >>> riemann_pv(lambda z: (z-1)/(z+1))
    
    >>> # With magnitude scaling
    >>> riemann_pv(lambda z: z**2, modulus_mode='arctan')
    
    >>> # Custom scaling function
    >>> def custom_scale(moduli):
    ...     return np.tanh(moduli / 2)
    >>> riemann_pv(lambda z: np.sin(z), modulus_mode='custom',
    ...           modulus_params={'scaling_func': custom_scale})
    """
    check_pyvista_available()
    ensure_pyvista_setup()
    
    # Default colormap
    if cmap is None:
        cmap = Phase(phase_sectors=6, v_base=0.6)
    
    # Default modulus parameters
    if modulus_params is None:
        modulus_params = get_default_scaling_params(modulus_mode, for_stl=False)
    
    # Create sphere mesh generator
    gen = RectangularSphereGenerator(
        n_theta=resolution,
        n_phi=resolution,
        avoid_poles=True,
        domain=domain
    )
    
    # Generate base sphere
    mesh = gen.generate()
    
    # Get sphere points and convert to complex plane
    points = mesh.points
    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]
    
    # Project to complex plane (from north pole - standard convention)
    from complexplorer.utils.mesh import sphere_to_complex
    w = sphere_to_complex(X, Y, Z, from_north=True)
    
    # Evaluate function
    f_vals = func(w)
    f_vals = np.asarray(f_vals)
    
    # Get colors
    rgb = cmap.rgb(f_vals)
    mesh["RGB"] = rgb
    
    # Apply modulus scaling
    moduli = np.abs(f_vals)
    
    # Use shared distortion function
    scaled_points, radii = apply_modulus_distortion(
        points, moduli, modulus_mode, modulus_params
    )
    mesh.points = scaled_points
    
    # Store original points for grid generation if needed
    original_points = points.copy()
    
    # Store additional scalars
    mesh["magnitude"] = np.abs(f_vals)
    mesh["phase"] = np.angle(f_vals)
    
    # Create plotter
    plotter_kwargs = {
        'window_size': window_size,
        'off_screen': not interactive,
    }
    if notebook is not None:
        plotter_kwargs['notebook'] = notebook
    
    # Filter kwargs to avoid passing our function parameters to PyVista
    filtered_kwargs = {k: v for k, v in kwargs.items() 
                      if k not in {'func', 'cmap', 'modulus_mode', 'modulus_params',
                                   'resolution', 'n', 'domain', 'interactive',
                                   'camera_position', 'radius', 'title',
                                   'filename', 'return_plotter', 'show_orientation',
                                   'show'}}
    plotter_kwargs.update(filtered_kwargs)
    
    plotter = pv.Plotter(**plotter_kwargs)
    
    # Add the sphere
    actor = plotter.add_mesh(
        mesh,
        scalars="RGB",
        rgb=True,
        smooth_shading=True,
        specular=0.5,
        specular_power=20,
        diffuse=0.8,
        ambient=0.2,
        show_edges=False,  # Never show triangulation edges
    )
    
    # Add latitude/longitude grid if requested
    if show_grid:
        add_lat_long_grid(plotter, radius=1.0, n_lat=10, n_long=12, 
                         modulus_mode=modulus_mode, modulus_params=modulus_params,
                         scaled_points=scaled_points, points=original_points)
    
    # Set camera
    plotter.camera_position = get_camera_position(camera_position)
    
    # Add title
    if title:
        plotter.add_text(title, position='upper_edge', font_size=14)
    
    # Add orientation widget with complex plane labels
    if show_orientation:
        add_axes_widget(plotter, labels=('Re', 'Im', 'z'))
    
    # Set background
    plotter.set_background('white')
    
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


def add_lat_long_grid(plotter: 'pv.Plotter', 
                     radius: float = 1.0,
                     n_lat: int = 10,
                     n_long: int = 12,
                     modulus_mode: str = 'constant',
                     modulus_params: Optional[dict] = None,
                     scaled_points: Optional[np.ndarray] = None,
                     points: Optional[np.ndarray] = None,
                     color: str = 'black',
                     line_width: float = 1.0,
                     opacity: float = 0.5) -> None:
    """Add latitude and longitude grid lines to the Riemann sphere.
    
    Parameters
    ----------
    plotter : pv.Plotter
        The plotter to add grid lines to.
    radius : float
        Base radius of sphere.
    n_lat : int
        Number of latitude lines.
    n_long : int  
        Number of longitude lines.
    modulus_mode : str
        Modulus scaling mode.
    modulus_params : dict, optional
        Parameters for modulus scaling.
    scaled_points : ndarray, optional
        Pre-scaled sphere points for modulus distortion.
    points : ndarray, optional
        Original sphere points.
    color : str
        Color of grid lines.
    line_width : float
        Width of grid lines.
    opacity : float
        Opacity of grid lines.
    """
    # For simplicity, create grid on unit sphere if modulus is constant
    # For non-constant modulus, we'd need more sophisticated interpolation
    
    # Create latitude lines (circles at constant theta)
    for i in range(1, n_lat):
        theta = i * np.pi / n_lat
        phi = np.linspace(0, 2 * np.pi, 100)
        
        # Points on latitude circle
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.full_like(phi, np.cos(theta))
        
        # Create polyline
        lat_points = np.column_stack([x, y, z])
        polyline = pv.PolyData()
        polyline.points = lat_points
        cells = np.column_stack([
            np.full(len(lat_points) - 1, 2, dtype=int),
            np.arange(len(lat_points) - 1),
            np.arange(1, len(lat_points))
        ])
        polyline.lines = cells.ravel()
        
        plotter.add_mesh(polyline, color=color, line_width=line_width, 
                        opacity=opacity, ambient=1.0, diffuse=0.0)
    
    # Create longitude lines (meridians at constant phi)
    for i in range(n_long):
        phi = i * 2 * np.pi / n_long
        theta = np.linspace(0.05, np.pi - 0.05, 50)  # Avoid poles
        
        # Points on longitude line
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)
        
        # Create polyline
        long_points = np.column_stack([x, y, z])
        polyline = pv.PolyData()
        polyline.points = long_points
        cells = np.column_stack([
            np.full(len(long_points) - 1, 2, dtype=int),
            np.arange(len(long_points) - 1),
            np.arange(1, len(long_points))
        ])
        polyline.lines = cells.ravel()
        
        plotter.add_mesh(polyline, color=color, line_width=line_width,
                        opacity=opacity, ambient=1.0, diffuse=0.0)
    
    # Add warning if modulus scaling is used
    if modulus_mode != 'constant':
        import warnings
        warnings.warn("Grid lines are shown on unit sphere; they don't follow modulus distortion", 
                     UserWarning)