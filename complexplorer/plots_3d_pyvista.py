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


def _add_axes_widget(plotter, labels=('Re', 'Im', 'Z'), position=(0.0, 0.0), 
                     size=0.25, label_size=(0.25, 0.1)):
    """
    Add a labeled axes widget to the plotter.
    
    Parameters
    ----------
    plotter : pv.Plotter
        The PyVista plotter object.
    labels : tuple of str, default=('Re', 'Im', 'Z')
        Labels for x, y, z axes.
    position : tuple of float, default=(0.0, 0.0)
        Position of widget in viewport coordinates (0-1).
    size : float, default=0.25
        Size of the widget as fraction of viewport.
    label_size : tuple of float, default=(0.25, 0.1)
        Width and height of the axes label actors (values between 0 and 1).
    """
    # Check if we're in off-screen mode (static rendering)
    is_static = plotter.off_screen
    
    if is_static:
        # For static rendering, use viewport-based approach
        axes_actor = plotter.add_axes(
            xlabel=labels[0],
            ylabel=labels[1],
            zlabel=labels[2],
            line_width=4,
            labels_off=False,
            box=False,
            interactive=True,
            viewport=(0, 0, size, size),
            label_size=label_size,
            cone_radius=0.4,
            shaft_length=0.8,
            tip_length=0.2,
            ambient=0.5,
            color='black',
        )
    else:
        # For interactive mode, use the standard add_axes without viewport
        # This seems to work better in Jupyter notebooks
        axes_actor = plotter.add_axes(
            xlabel=labels[0],
            ylabel=labels[1],
            zlabel=labels[2],
            line_width=4,
            labels_off=False,
            box=False,
            interactive=True,
            label_size=label_size,
            cone_radius=0.4,
            shaft_length=0.8,
            tip_length=0.2,
            ambient=0.5,
            color='black',
        )
    
    return axes_actor


def _ensure_pyvista_setup():
    """Ensure PyVista is properly configured for the current environment."""
    # Set conservative defaults for better compatibility
    # Users can increase these if their system supports it
    if pv.global_theme.multi_samples is None:
        pv.global_theme.multi_samples = 2  # Conservative default
    pv.global_theme.smooth_shading = True
    
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
    show_orientation: bool = True,
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
    show_orientation : bool, default=True
        If True, show orientation axes widget with Re/Im/Z labels.
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
    
    # Add orientation axes
    if show_orientation:
        _add_axes_widget(plotter, labels=('Re', 'Im', 'Z'))
    
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
    show_orientation: bool = True,
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
    show_orientation : bool, default=True
        If True, show orientation axes widget with Re/Im/Z labels.
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
    if show_orientation:
        _add_axes_widget(plotter, labels=('Re', 'Im', 'Z'))
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
    if show_orientation:
        _add_axes_widget(plotter, labels=('Re', 'Im', 'Z'))
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


def riemann_pv(
    func: Callable,
    n_theta: int = 100,
    n_phi: int = 100,
    mesh_type: str = 'rectangular',
    n_subdivisions: int = 4,
    cmap: Optional[Cmap] = None,
    scaling: str = 'constant',
    scaling_params: Optional[dict] = None,
    domain: Optional['Domain'] = None,
    project_from_north: bool = True,
    interactive: bool = True,
    camera_position: Union[str, Tuple] = 'iso',
    show_grid: bool = True,
    show_orientation: bool = True,
    window_size: Tuple[int, int] = (800, 800),
    title: Optional[str] = None,
    filename: Optional[str] = None,
    return_plotter: bool = False,
    high_quality: bool = True,
    anti_aliasing: Union[bool, str] = True,
    aa_samples: int = 4,
    use_pbr: bool = False,
    **kwargs
):
    """
    Plot a complex function on the Riemann sphere using PyVista.
    
    By default, this function uses rectangular (latitude-longitude) meshing which
    provides better visual quality in PyVista despite slight pole distortion.
    Icosahedral meshing is available as an alternative for uniform sampling.
    
    Parameters
    ----------
    func : Callable
        Complex function to visualize.
    n_theta : int, default=100
        Number of latitude divisions (for rectangular mesh).
    n_phi : int, default=100
        Number of longitude divisions (for rectangular mesh).
    mesh_type : str, default='rectangular'
        Type of sphere meshing:
        - 'rectangular': Latitude-longitude grid (better PyVista rendering)
        - 'icosahedral': Uniform triangular mesh (better mathematical properties)
        - 'uv': PyVista's built-in UV sphere
    n_subdivisions : int, default=4
        For icosahedral mesh: subdivision level (0-8).
        Level 0: 20 faces, Level n: 20 * 4^n faces.
    cmap : Cmap, optional
        Color map to use. Default is Phase(6, 0.6).
    scaling : str, default='constant'
        Modulus scaling method:
        - 'constant': Traditional Riemann sphere (radius = 1)
        - 'arctan': Smooth compression mapping [0, ∞) to [r_min, r_max]
        - 'logarithmic': Log scaling for large dynamic range
        - 'linear_clamp': Linear with saturation
        - 'custom': User-defined function
    scaling_params : dict, optional
        Parameters for the scaling function:
        - For 'constant': {'radius': 1.0}
        - For 'arctan': {'r_min': 0.2, 'r_max': 1.0}
        - For 'logarithmic': {'base': e, 'r_min': 0.2, 'r_max': 1.0}
        - For 'linear_clamp': {'m_max': 10, 'r_min': 0.2, 'r_max': 1.0}
        - For 'custom': {'scaling_func': callable, 'r_min': 0.2, 'r_max': 1.0}
    project_from_north : bool, default=True
        If True, use stereographic projection from north pole.
    interactive : bool, default=True
        If True, show interactive widget.
    camera_position : str or tuple, default='iso'
        Camera position: 'iso', 'xy', 'xz', 'yz', or custom.
    show_grid : bool, default=True
        If True, show latitude/longitude grid lines.
    show_orientation : bool, default=True
        If True, show orientation axes widget with Re/Im/Z labels.
    window_size : tuple, default=(800, 800)
        Window size in pixels.
    title : str, optional
        Title for the plot.
    filename : str, optional
        Save plot to file.
    return_plotter : bool, default=False
        If True, return the plotter object.
    high_quality : bool, default=True
        If True, enable high-quality rendering with enhanced shading.
    anti_aliasing : bool or str, default=True
        If True, automatically select best available anti-aliasing.
        If string, use specific method: 'msaa', 'ssaa', 'fxaa', or 'none'.
    aa_samples : int, default=4
        Number of samples for MSAA/SSAA (2, 4, or 8).
    use_pbr : bool, default=False
        If True, use physically based rendering (requires compatible GPU/drivers).
        May cause shader errors on some systems.
    **kwargs
        Additional arguments passed to plotter.add_mesh().
    
    Returns
    -------
    plotter : pv.Plotter, optional
        Only returned if return_plotter=True.
    
    Examples
    --------
    >>> import complexplorer as cp
    >>> # Traditional Riemann sphere with rectangular mesh
    >>> cp.riemann_pv(lambda z: (z-1)/(z+1))
    >>> # With modulus scaling
    >>> cp.riemann_pv(lambda z: z**2, scaling='arctan')
    >>> # Using icosahedral mesh
    >>> cp.riemann_pv(lambda z: z**2, mesh_type='icosahedral', n_subdivisions=4)
    
    Notes
    -----
    PyVista renders rectangular meshes more smoothly than triangular meshes,
    so the default rectangular mesh often looks better despite mathematical
    imperfections near poles.
    """
    from complexplorer.mesh_utils import (
        IcosphereGenerator, RectangularSphereGenerator, 
        stereographic_projection, ModulusScaling
    )
    
    _ensure_pyvista_setup()
    
    # Default colormap
    if cmap is None:
        cmap = Phase(6, 0.6)
    
    # Generate sphere mesh based on type
    if mesh_type == 'rectangular':
        generator = RectangularSphereGenerator(radius=1.0, n_theta=n_theta, n_phi=n_phi)
        sphere = generator.generate()
    elif mesh_type == 'uv':
        generator = RectangularSphereGenerator(radius=1.0, n_theta=n_theta, n_phi=n_phi)
        sphere = generator.generate_uv_sphere()
    elif mesh_type == 'icosahedral':
        generator = IcosphereGenerator(radius=1.0, subdivisions=n_subdivisions)
        sphere = generator.generate()
    else:
        raise ValueError(f"Unknown mesh_type: {mesh_type}. Use 'rectangular', 'uv', or 'icosahedral'")
    
    # Get sphere points
    points = sphere.points
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    # Stereographic projection to complex plane
    w = stereographic_projection(x, y, z, from_north=project_from_north)
    
    # Evaluate function
    f_vals = func(w)
    
    # Handle infinities and NaN
    finite_mask = np.isfinite(f_vals)
    f_vals[~finite_mask] = 0  # Temporary value for color calculation
    
    # Get colors from colormap
    rgb = cmap.rgb(f_vals.reshape(-1, 1)).squeeze()
    sphere["RGB"] = rgb
    
    # Apply modulus scaling
    moduli = np.abs(f_vals)
    
    # Set up scaling parameters with defaults
    if scaling_params is None:
        scaling_params = {}
    
    # Apply scaling based on method
    if scaling == 'constant':
        radius = scaling_params.get('radius', 1.0)
        radii = ModulusScaling.constant(moduli, radius)
    elif scaling == 'arctan':
        r_min = scaling_params.get('r_min', 0.2)
        r_max = scaling_params.get('r_max', 1.0)
        radii = ModulusScaling.arctan(moduli, r_min, r_max)
    elif scaling == 'logarithmic':
        base = scaling_params.get('base', np.e)
        r_min = scaling_params.get('r_min', 0.2)
        r_max = scaling_params.get('r_max', 1.0)
        radii = ModulusScaling.logarithmic(moduli, base, r_min, r_max)
    elif scaling == 'linear_clamp':
        m_max = scaling_params.get('m_max', 10)
        r_min = scaling_params.get('r_min', 0.2)
        r_max = scaling_params.get('r_max', 1.0)
        radii = ModulusScaling.linear_clamp(moduli, m_max, r_min, r_max)
    elif scaling == 'custom':
        scaling_func = scaling_params.get('scaling_func', lambda x: x)
        r_min = scaling_params.get('r_min', 0.2)
        r_max = scaling_params.get('r_max', 1.0)
        radii = ModulusScaling.custom(moduli, scaling_func, r_min, r_max)
    else:
        raise ValueError(f"Unknown scaling method: {scaling}")
    
    # Handle infinities in radii by interpolating from neighbors
    if not np.all(finite_mask):
        from scipy.ndimage import generic_filter
        
        if mesh_type == 'rectangular':
            # For structured grids, use 2D neighbor interpolation
            radii_2d = radii.reshape((n_phi, n_theta))
            finite_mask_2d = finite_mask.reshape((n_phi, n_theta))
            
            def neighbor_mean(values):
                """Average of finite neighbors."""
                finite_vals = values[np.isfinite(values)]
                return np.mean(finite_vals) if len(finite_vals) > 0 else np.nan
            
            # Apply filter iteratively until all NaN/inf are filled
            filled_radii = radii_2d.copy()
            max_iterations = 10
            for _ in range(max_iterations):
                old_filled = filled_radii.copy()
                # Use wrap mode to handle periodic boundary in phi direction
                filled_radii = generic_filter(filled_radii, neighbor_mean, size=3, mode='wrap')
                # Only update the invalid points
                filled_radii[finite_mask_2d] = radii_2d[finite_mask_2d]
                # Check if we've filled all invalid points
                if np.all(np.isfinite(filled_radii)):
                    break
                # Check if no progress was made
                if np.array_equal(old_filled, filled_radii):
                    # Fill remaining with median of valid values
                    filled_radii[~np.isfinite(filled_radii)] = np.median(radii[finite_mask])
                    break
            
            radii = filled_radii.ravel()
        else:
            # For unstructured meshes, use median of valid values
            valid_radii = radii[finite_mask]
            if len(valid_radii) > 0:
                radii[~finite_mask] = np.median(valid_radii)
            else:
                radii[~finite_mask] = 1.0
    
    # Scale sphere points
    if mesh_type == 'rectangular':
        # For structured grids, we need to reshape and recreate
        n_points = n_phi * n_theta
        radii_reshaped = radii.reshape((n_phi, n_theta))
        
        # Get original grid shape
        theta = np.linspace(0.01, np.pi - 0.01, n_theta)
        phi = np.linspace(0, 2 * np.pi, n_phi)
        THETA, PHI = np.meshgrid(theta, phi)
        
        # Apply radial scaling
        X = radii_reshaped * np.sin(THETA) * np.cos(PHI)
        Y = radii_reshaped * np.sin(THETA) * np.sin(PHI)
        Z = radii_reshaped * np.cos(THETA)
        
        # Create new scaled grid
        sphere = pv.StructuredGrid(X, Y, Z)
        sphere["RGB"] = rgb
    else:
        # For unstructured meshes, simple point scaling works
        scaled_points = points * radii[:, np.newaxis]
        sphere.points = scaled_points
    
    # Store additional data
    sphere["magnitude"] = moduli
    sphere["phase"] = np.angle(f_vals)
    
    # Create plotter
    plotter = pv.Plotter(
        off_screen=not interactive,
        window_size=window_size
    )
    
    # Enable anti-aliasing
    if anti_aliasing and anti_aliasing != 'none':
        aa_applied = False
        
        if isinstance(anti_aliasing, str):
            # User specified a specific method
            if anti_aliasing in ['msaa', 'ssaa']:
                try:
                    plotter.enable_anti_aliasing(anti_aliasing, multi_samples=aa_samples)
                    aa_applied = True
                except:
                    pass
            elif anti_aliasing == 'fxaa':
                try:
                    plotter.enable_anti_aliasing('fxaa')
                    aa_applied = True
                except:
                    pass
        else:
            # Auto-select best available method
            # Try methods in order of compatibility
            aa_methods = [
                ('msaa', {'multi_samples': aa_samples}),    # Most compatible
                ('fxaa', {}),                               # Fast fallback
                ('ssaa', {'multi_samples': aa_samples}),    # High quality but demanding
            ]
            
            for method, params in aa_methods:
                try:
                    plotter.enable_anti_aliasing(method, **params)
                    aa_applied = True
                    break
                except:
                    continue
        
        if not aa_applied:
            # Last resort: try setting multi_samples on the render window
            try:
                plotter.ren_win.SetMultiSamples(aa_samples)
            except:
                # If all else fails, continue without anti-aliasing
                pass
    
    # Configure mesh rendering parameters
    mesh_kwargs = {
        'rgb': True,
        'smooth_shading': True,
        'specular': 0.5,
        'specular_power': 20,
    }
    
    # Add high-quality rendering options
    if high_quality:
        if use_pbr:
            # PBR settings (may cause shader errors on some systems)
            mesh_kwargs.update({
                'pbr': True,
                'metallic': 0.2,
                'roughness': 0.5,
                'interpolate_before_map': True,
            })
        else:
            # Enhanced settings without PBR (more compatible)
            mesh_kwargs.update({
                'smooth_shading': True,
                'specular': 0.8,
                'specular_power': 50,
                'ambient': 0.3,
                'diffuse': 0.7,
            })
    
    mesh_kwargs.update(kwargs)
    
    actor = plotter.add_mesh(sphere, **mesh_kwargs)
    
    # Add grid lines if requested
    if show_grid:
        # Create latitude/longitude lines
        n_lat = 9  # Number of latitude lines
        n_lon = 12  # Number of longitude lines
        
        # Latitude lines
        for i in range(1, n_lat):
            lat = np.pi * i / n_lat - np.pi / 2
            theta = np.linspace(0, 2 * np.pi, 100)
            x_lat = np.cos(lat) * np.cos(theta)
            y_lat = np.cos(lat) * np.sin(theta)
            z_lat = np.full_like(theta, np.sin(lat))
            points_lat = np.column_stack([x_lat, y_lat, z_lat])
            
            if scaling == 'constant':
                points_lat *= scaling_params.get('radius', 1.0)
            
            line = pv.PolyData(points_lat)
            plotter.add_mesh(line, color='gray', line_width=0.5, opacity=0.3)
        
        # Longitude lines
        for i in range(n_lon):
            lon = 2 * np.pi * i / n_lon
            phi = np.linspace(-np.pi / 2, np.pi / 2, 100)
            x_lon = np.cos(phi) * np.cos(lon)
            y_lon = np.cos(phi) * np.sin(lon)
            z_lon = np.sin(phi)
            points_lon = np.column_stack([x_lon, y_lon, z_lon])
            
            if scaling == 'constant':
                points_lon *= scaling_params.get('radius', 1.0)
            
            line = pv.PolyData(points_lon)
            plotter.add_mesh(line, color='gray', line_width=0.5, opacity=0.3)
    
    # Set camera
    plotter.camera_position = camera_position
    
    # Add orientation axes
    if show_orientation:
        _add_axes_widget(plotter, labels=('Re', 'Im', 'Z'))
    
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