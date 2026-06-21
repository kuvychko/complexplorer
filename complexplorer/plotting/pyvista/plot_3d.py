"""3D plotting functions using PyVista.

This module provides high-performance, interactive 3D visualizations
using PyVista as an alternative to matplotlib-based plots.
"""

from collections.abc import Callable
from typing import Optional

import numpy as np

from ...core.colormap import Colormap, Phase
from ...core.domain import Domain
from ...core.field import ComplexField
from ...mesh import build_landscape
from ...utils.validation import ValidationError
from .utils import (
    add_axes_widget,
    check_pyvista_available,
    ensure_pyvista_setup,
    get_camera_position,
    handle_export,
)

# Import PyVista if available
try:
    import pyvista as pv

    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    pv = None


def create_complex_surface(
    domain: Domain | None,
    func: Callable | None,
    z: np.ndarray | None = None,
    f: np.ndarray | None = None,
    resolution: int = 100,
    cmap: Colormap | None = None,
    z_scale: float = 1.0,
    log_z: bool = False,
    z_max: float | None = None,
    modulus_mode: str = "none",
    modulus_params: dict | None = None,
) -> tuple["pv.StructuredGrid", np.ndarray]:
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
        raise ValidationError("Either domain or z must be provided")
    if f is None and func is None:
        raise ValidationError("Either f or func must be provided")
    if cmap is None:
        cmap = Phase(n_phi=6, v_base=0.6)

    # Resolve the sampling grid + out-of-domain mask (mask only when derived from a domain).
    if z is None:
        z = domain.mesh(resolution)
        mask = domain.outmask(resolution)
    else:
        z = np.asarray(z)
        mask = None

    # Resolve the function values.
    if f is None:
        with np.errstate(all="ignore"):
            f = np.asarray(func(z))
    else:
        f = np.asarray(f)
    if f.ndim == 0:  # scalar case
        f = np.full_like(z, f)

    # Delegate geometry + decoration to the surface kernel (output-preserving).
    with np.errstate(all="ignore"):
        field = ComplexField("planar", w=f, modulus=np.abs(f), phase=np.angle(f), mask=mask, z=z)
    sm = build_landscape(
        field,
        cmap=cmap,
        z_scale=z_scale,
        log_z=log_z,
        z_max=z_max,
        modulus_mode=modulus_mode,
        modulus_params=modulus_params,
    )
    grid = sm.to_pyvista()
    rgb = np.asarray(grid["RGB"]).reshape(*np.shape(z), 3)

    return grid, rgb


def plot_landscape_pv(
    domain: Domain | None = None,
    func: Callable | None = None,
    z: np.ndarray | None = None,
    f: np.ndarray | None = None,
    resolution: int = 100,
    cmap: Colormap | None = None,
    interactive: bool = True,
    notebook: bool | None = None,
    camera_position: str | tuple = "iso",
    show_edges: bool = False,
    edge_color: str = "gray",
    z_scale: float = 1.0,
    log_z: bool = False,
    z_max: float | None = None,
    modulus_mode: str = "none",
    modulus_params: dict | None = None,
    window_size: tuple[int, int] = (800, 600),
    title: str | None = None,
    filename: str | None = None,
    return_plotter: bool = False,
    show_orientation: bool = True,
    **kwargs,
) -> Optional["pv.Plotter"]:
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
        domain, func, z, f, resolution, cmap, z_scale, log_z, z_max, modulus_mode, modulus_params
    )

    # Create plotter
    plotter_kwargs = {
        "window_size": window_size,
        "off_screen": not interactive,
    }
    if notebook is not None:
        plotter_kwargs["notebook"] = notebook

    # Add any user-provided kwargs, filtering out our function parameters
    # that might have been accidentally passed as kwargs
    filtered_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k
        not in {
            "resolution",
            "n",
            "domain",
            "func",
            "z",
            "f",
            "cmap",
            "interactive",
            "camera_position",
            "show_edges",
            "edge_color",
            "z_scale",
            "log_z",
            "z_max",
            "modulus_mode",
            "modulus_params",
            "title",
            "filename",
            "return_plotter",
            "show_orientation",
            "show",
        }
    }
    plotter_kwargs.update(filtered_kwargs)

    plotter = pv.Plotter(**plotter_kwargs)

    # Add the surface
    plotter.add_mesh(
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
        plotter.add_text(title, position="upper_edge", font_size=14)

    # Add orientation widget
    if show_orientation:
        add_axes_widget(plotter, labels=("Re", "Im", "|f|"))

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
    domain: Domain | None = None,
    func: Callable | None = None,
    z: np.ndarray | None = None,
    f: np.ndarray | None = None,
    resolution: int = 100,
    cmap: Colormap | None = None,
    interactive: bool = True,
    notebook: bool | None = None,
    camera_position: str | tuple = "iso",
    z_scale: float = 1.0,
    log_z: bool = False,
    z_max: float | None = None,
    modulus_mode: str = "none",
    modulus_params: dict | None = None,
    window_size: tuple[int, int] = (1200, 600),
    title: str | None = None,
    filename: str | None = None,
    return_plotter: bool = False,
    **kwargs,
) -> Optional["pv.Plotter"]:
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
        "window_size": window_size,
        "off_screen": not interactive,
        "shape": (1, 2),
    }
    if notebook is not None:
        plotter_kwargs["notebook"] = notebook

    # Filter kwargs to avoid passing our function parameters to PyVista
    filtered_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k
        not in {
            "resolution",
            "n",
            "domain",
            "func",
            "z",
            "f",
            "cmap",
            "interactive",
            "camera_position",
            "show_edges",
            "edge_color",
            "z_scale",
            "log_z",
            "z_max",
            "modulus_mode",
            "modulus_params",
            "title",
            "filename",
            "return_plotter",
            "show_orientation",
            "show",
        }
    }
    plotter_kwargs.update(filtered_kwargs)

    plotter = pv.Plotter(**plotter_kwargs)

    # Left subplot: Domain (identity function)
    plotter.subplot(0, 0)
    grid_domain, _ = create_complex_surface(
        domain,
        lambda x: x,
        z,
        z,
        resolution,
        cmap,
        z_scale,
        log_z,
        z_max,
        modulus_mode,
        modulus_params,
    )
    plotter.add_mesh(
        grid_domain,
        scalars="RGB",
        rgb=True,
        smooth_shading=True,
        specular=0.5,
        specular_power=15,
    )
    plotter.add_text("Domain z", position="upper_edge")
    add_axes_widget(plotter, labels=("Re", "Im", "|z|"))
    plotter.camera_position = get_camera_position(camera_position)

    # Right subplot: Codomain
    plotter.subplot(0, 1)
    grid_codomain, _ = create_complex_surface(
        domain, func, z, f, resolution, cmap, z_scale, log_z, z_max, modulus_mode, modulus_params
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
    plotter.add_text(codomain_label, position="upper_edge")
    add_axes_widget(plotter, labels=("Re", "Im", "|f|"))
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
