"""3D plotting functions using PyVista.

This module provides high-performance, interactive 3D visualizations
using PyVista as an alternative to matplotlib-based plots.
"""

from collections.abc import Callable

import numpy as np
import pyvista as pv

from ...core.colormap import Colormap, Phase
from ...core.domain import Domain
from ...core.field import ComplexField, resolve_plane_inputs
from ...mesh import build_landscape
from .utils import (
    add_axes_widget,
    ensure_pyvista_setup,
    finalize_plot,
    get_camera_position,
    reject_unknown_kwargs,
)


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
    if cmap is None:
        cmap = Phase(n_phi=6, v_base=0.6)

    # Resolve the sampling grid, values, and out-of-domain mask (shared with the 2D backend).
    z, f, mask = resolve_plane_inputs(domain, func, z, f, resolution)

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
) -> "pv.Plotter | None":
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
        Reserved. Passing any keyword argument here raises ``ValidationError``; removed 2.x
        names are reported with their 3.0 replacement (e.g. ``n_theta``/``n_phi`` →
        ``resolution``, ``show`` → ``interactive``).

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
    reject_unknown_kwargs(kwargs)
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

    return finalize_plot(plotter, filename, interactive, return_plotter)


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
) -> "pv.Plotter | None":
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
        Overall figure title (shown above the paired views; the codomain panel keeps its own
        ``Codomain f(z)`` label).
    filename : str, optional
        Save plot to file.
    return_plotter : bool, optional
        If True, return the plotter object.
    **kwargs
        Reserved. Passing any keyword argument here raises ``ValidationError``; removed 2.x
        names are reported with their 3.0 replacement (e.g. ``n_theta``/``n_phi`` →
        ``resolution``, ``show`` → ``interactive``).

    Returns
    -------
    pv.Plotter or None
        The plotter object if return_plotter=True.
    """
    reject_unknown_kwargs(kwargs)
    ensure_pyvista_setup()

    # Create plotter with two viewports
    plotter_kwargs = {
        "window_size": window_size,
        "off_screen": not interactive,
        "shape": (1, 2),
    }
    if notebook is not None:
        plotter_kwargs["notebook"] = notebook

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
    plotter.add_text("Codomain f(z)", position="upper_edge")
    add_axes_widget(plotter, labels=("Re", "Im", "|f|"))
    plotter.camera_position = get_camera_position(camera_position)

    # Link cameras for synchronized interaction
    if interactive:
        plotter.link_views()

    # Overall figure title, kept distinct from the per-panel labels (it does not replace the
    # "Codomain f(z)" label). Placed at the top-left of the paired window.
    if title:
        plotter.subplot(0, 0)
        plotter.add_text(title, position="upper_left", font_size=12)

    return finalize_plot(plotter, filename, interactive, return_plotter)
