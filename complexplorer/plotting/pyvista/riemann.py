"""Riemann sphere visualization using PyVista.

This module provides high-performance visualization of complex functions
on the Riemann sphere with various modulus scaling options.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Optional

import numpy as np
import pyvista as pv

from ...core.colormap import Colormap, Phase
from ...core.field import sample_sphere
from ...mesh import build_relief
from ...utils.mesh_distortion import get_default_scaling_params
from .utils import (
    add_axes_widget,
    ensure_pyvista_setup,
    finalize_plot,
    get_camera_position,
    reject_unknown_kwargs,
)

if TYPE_CHECKING:
    from ...core.domain import Domain


def riemann_pv(
    func: Callable,
    resolution: int = 100,
    cmap: Colormap | None = None,
    domain: Optional["Domain"] = None,
    interactive: bool = True,
    notebook: bool | None = None,
    camera_position: str | tuple = (2.5, 2.5, 2.5),
    window_size: tuple[int, int] = (800, 800),
    title: str | None = None,
    filename: str | None = None,
    show_orientation: bool = True,
    show_grid: bool = False,
    modulus_mode: str = "constant",
    modulus_params: dict | None = None,
    return_plotter: bool = False,
    **kwargs,
) -> Optional["pv.Plotter"]:
    """Plot complex function on the Riemann sphere using PyVista.

    .. versionchanged:: 2.2
        The sphere orientation is corrected to the canonical convention (``z = 0`` at the
        south pole), matching the matplotlib ``riemann`` renderer and the STL ornament.
        Previously this renderer was vertically mirrored relative to the rest of the
        library.

    This function provides high-performance, interactive visualization
    of complex functions on the Riemann sphere with various options
    for incorporating magnitude information.

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
        Reserved. Passing any keyword argument here raises ``ValidationError``; removed 2.x
        names are reported with their 3.0 replacement (e.g. ``n_theta``/``n_phi`` →
        ``resolution``, ``show`` → ``interactive``).

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
    reject_unknown_kwargs(kwargs)
    ensure_pyvista_setup()

    # Default colormap
    if cmap is None:
        cmap = Phase(n_phi=6, v_base=0.6)

    # Default modulus parameters
    if modulus_params is None:
        modulus_params = get_default_scaling_params(modulus_mode, for_stl=False)

    # Sample on the unit sphere and build the relief via the surface kernel. The kernel
    # uses the canonical projection (z=0 at the south pole), which corrects riemann_pv's
    # previously mirrored orientation to agree with matplotlib `riemann` and the STL
    # ornament. See docs/development/backend-policy.md and the riemann-sphere capability.
    field = sample_sphere(func, resolution=resolution, domain=domain)
    sm = build_relief(field, cmap=cmap, scaling=modulus_mode, scaling_params=modulus_params)
    mesh = sm.to_pyvista()

    # Create plotter
    plotter_kwargs = {
        "window_size": window_size,
        "off_screen": not interactive,
    }
    if notebook is not None:
        plotter_kwargs["notebook"] = notebook

    plotter = pv.Plotter(**plotter_kwargs)

    # Add the sphere
    plotter.add_mesh(
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
        add_lat_long_grid(plotter, radius=1.0, n_lat=10, n_long=12, modulus_mode=modulus_mode)

    # Set camera
    plotter.camera_position = get_camera_position(camera_position)

    # Add title
    if title:
        plotter.add_text(title, position="upper_edge", font_size=14)

    # Add orientation widget with complex plane labels
    if show_orientation:
        add_axes_widget(plotter, labels=("Re", "Im", "z"))

    # Set background
    plotter.set_background("white")

    return finalize_plot(plotter, filename, interactive, return_plotter)


def add_lat_long_grid(
    plotter: "pv.Plotter",
    radius: float = 1.0,
    n_lat: int = 10,
    n_long: int = 12,
    modulus_mode: str = "constant",
    color: str = "black",
    line_width: float = 1.0,
    opacity: float = 0.5,
) -> None:
    """Add latitude and longitude grid lines to the Riemann sphere.

    The grid is always drawn on the unit sphere; under a non-constant ``modulus_mode`` it does
    not follow the modulus distortion (a warning is emitted).

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
        Modulus scaling mode (only used to decide whether to warn about the flat grid).
    color : str
        Color of grid lines.
    line_width : float
        Width of grid lines.
    opacity : float
        Opacity of grid lines.
    """
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
        cells = np.column_stack(
            [
                np.full(len(lat_points) - 1, 2, dtype=int),
                np.arange(len(lat_points) - 1),
                np.arange(1, len(lat_points)),
            ]
        )
        polyline.lines = cells.ravel()

        plotter.add_mesh(
            polyline, color=color, line_width=line_width, opacity=opacity, ambient=1.0, diffuse=0.0
        )

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
        cells = np.column_stack(
            [
                np.full(len(long_points) - 1, 2, dtype=int),
                np.arange(len(long_points) - 1),
                np.arange(1, len(long_points)),
            ]
        )
        polyline.lines = cells.ravel()

        plotter.add_mesh(
            polyline, color=color, line_width=line_width, opacity=opacity, ambient=1.0, diffuse=0.0
        )

    # Add warning if modulus scaling is used
    if modulus_mode != "constant":
        import warnings

        warnings.warn(
            "Grid lines are shown on unit sphere; they don't follow modulus distortion",
            UserWarning,
            stacklevel=2,
        )
