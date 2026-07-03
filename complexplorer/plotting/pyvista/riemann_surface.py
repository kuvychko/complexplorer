"""PyVista renderer for Riemann *surfaces* of multivalued functions.

``riemann_surface_pv`` draws the multi-sheeted cover of a multivalued family (power roots
``z**(1/n)``, ``log``, or the algebraic curve ``w**2 = P(z)``) — the surface on which the
function becomes single-valued. This is distinct from ``riemann_pv``, which renders a
*single*-valued function on the Riemann *sphere*.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...core.colormap import Colormap, Phase
from ...mesh import build_riemann_surface
from .utils import (
    add_axes_widget,
    check_pyvista_available,
    ensure_pyvista_setup,
    get_camera_position,
    handle_export,
)

if TYPE_CHECKING:
    import pyvista as pv

try:
    import pyvista as pv
except ImportError:  # pragma: no cover - exercised only without the 3D backend
    pv = None

_OWN_KWARGS = frozenset(
    {
        "family",
        "n",
        "turns",
        "p",
        "r_max",
        "resolution",
        "cmap",
        "interactive",
        "notebook",
        "camera_position",
        "window_size",
        "title",
        "filename",
        "return_plotter",
        "show_orientation",
        "show",
    }
)


def riemann_surface_pv(
    family: str = "power",
    *,
    n: int = 2,
    turns: int = 3,
    p=None,
    r_max: float = 1.5,
    resolution: int = 60,
    cmap: Colormap | None = None,
    interactive: bool = True,
    notebook: bool | None = None,
    camera_position: str | tuple = (2.5, 2.5, 2.5),
    window_size: tuple[int, int] = (800, 800),
    title: str | None = None,
    filename: str | None = None,
    show_orientation: bool = True,
    return_plotter: bool = False,
    **kwargs,
) -> pv.Plotter | None:
    """Render the Riemann surface of a multivalued family with PyVista.

    Parameters
    ----------
    family : str, default="power"
        ``"power"`` (``z**(1/n)``), ``"log"``, or ``"algebraic"`` (``w**2 = P(z)``).
    n : int, default=2
        Sheet count for the power family (sqrt=2, cbrt=3, ...).
    turns : int, default=3
        Number of 2*pi turns for the log helicoid.
    p : sequence of numbers, optional
        Polynomial coefficients of ``P`` for the algebraic family (``numpy.polyval``
        order, highest degree first); e.g. ``[1, 0, -1, 0]`` is the elliptic curve
        ``w**2 = z**3 - z``. Required when ``family="algebraic"``.
    r_max : float, default=1.5
        Radius in the ``z``-plane that the surface spans. For the algebraic family choose
        it to enclose the interesting branch points (the roots of ``P``).
    resolution : int, default=60
        Radial sample count.
    cmap : Colormap, optional
        Colormap for the phase of the value. Defaults to ``Phase(n_phi=6, v_base=0.6)``.
    interactive : bool, default=True
        Show an interactive window. If False, render off-screen.
    notebook, camera_position, window_size, title, filename, show_orientation, return_plotter
        Standard PyVista renderer options (see ``riemann_pv``).

    Returns
    -------
    pyvista.Plotter or None
        The plotter if ``return_plotter`` is True, else None.
    """
    check_pyvista_available()
    ensure_pyvista_setup()

    if cmap is None:
        cmap = Phase(n_phi=6, v_base=0.6)

    sm = build_riemann_surface(
        family, n=n, turns=turns, p=p, r_max=r_max, resolution=resolution, cmap=cmap
    )
    mesh = sm.to_pyvista()

    plotter_kwargs = {"window_size": window_size, "off_screen": not interactive}
    if notebook is not None:
        plotter_kwargs["notebook"] = notebook
    plotter_kwargs.update({k: v for k, v in kwargs.items() if k not in _OWN_KWARGS})

    plotter = pv.Plotter(**plotter_kwargs)
    plotter.add_mesh(
        mesh,
        scalars="RGB",
        rgb=True,
        smooth_shading=True,
        specular=0.5,
        specular_power=20,
        diffuse=0.8,
        ambient=0.2,
        show_edges=False,
    )
    plotter.camera_position = get_camera_position(camera_position)
    if title:
        plotter.add_text(title, position="upper_edge", font_size=14)
    if show_orientation:
        z_label = "Im w" if family == "log" else "Re w"
        add_axes_widget(plotter, labels=("Re", "Im", z_label))
    plotter.set_background("white")

    if filename:
        if interactive:
            plotter.show()
        handle_export(plotter, filename, interactive)
    elif interactive:
        plotter.show()

    if return_plotter:
        return plotter
    return None
