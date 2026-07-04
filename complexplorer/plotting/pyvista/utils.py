"""Utility functions for PyVista plotting.

This module provides helper functions for PyVista-based visualizations.
"""

import warnings

import pyvista as pv

from complexplorer.exceptions import ValidationError


def handle_export(plotter: "pv.Plotter", filename: str, interactive: bool) -> None:
    """Handle file export based on extension.

    Parameters
    ----------
    plotter : pv.Plotter
        The plotter to export.
    filename : str
        Output filename with extension.
    interactive : bool
        Whether in interactive mode.
    """
    if filename.endswith(".html"):
        # HTML export works differently with interactive mode
        if not interactive:
            warnings.warn(
                "HTML export works best with interactive=True. "
                "The file will be created but may not display properly.",
                stacklevel=2,
            )
        try:
            plotter.export_html(filename)
            print(f"Interactive HTML saved to: {filename}")
        except ImportError as err:
            raise ImportError(
                "HTML export requires 'trame'. Install with: pip install trame"
            ) from err
    elif filename.endswith((".pdf", ".svg", ".eps")):
        plotter.save_graphic(filename)
    else:
        plotter.screenshot(filename)


def finalize_plot(
    plotter: "pv.Plotter",
    filename: str | None,
    interactive: bool,
    return_plotter: bool = False,
) -> "pv.Plotter | None":
    """Show and/or export a plotter, then optionally return it.

    Interactive plots are shown first; a filename (if any) is exported afterward. This is the
    shared tail used by every PyVista entry point so the show/export/return behavior stays
    identical across them.
    """
    if interactive:
        plotter.show()
    if filename:
        handle_export(plotter, filename, interactive)
    return plotter if return_plotter else None


def add_axes_widget(
    plotter: "pv.Plotter",
    labels: tuple[str, str, str] = ("Re", "Im", "Z"),
    size: float = 0.25,
    label_size: tuple[float, float] = (0.25, 0.1),
) -> None:
    """Add a labeled axes widget to the plotter.

    Parameters
    ----------
    plotter : pv.Plotter
        The PyVista plotter object.
    labels : tuple of str, default=('Re', 'Im', 'Z')
        Labels for x, y, z axes.
    size : float, default=0.25
        Size of the widget as fraction of viewport (off-screen rendering only).
    label_size : tuple of float, default=(0.25, 0.1)
        Width and height of the axes label actors (values between 0 and 1).
    """
    kwargs = dict(
        xlabel=labels[0],
        ylabel=labels[1],
        zlabel=labels[2],
        line_width=4,
        labels_off=False,
        interactive=True,
        label_size=label_size,
        cone_radius=0.4,
        shaft_length=0.8,
        tip_length=0.2,
        ambient=0.5,
        color="black",
    )
    # For static (off-screen) rendering, pin the widget into a viewport corner; the
    # interactive path works better with the default placement (esp. in Jupyter).
    if plotter.off_screen:
        kwargs["viewport"] = (0, 0, size, size)
    plotter.add_axes(**kwargs)


def ensure_pyvista_setup():
    """Ensure PyVista is properly configured for the current environment."""
    # Set conservative defaults for better compatibility
    # Users can increase these if their system supports it
    if pv.global_theme.multi_samples is None:
        pv.global_theme.multi_samples = 2  # Conservative default
    pv.global_theme.smooth_shading = True

    backend = pv.global_theme.jupyter_backend
    if backend is not None and backend != "trame":
        warnings.warn(
            f"PyVista backend is '{backend}', but 'trame' is recommended for "
            "interactive Jupyter visualizations. Set with: pv.set_jupyter_backend('trame')",
            stacklevel=2,
        )


# Keyword arguments accepted by earlier (2.x) releases and removed in 3.0. Mapped to their
# current replacement (or None when there is no replacement) so the error can guide callers.
_REMOVED_KWARGS = {
    "n": "resolution",
    "n_theta": "resolution",
    "n_phi": "resolution",
    "show": "interactive",
    "radius": None,
    "project_from_north": None,
}


def reject_unknown_kwargs(kwargs: dict) -> None:
    """Raise a clear ``ValidationError`` for any keyword argument the renderer does not accept.

    The PyVista entry points take an explicit signature and do not forward arbitrary keyword
    arguments to ``pyvista.Plotter``. Passing an unknown one — including a 2.x name removed in
    3.0 — raises here with the current replacement named, instead of a raw ``TypeError`` from
    PyVista or a silent no-op.
    """
    if not kwargs:
        return
    parts = []
    for key in kwargs:
        if key in _REMOVED_KWARGS:
            replacement = _REMOVED_KWARGS[key]
            parts.append(
                f"{key!r} (use {replacement!r})" if replacement else f"{key!r} (removed in 3.0)"
            )
        else:
            parts.append(repr(key))
    raise ValidationError(
        "Unexpected keyword argument(s) not accepted by this renderer: " + ", ".join(parts)
    )


def get_camera_position(position: str | tuple) -> str | tuple:
    """Validate and return camera position.

    Parameters
    ----------
    position : str or tuple
        Camera position specification.

    Returns
    -------
    str or tuple
        Validated camera position.
    """
    valid_strings = ["iso", "xy", "xz", "yz"]

    if isinstance(position, str):
        if position not in valid_strings:
            raise ValidationError(
                f"Invalid camera position '{position}'. Must be one of: {valid_strings}"
            )
        return position

    # Assume it's a tuple/list of camera parameters
    return position
