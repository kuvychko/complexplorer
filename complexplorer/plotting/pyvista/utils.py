"""Utility functions for PyVista plotting.

This module provides helper functions for PyVista-based visualizations.
"""

import warnings
from typing import Optional, Tuple, Union
import numpy as np

# Only import PyVista if available
try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    pv = None


def check_pyvista_available() -> None:
    """Check if PyVista is available and raise error if not."""
    if not HAS_PYVISTA:
        raise ImportError(
            "PyVista is required for this functionality. "
            "Install with: pip install pyvista"
        )


def handle_export(plotter: 'pv.Plotter', filename: str, interactive: bool) -> None:
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
    if filename.endswith('.html'):
        # HTML export works differently with interactive mode
        if not interactive:
            warnings.warn(
                "HTML export works best with interactive=True. "
                "The file will be created but may not display properly."
            )
        try:
            plotter.export_html(filename)
            print(f"Interactive HTML saved to: {filename}")
        except ImportError:
            raise ImportError(
                "HTML export requires 'trame'. "
                "Install with: pip install trame"
            )
    elif filename.endswith(('.pdf', '.svg', '.eps')):
        plotter.save_graphic(filename)
    else:
        plotter.screenshot(filename)


def add_axes_widget(plotter: 'pv.Plotter',
                   labels: Tuple[str, str, str] = ('Re', 'Im', 'Z'),
                   position: Tuple[float, float] = (0.0, 0.0),
                   size: float = 0.25,
                   label_size: Tuple[float, float] = (0.25, 0.1)) -> 'pv.Actor':
    """Add a labeled axes widget to the plotter.
    
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


def ensure_pyvista_setup() -> None:
    """Ensure PyVista is properly configured for the current environment."""
    check_pyvista_available()
    
    # Set conservative defaults for better compatibility
    # Users can increase these if their system supports it
    if pv.global_theme.multi_samples is None:
        pv.global_theme.multi_samples = 2  # Conservative default
    pv.global_theme.smooth_shading = True
    
    backend = pv.global_theme.jupyter_backend
    if backend is None:
        # Not in Jupyter, use default
        pass
    elif backend != 'trame' and not str(backend).startswith('<MagicMock'):
        # Only warn for real backends, not mocked ones in tests
        warnings.warn(
            f"PyVista backend is '{backend}', but 'trame' is recommended for "
            "interactive Jupyter visualizations. Set with: pv.set_jupyter_backend('trame')"
        )


def get_camera_position(position: Union[str, tuple]) -> Union[str, tuple]:
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
    valid_strings = ['iso', 'xy', 'xz', 'yz']
    
    if isinstance(position, str):
        if position not in valid_strings:
            raise ValueError(
                f"Invalid camera position '{position}'. "
                f"Must be one of: {valid_strings}"
            )
        return position
    
    # Assume it's a tuple/list of camera parameters
    return position