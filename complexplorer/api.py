"""High-level API for common complexplorer workflows.

This module provides convenient functions for typical use cases.
"""

from collections.abc import Callable

import numpy as np

from complexplorer.core.colormap import Phase
from complexplorer.core.domain import Domain, Rectangle

# Import plotting functions. 2D is matplotlib; 3D/Riemann are PyVista (a required core
# dependency as of 3.0 — there is no matplotlib 3D backend).
from complexplorer.plotting.matplotlib.plot_2d import plot as plot_2d
from complexplorer.utils.validation import ValidationError


def quick_plot(
    func: Callable[[complex], complex], domain: Domain | None = None, mode: str = "2d", **kwargs
):
    """Quick visualization of a complex function.

    Parameters
    ----------
    func : callable
        Complex function to visualize
    domain : Domain, optional
        Domain to plot. Defaults to Rectangle(4, 4)
    mode : str
        Plot mode: '2d', '3d', 'riemann'
    **kwargs
        Additional arguments passed to plotting function

    Returns
    -------
    Axes or Plotter object depending on mode
    """
    if domain is None:
        domain = Rectangle(4, 4)

    if "cmap" not in kwargs:
        kwargs["cmap"] = Phase(n_phi=6, auto_scale_r=True)

    # Pop the backend selector so it never leaks into the renderer. 2D is matplotlib;
    # 3D/Riemann are PyVista-only (the matplotlib 3D backend was removed in 3.0).
    backend = kwargs.pop("backend", None)

    if mode == "2d":
        return plot_2d(domain, func, **kwargs)
    if mode in ("3d", "riemann"):
        if backend == "matplotlib":
            raise ValidationError(
                "The matplotlib 3D backend was removed in 3.0; 3D/Riemann use PyVista."
            )
        if mode == "3d":
            from complexplorer.plotting.pyvista.plot_3d import plot_landscape_pv

            return plot_landscape_pv(domain, func, **kwargs)
        from complexplorer.plotting.pyvista.riemann import riemann_pv

        return riemann_pv(func, **kwargs)
    raise ValueError(f"Unknown mode: {mode}")


def analyze_function(
    func: Callable[[complex], complex],
    domain: Domain | None = None,
    show_zeros: bool = True,
    show_poles: bool = True,
    **kwargs,
):
    """Analyze a complex function with automatic feature detection.

    Parameters
    ----------
    func : callable
        Complex function to analyze
    domain : Domain, optional
        Domain to analyze. Defaults to Rectangle(4, 4)
    show_zeros : bool
        Highlight zeros of the function
    show_poles : bool
        Highlight poles of the function
    **kwargs
        Additional plotting arguments

    Returns
    -------
    dict
        Analysis results including plot object
    """
    if domain is None:
        domain = Rectangle(4, 4)

    # Use enhanced phase portrait for analysis
    cmap = Phase(n_phi=12, auto_scale_r=True, scale_radius=1.0)

    # Create the plot
    ax = plot_2d(domain, func, cmap=cmap, **kwargs)

    # Simple feature detection (more sophisticated analysis could be added)
    results = {"plot": ax, "domain": domain, "function": func, "colormap": cmap}

    # TODO: Add automatic zero/pole detection
    if show_zeros or show_poles:
        print("Note: Automatic zero/pole detection not yet implemented")

    return results


def create_animation(
    func_family: Callable[[complex, float], complex],
    domain: Domain | None = None,
    t_values: np.ndarray | None = None,
    mode: str = "2d",
    filename: str | None = None,
    **kwargs,
):
    """Create an animation of a parametric family of functions.

    Parameters
    ----------
    func_family : callable
        Function f(z, t) where t is the parameter
    domain : Domain, optional
        Domain to plot
    t_values : array-like, optional
        Parameter values. Defaults to linspace(0, 1, 30)
    mode : str
        Plot mode: '2d' or '3d'
    filename : str, optional
        Save animation to file
    **kwargs
        Additional plotting arguments

    Returns
    -------
    Animation object or saved filename
    """
    if domain is None:
        domain = Rectangle(4, 4)

    if t_values is None:
        t_values = np.linspace(0, 1, 30)

    # TODO: Implement animation functionality
    raise NotImplementedError("Animation functionality coming soon!")


def compare_functions(
    funcs: list[Callable[[complex], complex]],
    domain: Domain | None = None,
    labels: list[str] | None = None,
    mode: str = "2d",
    **kwargs,
):
    """Compare multiple complex functions side by side.

    Parameters
    ----------
    funcs : list of callables
        Functions to compare
    domain : Domain, optional
        Domain for all functions
    labels : list of str, optional
        Labels for each function
    mode : str
        Plot mode: '2d' or '3d'
    **kwargs
        Additional plotting arguments

    Returns
    -------
    Figure with subplots
    """
    if domain is None:
        domain = Rectangle(4, 4)

    if labels is None:
        labels = [f"f_{i}" for i in range(len(funcs))]

    # TODO: Implement comparison plots
    raise NotImplementedError("Comparison functionality coming soon!")


# Preset configurations for common use cases
class Presets:
    """Common preset configurations."""

    @staticmethod
    def publication_ready():
        """Settings for publication-quality figures."""
        return {"cmap": Phase(n_phi=12, auto_scale_r=True, scale_radius=0.8), "resolution": 800}

    @staticmethod
    def interactive():
        """Settings for interactive exploration."""
        return {"cmap": Phase(n_phi=6, auto_scale_r=True), "resolution": 400}

    @staticmethod
    def high_contrast():
        """Settings for high contrast visualization."""
        return {"cmap": Phase(n_phi=16, auto_scale_r=True, scale_radius=0.5), "resolution": 600}


# Export convenient aliases
visualize = quick_plot
explore = quick_plot


__all__ = [
    "quick_plot",
    "analyze_function",
    "create_animation",
    "compare_functions",
    "Presets",
    "visualize",
    "explore",
]
