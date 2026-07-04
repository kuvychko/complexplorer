"""High-level API for common complexplorer workflows.

This module provides convenient functions for typical use cases.
"""

from collections.abc import Callable

from complexplorer.core.colormap import Phase
from complexplorer.core.domain import Domain, Rectangle
from complexplorer.exceptions import ValidationError

# Import plotting functions. 2D is matplotlib; 3D/Riemann are PyVista (a required core
# dependency as of 3.0 — there is no matplotlib 3D backend).
from complexplorer.plotting.matplotlib.plot_2d import plot as plot_2d


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
    # Remember whether the caller supplied a domain: in Riemann mode a domain masks the
    # sphere, so the Rectangle(4, 4) default must NOT be forwarded (default = full sphere).
    domain_supplied = domain is not None
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

        if domain_supplied:
            kwargs["domain"] = domain
        return riemann_pv(func, **kwargs)
    raise ValidationError(f"Unknown mode: {mode}")


# Preset configurations for common use cases
class Presets:
    """Named plot-configuration presets (colormap + resolution bundles).

    Each preset returns a plain dict of keyword arguments to spread into a
    plotting entry point, e.g. ``quick_plot(f, **Presets.publication_ready())``.

    Not to be confused with the function preset registry ``complexplorer.catalog``,
    whose ``FunctionPreset`` entries describe curated *functions* (expression,
    domain/colormap/scaling specs, singularity answer keys) rather than plot settings.
    """

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


__all__ = [
    "quick_plot",
    "Presets",
]
