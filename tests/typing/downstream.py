"""A program that uses complexplorer the way a user does, checked as a consumer would be.

This file is never executed by the test suite. It exists to be type-checked: the package ships
``py.typed``, so its annotations are a promise, and the failure this guards against is an
annotation that *rejects correct user code*. Checking the library against itself cannot catch
that -- only checking it from the outside can.

Every form below is one somebody writes in practice. If a change to the public annotations makes
any of them an error, the promise has been broken and the typing job fails.
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes

import complexplorer as cp

# --------------------------------------------------------------------------------------
# The four ways people write the function
# --------------------------------------------------------------------------------------

# 1. A lambda, which is how most portraits start.
rational = lambda z: (z**2 - 1) / (z**2 + 1)  # noqa: E731


# 2. A def annotated for arrays, which is how a careful user writes it.
def annotated(z: np.ndarray) -> np.ndarray:
    return np.exp(1 / z)


# 3. A def with no annotations at all.
def unannotated(z):
    return z**3 - 1


# 4. A callable object: a transfer function is a plain callable, and the library leans on that.
notch = cp.ee.TransferFunction([1, 0, 4], [1, 1.2, 5, 2])

# --------------------------------------------------------------------------------------
# The entry points, with what they return
# --------------------------------------------------------------------------------------

domain: cp.Domain = cp.Rectangle(re_length=4, im_length=4)
disk: cp.Domain = cp.Disk(radius=2)
annulus: cp.Domain = cp.Annulus(inner_radius=0.2, outer_radius=3)
composite: cp.Domain = (disk | annulus) - cp.Disk(radius=0.1)

cmap: cp.Colormap = cp.Phase(phase_sectors=6, auto_scale_r=True)
perceptual: cp.Colormap = cp.CubehelixPhase(phase_sectors=6)
pattern: cp.Colormap = cp.Chessboard(spacing=0.5)

axes: Axes = cp.plot(domain, rational, cmap=cmap, legend=True)
axes_from_annotated: Axes = cp.plot(domain, annotated, cmap=perceptual)
axes_from_unannotated: Axes = cp.plot(domain, unannotated, cmap=pattern)
axes_from_callable_object: Axes = cp.plot(domain, notch, cmap=cmap)

# quick_plot dispatches, so its return is a union the caller narrows.
quick = cp.quick_plot(rational)
quick_annotated = cp.quick_plot(annotated)  # the array-typed def must be accepted
quick_object = cp.quick_plot(notch)  # and so must a callable object
quick_3d = cp.quick_plot(rational, mode="3d")

# Render settings spread into an entry point: PlotPresets configures a render,
# catalog supplies a function.
settings = cp.PlotPresets.publication_ready()
cp.quick_plot(rational, **settings)

preset: cp.FunctionPreset = cp.catalog.get("pole_flower_10")
cp.quick_plot(preset.func, **cp.PlotPresets.high_contrast())
ornament_presets: list[cp.FunctionPreset] = cp.catalog.filter(tag="ornament")

# The colormap contract: complex array in, RGB array out.
samples: np.ndarray = np.array([1 + 1j, 0j, np.inf + 0j])
rgb: np.ndarray = cmap.rgb(samples)

# Engineering mode.
poles: np.ndarray = notch.poles
stable: bool = notch.is_stable
omega, response = notch.frequency_response()

# STL export returns the path it wrote.
written: str = cp.create_ornament(rational, "ornament.stl", size_mm=60)

# The gallery returns its manifest.
manifest: dict = cp.generate_gallery("bundle", selection=["identity"], dpi=50, resolution=60)
