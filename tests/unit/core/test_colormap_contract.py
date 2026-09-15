"""Every public colormap honours the colormap contract.

`reconcile-with-2-0-release` ports nine colormap families from 2.0.0 onto rev3's stricter
contract (finite, in-gamut, deterministic output for ANY input, including non-finite values).
These checks run against the whole exported family so a future colormap cannot skip them.
"""

import inspect

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import complexplorer as cp
from complexplorer.core.colormap import BasePhasePortrait, Colormap

# Values the contract must survive: origin, poles, NaN, infinities, tiny and huge moduli.
Z = np.array(
    [
        [0, 1e-12, 1e12, 1 + 1j],
        [np.nan, np.inf, -np.inf, complex(np.nan, np.nan)],
        [1, -1, 1j, -1j],
    ],
    dtype=complex,
)


def _concrete_colormaps():
    out = []
    for name in cp.__all__:
        obj = getattr(cp, name)
        if (
            inspect.isclass(obj)
            and issubclass(obj, Colormap)
            and obj not in (Colormap, BasePhasePortrait)
            and not inspect.isabstract(obj)
        ):
            out.append(name)
    return sorted(out)


COLORMAPS = _concrete_colormaps()
# The enhancement parameters every phase portrait shares; pattern colormaps take none.
ENHANCED = {"phase_sectors": 6}


def _make(name, enhanced):
    cls = getattr(cp, name)
    if enhanced and issubclass(cls, BasePhasePortrait):
        return cls(**ENHANCED)
    if enhanced and name == "PolarChessboard":
        return cls(**ENHANCED)
    return cls()


def test_the_exported_family_is_not_empty():
    # Guards against the parametrisation silently collecting nothing.
    assert len(COLORMAPS) >= 13, COLORMAPS


@pytest.mark.parametrize("name", COLORMAPS)
@pytest.mark.parametrize("enhanced", [False, True], ids=["default", "enhanced"])
class TestColormapContract:
    def test_rgb_is_finite_and_in_gamut(self, name, enhanced):
        rgb = _make(name, enhanced).rgb(Z)
        assert rgb.shape == (*Z.shape, 3)
        assert np.all(np.isfinite(rgb)), f"{name} produced non-finite RGB"
        assert np.all((rgb >= 0.0) & (rgb <= 1.0)), f"{name} left the [0, 1] gamut"

    def test_rgb_matches_hsv(self, name, enhanced):
        cmap = _make(name, enhanced)
        expected = matplotlib.colors.hsv_to_rgb(cmap.hsv(Z))
        assert np.array_equal(cmap.rgb(Z), expected)

    def test_output_is_deterministic(self, name, enhanced):
        first = _make(name, enhanced).rgb(Z)
        second = _make(name, enhanced).rgb(Z)
        assert np.array_equal(first, second), f"{name} is not deterministic"

    def test_non_finite_and_masked_points_use_the_out_of_domain_colour(self, name, enhanced):
        cmap = _make(name, enhanced)
        expected = matplotlib.colors.hsv_to_rgb(np.array(cmap.out_of_domain_hsv))
        rgb = cmap.rgb(Z)
        non_finite = ~np.isfinite(Z)
        assert np.allclose(rgb[non_finite], expected)

        outmask = np.zeros(Z.shape, dtype=bool)
        outmask[0, 3] = True
        masked = cmap.rgb(Z, outmask=outmask)
        assert np.allclose(masked[0, 3], expected)

    def test_renders_through_the_phase_wheel_legend(self, name, enhanced):
        cmap = _make(name, enhanced)
        fig, ax = plt.subplots(figsize=(2, 2))
        try:
            cp.plot(cp.Rectangle(2, 2), lambda z: z, cmap=cmap, ax=ax, resolution=48, legend=True)
        finally:
            plt.close(fig)
