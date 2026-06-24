"""Tests for non-finite colormap handling (fix-colormap-nonfinite)."""

import warnings

import matplotlib.colors as mcolors
import numpy as np
import pytest

from complexplorer.core.colormap import Chessboard, LogRings, Phase, PolarChessboard

CMAPS = [Phase(n_phi=6, v_base=0.6), Chessboard(spacing=0.5), PolarChessboard(n_phi=6), LogRings()]
_GRID = np.add.outer(np.linspace(-2, 2, 25), 1j * np.linspace(-2, 2, 25))


def _ood_rgb(cmap):
    return mcolors.hsv_to_rgb(np.array(cmap.out_of_domain_hsv))


@pytest.mark.parametrize("cmap", CMAPS, ids=lambda c: type(c).__name__)
@pytest.mark.parametrize("bad", [np.inf, np.nan, complex(np.inf, np.nan)])
def test_nonfinite_is_valid_and_out_of_domain(cmap, bad):
    warnings.simplefilter("ignore")
    z = _GRID.copy()
    z[12, 12] = bad
    rgb = cmap.rgb(z)
    assert np.all(np.isfinite(rgb))
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0
    np.testing.assert_allclose(rgb[12, 12], _ood_rgb(cmap))


@pytest.mark.parametrize("cmap", CMAPS, ids=lambda c: type(c).__name__)
def test_nonfinite_coloring_is_deterministic(cmap):
    warnings.simplefilter("ignore")
    z = _GRID.copy()
    z[12, 12] = np.nan
    np.testing.assert_array_equal(cmap.rgb(z), cmap.rgb(z))


@pytest.mark.parametrize("cmap", CMAPS, ids=lambda c: type(c).__name__)
def test_finite_points_unaffected_by_a_nonfinite_point(cmap):
    """Introducing one non-finite point must not change the colors of the other points."""
    warnings.simplefilter("ignore")
    base = cmap.rgb(_GRID)
    z = _GRID.copy()
    z[12, 12] = np.nan
    perturbed = cmap.rgb(z)
    mask = np.ones(_GRID.shape, bool)
    mask[12, 12] = False
    np.testing.assert_array_equal(base[mask], perturbed[mask])


def test_pole_function_rgb_is_finite():
    """1/z over a grid hitting the origin: rgb is finite and in-gamut (was the bug)."""
    warnings.simplefilter("ignore")
    z = np.add.outer(np.linspace(-1.5, 1.5, 25), 1j * np.linspace(-1.5, 1.5, 25))
    with np.errstate(all="ignore"):
        f = 1 / z  # non-finite at the origin grid node
    rgb = Phase(n_phi=6).rgb(f)
    assert np.all(np.isfinite(rgb)) and rgb.min() >= 0 and rgb.max() <= 1


def test_scalar_nonfinite():
    warnings.simplefilter("ignore")
    hsv = Phase(n_phi=6).hsv(np.asarray(np.nan + 0j))
    np.testing.assert_allclose(hsv, np.array(Phase(n_phi=6).out_of_domain_hsv))
