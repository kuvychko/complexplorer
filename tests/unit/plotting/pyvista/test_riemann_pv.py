"""Tests for PyVista Riemann sphere plotting.

These exercise the real renderer off-screen (PyVista is a required dependency as of 3.0),
rather than mocking ``pyvista.Plotter`` — so they actually cover mesh construction, kwarg
handling, and export.
"""

import os
import sys

import numpy as np
import pytest

pyvista = pytest.importorskip("pyvista")

from complexplorer.core.colormap import LogRings, Phase
from complexplorer.exceptions import ValidationError
from complexplorer.plotting.pyvista.riemann import riemann_pv

# A real offscreen VTK screenshot render crashes (access violation) on the headless Windows
# CI runner (no GPU / no reliable offscreen GL). Building the plotter is fine; only the
# render-to-file step is unsafe there. Linux CI exercises it via the headless-display action.
_NO_OFFSCREEN_RENDER = sys.platform == "win32" and os.environ.get("CI") == "true"


class TestRiemannPV:
    """Real off-screen rendering of the Riemann sphere."""

    def test_returns_plotter_when_requested(self):
        p = riemann_pv(lambda z: z**2, resolution=20, interactive=False, return_plotter=True)
        assert isinstance(p, pyvista.Plotter)
        p.close()

    def test_returns_none_without_return_plotter(self):
        result = riemann_pv(lambda z: z**2, resolution=20, interactive=False)
        assert result is None

    def test_custom_colormap(self):
        cmap = Phase(n_phi=12, auto_scale_r=True)
        p = riemann_pv(
            lambda z: (z - 1) / (z + 1),
            cmap=cmap,
            resolution=20,
            interactive=False,
            return_plotter=True,
        )
        assert isinstance(p, pyvista.Plotter)
        p.close()

    @pytest.mark.parametrize("mode", ["constant", "arctan", "logarithmic", "adaptive"])
    def test_modulus_scaling_modes(self, mode):
        p = riemann_pv(
            lambda z: z**3 - z,
            modulus_mode=mode,
            resolution=16,
            interactive=False,
            return_plotter=True,
        )
        assert isinstance(p, pyvista.Plotter)
        p.close()

    def test_custom_modulus_params(self):
        p = riemann_pv(
            lambda z: 1 / (z - 1),
            modulus_mode="arctan",
            modulus_params={"r_min": 0.3, "r_max": 0.9},
            resolution=16,
            interactive=False,
            return_plotter=True,
        )
        assert isinstance(p, pyvista.Plotter)
        p.close()

    @pytest.mark.skipif(
        _NO_OFFSCREEN_RENDER, reason="offscreen VTK screenshot crashes on headless Windows CI"
    )
    def test_save_screenshot(self, tmp_path):
        out = tmp_path / "riemann_test.png"
        riemann_pv(lambda z: z**2 + 1, resolution=20, interactive=False, filename=str(out))
        assert out.exists()
        assert out.stat().st_size > 0

    def test_logarithmic_rings_colormap(self):
        p = riemann_pv(
            lambda z: z**2 / (z**2 + 1),
            cmap=LogRings(),
            resolution=16,
            interactive=False,
            return_plotter=True,
        )
        assert isinstance(p, pyvista.Plotter)
        p.close()

    def test_essential_singularity(self):
        def func(z):
            with np.errstate(divide="ignore", invalid="ignore"):
                result = np.exp(1 / z)
            return np.where(np.isfinite(result), result, 0)

        p = riemann_pv(
            func, modulus_mode="logarithmic", resolution=16, interactive=False, return_plotter=True
        )
        assert isinstance(p, pyvista.Plotter)
        p.close()

    def test_custom_resolution(self):
        for res in (10, 40):
            p = riemann_pv(
                lambda z: z**4 - 1, resolution=res, interactive=False, return_plotter=True
            )
            assert isinstance(p, pyvista.Plotter)
            p.close()

    def test_title(self):
        p = riemann_pv(
            lambda z: np.sin(z),
            title="sin(z)",
            resolution=16,
            interactive=False,
            return_plotter=True,
        )
        assert isinstance(p, pyvista.Plotter)
        p.close()


class TestRiemannPVKwargValidation:
    """Removed 2.x keyword arguments are rejected with a clear error, not forwarded."""

    @pytest.mark.parametrize(
        "bad",
        [
            {"n_theta": 30},
            {"n_phi": 30},
            {"show": False},
            {"project_from_north": False},
            {"bogus": 1},
        ],
    )
    def test_removed_or_unknown_kwargs_rejected(self, bad):
        with pytest.raises(ValidationError):
            riemann_pv(lambda z: z**2, resolution=16, interactive=False, **bad)

    def test_error_names_replacement(self):
        with pytest.raises(ValidationError, match="resolution"):
            riemann_pv(lambda z: z**2, n_theta=30, interactive=False)
