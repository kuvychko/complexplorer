"""Tests for the Riemann surface PyVista renderer (add-riemann-surfaces)."""

import warnings

import pytest

pytest.importorskip("pyvista")

from complexplorer import riemann_surface_pv
from complexplorer.utils.validation import ValidationError


class TestRiemannSurfacePV:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"family": "power", "n": 2},
            {"family": "power", "n": 3},
            {"family": "log", "turns": 2},
            {"family": "algebraic", "p": [1, 0, -1, 0]},
        ],
    )
    def test_returns_plotter_without_rendering(self, kwargs):
        warnings.simplefilter("ignore")
        plotter = riemann_surface_pv(
            **kwargs, resolution=25, interactive=False, return_plotter=True
        )
        assert plotter is not None
        plotter.close()

    def test_unknown_family_raises(self):
        with pytest.raises(ValidationError):
            riemann_surface_pv(family="nope", interactive=False, return_plotter=True)

    def test_algebraic_without_p_raises(self):
        with pytest.raises(ValidationError):
            riemann_surface_pv(family="algebraic", interactive=False, return_plotter=True)

    def test_algebraic_screenshot_export(self, tmp_path):
        warnings.simplefilter("ignore")
        out = tmp_path / "elliptic.png"
        riemann_surface_pv(
            family="algebraic",
            p=[1, 0, -1, 0],
            r_max=1.6,
            resolution=25,
            interactive=False,
            filename=str(out),
        )
        assert out.exists() and out.stat().st_size > 0
