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
