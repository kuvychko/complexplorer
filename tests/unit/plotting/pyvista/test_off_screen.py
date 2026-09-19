"""A session-level off-screen instruction must beat a per-call default the user never chose.

PyVista's documented way to go headless is ``pv.OFF_SCREEN = True``, or the ``PYVISTA_OFF_SCREEN``
environment variable that sets it. Every renderer here used to compute ``off_screen = not
interactive``, which overrode that global -- so a script that had correctly set it still tried to
open a window, and blocked. These tests pin the three cases apart: the global alone, the argument
alone, and neither.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import pyvista as pv

import complexplorer as cp
from complexplorer.plotting.pyvista.utils import should_render_off_screen


class TestTheDecisionItself:
    """The helper is the whole contract; the renderers just call it."""

    def test_the_global_alone_is_enough(self, monkeypatch):
        monkeypatch.setattr(pv, "OFF_SCREEN", True)
        assert should_render_off_screen(interactive=True) is True

    def test_the_argument_alone_is_enough(self, monkeypatch):
        monkeypatch.setattr(pv, "OFF_SCREEN", False)
        assert should_render_off_screen(interactive=False) is True

    def test_both_together(self, monkeypatch):
        monkeypatch.setattr(pv, "OFF_SCREEN", True)
        assert should_render_off_screen(interactive=False) is True

    def test_neither_means_a_window(self, monkeypatch):
        """Guard against the fix quietly making everything headless."""
        monkeypatch.setattr(pv, "OFF_SCREEN", False)
        assert should_render_off_screen(interactive=True) is False

    def test_the_environment_variable_reaches_the_decision(self, monkeypatch):
        """PYVISTA_OFF_SCREEN sets pv.OFF_SCREEN at import; this is that path, simulated."""
        monkeypatch.setenv("PYVISTA_OFF_SCREEN", "true")
        monkeypatch.setattr(pv, "OFF_SCREEN", True)
        assert should_render_off_screen(interactive=True) is True


def _plotter_kwargs(mock_plotter_class) -> dict:
    """The keyword arguments the renderer used to construct its plotter."""
    assert mock_plotter_class.called, "no plotter was constructed"
    return mock_plotter_class.call_args.kwargs


class TestEveryRendererHonoursIt:
    """Each renderer must ask the helper, not re-derive the answer."""

    FUNC = staticmethod(lambda z: (z**2 - 1) / (z**2 + 1))

    def _call(self, name):
        domain = cp.Rectangle(2, 2)
        if name == "plot_landscape_pv":
            cp.plot_landscape_pv(domain, self.FUNC, resolution=10)
        elif name == "pair_plot_landscape_pv":
            cp.pair_plot_landscape_pv(domain, self.FUNC, resolution=10)
        elif name == "riemann_pv":
            cp.riemann_pv(self.FUNC, resolution=10)
        elif name == "riemann_surface_pv":
            cp.riemann_surface_pv("power", n=2, resolution=10)
        else:  # pragma: no cover - guard against a typo in the parametrization
            raise AssertionError(name)

    @pytest.mark.parametrize(
        "renderer",
        ["plot_landscape_pv", "pair_plot_landscape_pv", "riemann_pv", "riemann_surface_pv"],
    )
    def test_the_global_prevents_a_window(self, renderer, monkeypatch):
        monkeypatch.setattr(pv, "OFF_SCREEN", True)
        with patch("pyvista.Plotter", MagicMock()) as plotter_class:
            self._call(renderer)
            assert _plotter_kwargs(plotter_class)["off_screen"] is True, (
                f"{renderer} opened a window despite pv.OFF_SCREEN being set"
            )

    @pytest.mark.parametrize(
        "renderer",
        ["plot_landscape_pv", "pair_plot_landscape_pv", "riemann_pv", "riemann_surface_pv"],
    )
    def test_the_default_is_unchanged(self, renderer, monkeypatch):
        """With neither switch set, behaviour is exactly what it was."""
        monkeypatch.setattr(pv, "OFF_SCREEN", False)
        with patch("pyvista.Plotter", MagicMock()) as plotter_class:
            self._call(renderer)
            assert _plotter_kwargs(plotter_class)["off_screen"] is False
