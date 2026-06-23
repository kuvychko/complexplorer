"""Tests for quick_plot backend selection (fix-quick-plot-backend)."""

import warnings

import pytest

import complexplorer.api as api_mod
from complexplorer.api import quick_plot


def _f(z):
    return z**2 - 1


class TestPyVistaDefault:
    """3D/Riemann default to PyVista when available (per the backend policy)."""

    def test_riemann_with_modulus_mode(self):
        pytest.importorskip("pyvista")
        warnings.simplefilter("ignore")
        # Previously raised TypeError (defaulted to the matplotlib riemann, which rejects
        # modulus_mode). No real render: return the plotter, don't show/screenshot.
        plotter = quick_plot(
            _f, mode="riemann", modulus_mode="arctan", interactive=False, return_plotter=True
        )
        assert plotter is not None
        plotter.close()

    def test_3d_defaults_to_pyvista(self):
        pytest.importorskip("pyvista")
        warnings.simplefilter("ignore")
        plotter = quick_plot(_f, mode="3d", interactive=False, return_plotter=True)
        assert plotter is not None
        plotter.close()

    def test_backend_not_leaked(self):
        pytest.importorskip("pyvista")
        warnings.simplefilter("ignore")
        # Explicit backend="pyvista" must be popped, not forwarded into the plotter.
        plotter = quick_plot(
            _f, mode="3d", backend="pyvista", interactive=False, return_plotter=True
        )
        assert plotter is not None
        plotter.close()


class TestMatplotlibFallback:
    def test_falls_back_to_matplotlib_without_pyvista(self, monkeypatch):
        monkeypatch.setattr(api_mod, "HAS_PYVISTA", False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ax = quick_plot(_f, mode="3d", resolution=20)
        assert ax is not None  # matplotlib axes from the (deprecated) fallback

    def test_explicit_matplotlib_backend(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ax = quick_plot(_f, mode="3d", resolution=20, backend="matplotlib")
        assert ax is not None


def test_2d_unaffected():
    warnings.simplefilter("ignore")
    ax = quick_plot(_f, mode="2d", resolution=30)
    assert ax is not None


def test_api_pyvista_detection_matches_package():
    """Regression: api.HAS_PYVISTA must agree with the package flag (was wrongly False due
    to a circular import when detection imported the wrapper functions)."""
    import complexplorer as cp

    assert api_mod.HAS_PYVISTA == cp.HAS_PYVISTA
