"""Tests for quick_plot backend selection (fix-quick-plot-backend)."""

import warnings

import pytest

from complexplorer.api import quick_plot
from complexplorer.utils.validation import ValidationError


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


class TestMatplotlibBackendRemoved:
    def test_matplotlib_backend_for_3d_raises(self):
        with pytest.raises(ValidationError, match="matplotlib 3D backend was removed"):
            quick_plot(_f, mode="3d", backend="matplotlib")

    def test_matplotlib_backend_for_riemann_raises(self):
        with pytest.raises(ValidationError, match="matplotlib 3D backend was removed"):
            quick_plot(_f, mode="riemann", backend="matplotlib")


def test_2d_unaffected():
    warnings.simplefilter("ignore")
    ax = quick_plot(_f, mode="2d", resolution=30)
    assert ax is not None
