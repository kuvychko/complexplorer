"""Tests for quick_plot backend selection (fix-quick-plot-backend)."""

import warnings

import pytest

from complexplorer.api import quick_plot
from complexplorer.core.domain import Disk
from complexplorer.exceptions import ValidationError


def _f(z):
    return z**2 - 1


class TestRiemannDomainForwarding:
    """A caller-supplied domain is forwarded to riemann_pv; the default is not."""

    def test_supplied_domain_is_forwarded(self, monkeypatch):
        captured = {}

        def fake_riemann_pv(func, **kwargs):
            captured.update(kwargs)
            return "plotter"

        monkeypatch.setattr("complexplorer.plotting.pyvista.riemann.riemann_pv", fake_riemann_pv)
        dom = Disk(2)
        quick_plot(_f, domain=dom, mode="riemann")
        assert captured.get("domain") is dom

    def test_default_domain_not_forwarded(self, monkeypatch):
        captured = {}

        def fake_riemann_pv(func, **kwargs):
            captured.update(kwargs)
            return "plotter"

        monkeypatch.setattr("complexplorer.plotting.pyvista.riemann.riemann_pv", fake_riemann_pv)
        quick_plot(_f, mode="riemann")
        assert "domain" not in captured  # default full-sphere, no mask


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


class TestCuratedSurface:
    """3.0 curation (curate-high-level-api): no stubs, no aliases on the surface."""

    def test_removed_api_names_not_importable(self):
        import complexplorer.api as api

        for name in ("create_animation", "compare_functions", "analyze_function"):
            assert not hasattr(api, name)

    def test_removed_top_level_names_not_importable(self):
        import complexplorer as cp

        for name in ("visualize", "explore", "analyze_function"):
            assert not hasattr(cp, name)

    def test_top_level_all_is_curated(self):
        import complexplorer as cp

        assert "quick_plot" in cp.__all__
        assert "Presets" in cp.__all__
        for name in ("visualize", "explore", "analyze_function"):
            assert name not in cp.__all__
        # __all__ must be importable in full
        for name in cp.__all__:
            assert hasattr(cp, name)
