"""Tests for PyVista 3D plotting functions.

The landscape renderers are exercised off-screen against the real PyVista backend (required
as of 3.0) instead of mocking ``pyvista.Plotter``.
"""

import numpy as np
import pytest

pyvista = pytest.importorskip("pyvista")

from complexplorer.core.colormap import Chessboard, Phase
from complexplorer.core.domain import Disk, Rectangle
from complexplorer.exceptions import ValidationError
from complexplorer.plotting.pyvista.plot_3d import (
    create_complex_surface,
    pair_plot_landscape_pv,
    plot_landscape_pv,
)
from complexplorer.plotting.pyvista.utils import ensure_pyvista_setup


class TestCreateComplexSurface:
    """Test the mesh creation utilities."""

    def test_basic_surface_creation(self):
        """Test basic surface creation with domain and function."""
        domain = Rectangle(2, 2)
        func = lambda z: z**2

        grid, rgb = create_complex_surface(domain, func, resolution=50)

        assert isinstance(grid, pyvista.StructuredGrid)
        assert grid.n_points == 50 * 50
        assert "RGB" in grid.array_names
        assert "magnitude" in grid.array_names
        assert "phase" in grid.array_names
        assert rgb.shape == (50, 50, 3)

    def test_surface_with_arrays(self):
        """Test surface creation with pre-computed arrays."""
        x = np.linspace(-1, 1, 30)
        y = np.linspace(-1, 1, 30)
        X, Y = np.meshgrid(x, y)
        z = X + 1j * Y
        f = z**2

        grid, rgb = create_complex_surface(None, None, z=z, f=f)

        assert grid.n_points == 30 * 30
        assert np.allclose(grid["magnitude"], np.abs(f).ravel())

    def test_custom_colormap(self):
        """Test surface creation with custom colormap."""
        domain = Rectangle(2, 2)
        func = lambda z: z**2
        cmap = Chessboard(spacing=0.5)

        grid, rgb = create_complex_surface(domain, func, cmap=cmap, resolution=40)

        assert grid.n_points == 40 * 40
        assert rgb.shape == (40, 40, 3)

    def test_missing_inputs_raise(self):
        with pytest.raises(ValidationError):
            create_complex_surface(None, None, resolution=20)


class TestPlotLandscapePV:
    """Real off-screen rendering of the 3D landscape."""

    def test_returns_plotter_when_requested(self):
        p = plot_landscape_pv(
            Rectangle(2, 2), lambda z: z**2, resolution=30, interactive=False, return_plotter=True
        )
        assert isinstance(p, pyvista.Plotter)
        p.close()

    def test_returns_none_without_return_plotter(self):
        result = plot_landscape_pv(
            Rectangle(2, 2), lambda z: z**2, resolution=30, interactive=False
        )
        assert result is None

    def test_with_colormap(self):
        p = plot_landscape_pv(
            Disk(2),
            lambda z: (z - 1) / (z + 1),
            cmap=Phase(n_phi=6, auto_scale_r=True),
            resolution=40,
            interactive=False,
            return_plotter=True,
        )
        assert isinstance(p, pyvista.Plotter)
        p.close()

    def test_save_to_file(self, tmp_path):
        out = tmp_path / "landscape.png"
        plot_landscape_pv(
            Rectangle(1, 1), lambda z: z, resolution=30, interactive=False, filename=str(out)
        )
        assert out.exists()
        assert out.stat().st_size > 0

    def test_custom_title(self):
        p = plot_landscape_pv(
            Rectangle(2, 2),
            lambda z: z**3 - 1,
            title="Cubic",
            resolution=20,
            interactive=False,
            return_plotter=True,
        )
        assert isinstance(p, pyvista.Plotter)
        p.close()

    @pytest.mark.parametrize("bad", [{"n_theta": 30}, {"show": False}, {"bogus": 1}])
    def test_unknown_kwargs_rejected(self, bad):
        with pytest.raises(ValidationError):
            plot_landscape_pv(
                Rectangle(2, 2), lambda z: z**2, resolution=20, interactive=False, **bad
            )


class TestPairPlotLandscapePV:
    """Real off-screen rendering of the paired landscape."""

    def test_returns_plotter(self):
        p = pair_plot_landscape_pv(
            Rectangle(2, 2), lambda z: z**2, resolution=30, interactive=False, return_plotter=True
        )
        assert isinstance(p, pyvista.Plotter)
        p.close()

    def test_title_renders(self):
        # The title is applied as a figure-level annotation; it must not raise.
        p = pair_plot_landscape_pv(
            Rectangle(2, 2),
            lambda z: np.sin(z),
            title="Sine",
            resolution=20,
            interactive=False,
            return_plotter=True,
        )
        assert isinstance(p, pyvista.Plotter)
        p.close()

    def test_save_pair_plot(self, tmp_path):
        out = tmp_path / "pair.png"
        pair_plot_landscape_pv(
            Rectangle(1, 1), lambda z: z**2 - 1, resolution=20, interactive=False, filename=str(out)
        )
        assert out.exists()
        assert out.stat().st_size > 0

    def test_unknown_kwargs_rejected(self):
        with pytest.raises(ValidationError):
            pair_plot_landscape_pv(
                Rectangle(2, 2), lambda z: z**2, resolution=20, interactive=False, show=False
            )


class TestPyVistaSetup:
    """Test PyVista setup utilities against the real global theme."""

    def test_ensure_setup_configures_theme(self):
        ensure_pyvista_setup()
        assert pyvista.global_theme.smooth_shading is True
        assert pyvista.global_theme.multi_samples is not None
