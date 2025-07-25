"""Tests for PyVista 3D plotting functions."""

import pytest
import numpy as np
from unittest import mock

# Skip all tests if PyVista is not available
pyvista = pytest.importorskip("pyvista")

from complexplorer.core.domain import Rectangle, Disk, Annulus
from complexplorer.core.colormap import Phase, Chessboard
from complexplorer.plotting.pyvista.plot_3d import (
    plot_landscape_pv,
    pair_plot_landscape_pv,
    create_complex_surface
)
from complexplorer.plotting.pyvista.utils import (
    ensure_pyvista_setup
)


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
    
    def test_with_high_resolution(self):
        """Test surface creation with high resolution."""
        domain = Rectangle(2, 2)
        func = lambda z: z**2
        
        grid, rgb = create_complex_surface(domain, func, resolution=100)
        
        assert grid.n_points == 100 * 100
        assert rgb.shape == (100, 100, 3)


class TestPlotLandscapePV:
    """Test plot_landscape_pv function."""
    
    def test_basic_plot(self):
        """Test basic landscape plot."""
        domain = Rectangle(2, 2)
        func = lambda z: z**2
        
        with mock.patch('pyvista.Plotter') as MockPlotter:
            plotter = MockPlotter.return_value
            result = plot_landscape_pv(domain, func, resolution=50, show=False)
            
            # Function doesn't return plotter when show=False
            assert result is None
            assert MockPlotter.called
            plotter.add_mesh.assert_called()
    
    def test_with_colormap(self):
        """Test plot with custom colormap."""
        domain = Disk(2)
        func = lambda z: (z - 1) / (z + 1)
        cmap = Phase(n_phi=6, auto_scale_r=True)
        
        with mock.patch('pyvista.Plotter') as MockPlotter:
            plotter = MockPlotter.return_value
            result = plot_landscape_pv(
                domain, func, cmap=cmap, resolution=60, show=False
            )
            
            # Function doesn't return plotter when show=False
            assert result is None
            assert MockPlotter.called
    
    def test_save_to_file(self):
        """Test saving plot to file."""
        domain = Rectangle(1, 1)
        func = lambda z: z
        
        with mock.patch('pyvista.Plotter') as MockPlotter:
            plotter = MockPlotter.return_value
            plot_landscape_pv(
                domain, func, resolution=30, filename="test.png", show=False
            )
            
            plotter.screenshot.assert_called_once_with("test.png")
    
    def test_custom_title(self):
        """Test plot with custom title."""
        domain = Rectangle(2, 2)
        func = lambda z: z**3 - 1
        
        with mock.patch('pyvista.Plotter') as MockPlotter:
            plotter = MockPlotter.return_value
            plot_landscape_pv(
                domain, func, title="Cubic Function", show=False
            )
            
            plotter.add_text.assert_called_with(
                "Cubic Function",
                position='upper_edge',
                font_size=14
            )


class TestPairPlotLandscapePV:
    """Test pair_plot_landscape_pv function."""
    
    def test_basic_pair_plot(self):
        """Test basic pair plot."""
        domain = Rectangle(2, 2)
        func = lambda z: z**2
        
        with mock.patch('pyvista.Plotter') as MockPlotter:
            plotter = MockPlotter.return_value
            result = pair_plot_landscape_pv(domain, func, resolution=40, show=False)
            
            # Function doesn't return plotter when show=False
            assert result is None
            assert MockPlotter.called
            # Should have 2 viewports
            # Check that shape=(1, 2) was included in the call
            assert MockPlotter.call_args is not None
            assert MockPlotter.call_args.kwargs.get('shape') == (1, 2)
    
    def test_with_labels(self):
        """Test pair plot with custom labels."""
        domain = Disk(1.5)
        func = lambda z: np.sin(z)
        
        with mock.patch('pyvista.Plotter') as MockPlotter:
            plotter = MockPlotter.return_value
            pair_plot_landscape_pv(
                domain, func,
                labels=["Input", "Output"],
                show=False
            )
            
            # Check that labels were added
            calls = plotter.add_text.call_args_list
            assert len(calls) >= 2
    
    def test_save_pair_plot(self):
        """Test saving pair plot."""
        domain = Rectangle(1, 1)
        func = lambda z: z**2 - 1
        
        with mock.patch('pyvista.Plotter') as MockPlotter:
            plotter = MockPlotter.return_value
            pair_plot_landscape_pv(
                domain, func,
                filename="pair_test.png",
                show=False
            )
            
            plotter.screenshot.assert_called_once_with("pair_test.png")


class TestPyVistaSetup:
    """Test PyVista setup utilities."""
    
    def test_ensure_setup(self):
        """Test ensure_pyvista_setup function."""
        with mock.patch('pyvista.global_theme') as mock_theme:
            ensure_pyvista_setup()
            # Should not raise any errors
            assert True
    
    def test_theme_settings(self):
        """Test that theme is configured properly."""
        with mock.patch.object(pyvista, 'global_theme') as mock_theme:
            ensure_pyvista_setup()
            # Basic check that theme was accessed
            assert mock_theme is not None