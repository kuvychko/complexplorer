"""Tests for PyVista Riemann sphere plotting."""

import pytest
import numpy as np
from unittest import mock

# Skip all tests if PyVista is not available
pyvista = pytest.importorskip("pyvista")

from complexplorer.core.colormap import Phase, LogRings
from complexplorer.plotting.pyvista.riemann import riemann_pv


class TestRiemannPV:
    """Test riemann_pv function."""
    
    def test_basic_riemann_plot(self):
        """Test basic Riemann sphere plot."""
        func = lambda z: z**2
        
        with mock.patch('pyvista.Plotter') as MockPlotter:
            plotter = MockPlotter.return_value
            result = riemann_pv(func, n_theta=30, n_phi=30, show=False)
            
            # Function doesn't return plotter when show=False
            assert result is None
            assert MockPlotter.called
            plotter.add_mesh.assert_called()
    
    def test_with_custom_colormap(self):
        """Test Riemann sphere with custom colormap."""
        func = lambda z: (z - 1) / (z + 1)
        cmap = Phase(n_phi=12, auto_scale_r=True)
        
        with mock.patch('pyvista.Plotter') as MockPlotter:
            plotter = MockPlotter.return_value
            result = riemann_pv(
                func, cmap=cmap, n_theta=40, n_phi=40, show=False
            )
            
            # Function doesn't return plotter when show=False
            assert result is None
            assert MockPlotter.called
    
    def test_modulus_scaling_modes(self):
        """Test different modulus scaling modes."""
        func = lambda z: z**3 - z
        
        modes = ['constant', 'arctan', 'logarithmic', 'adaptive']
        
        for mode in modes:
            with mock.patch('pyvista.Plotter') as MockPlotter:
                plotter = MockPlotter.return_value
                riemann_pv(
                    func,
                    modulus_mode=mode,
                    n_theta=20,
                    n_phi=20,
                    show=False
                )
                
                plotter.add_mesh.assert_called()
    
    def test_custom_modulus_params(self):
        """Test custom modulus parameters."""
        func = lambda z: 1 / (z - 1)
        
        with mock.patch('pyvista.Plotter') as MockPlotter:
            plotter = MockPlotter.return_value
            riemann_pv(
                func,
                modulus_mode='arctan',
                modulus_params={'r_min': 0.3, 'r_max': 0.9},
                show=False
            )
            
            assert MockPlotter.called
    
    def test_save_riemann_plot(self):
        """Test saving Riemann sphere plot."""
        func = lambda z: z**2 + 1
        
        with mock.patch('pyvista.Plotter') as MockPlotter:
            plotter = MockPlotter.return_value
            riemann_pv(
                func,
                n_theta=25,
                n_phi=25,
                filename="riemann_test.png",
                show=False
            )
            
            plotter.screenshot.assert_called_once_with("riemann_test.png")
    
    def test_title_and_labels(self):
        """Test plot with title."""
        func = lambda z: np.sin(z)
        
        with mock.patch('pyvista.Plotter') as MockPlotter:
            plotter = MockPlotter.return_value
            riemann_pv(
                func,
                title="sin(z) on Riemann Sphere",
                show=False
            )
            
            plotter.add_text.assert_called_with(
                "sin(z) on Riemann Sphere",
                position='upper_edge',
                font_size=14
            )
    
    def test_logarithmic_rings_colormap(self):
        """Test with logarithmic rings colormap."""
        func = lambda z: z**2 / (z**2 + 1)
        cmap = LogRings()
        
        with mock.patch('pyvista.Plotter') as MockPlotter:
            plotter = MockPlotter.return_value
            riemann_pv(
                func,
                cmap=cmap,
                n_theta=30,
                n_phi=30,
                show=False
            )
            
            assert MockPlotter.called
    
    def test_essential_singularity(self):
        """Test function with essential singularity."""
        def func(z):
            # exp(1/z) with safety
            with np.errstate(divide='ignore', invalid='ignore'):
                result = np.exp(1/z)
            return np.where(np.isfinite(result), result, 0)
        
        with mock.patch('pyvista.Plotter') as MockPlotter:
            plotter = MockPlotter.return_value
            riemann_pv(
                func,
                modulus_mode='logarithmic',
                n_theta=20,
                n_phi=20,
                show=False
            )
            
            plotter.add_mesh.assert_called()
    
    def test_custom_resolution(self):
        """Test with different resolution settings."""
        func = lambda z: z**4 - 1
        
        # Test high resolution
        with mock.patch('pyvista.Plotter') as MockPlotter:
            riemann_pv(func, n_theta=100, n_phi=100, show=False)
            assert MockPlotter.called
        
        # Test low resolution
        with mock.patch('pyvista.Plotter') as MockPlotter:
            riemann_pv(func, n_theta=10, n_phi=10, show=False)
            assert MockPlotter.called
    
    def test_projection_from_south(self):
        """Test projection from south pole."""
        func = lambda z: z**2 - z + 1
        
        with mock.patch('pyvista.Plotter') as MockPlotter:
            plotter = MockPlotter.return_value
            riemann_pv(
                func,
                project_from_north=False,
                n_theta=30,
                n_phi=30,
                show=False
            )
            
            assert MockPlotter.called