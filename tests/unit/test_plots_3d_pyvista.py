"""
Unit tests for PyVista-based 3D plotting functions.
"""

import pytest
import numpy as np
import sys

# Skip all tests if PyVista is not available
pyvista = pytest.importorskip("pyvista")

from complexplorer import Rectangle, Disk, Annulus, Phase, Chessboard
from complexplorer.plots_3d_pyvista import (
    plot_landscape_pv, 
    pair_plot_landscape_pv,
    riemann_pv,
    _create_complex_surface
)


class TestCreateComplexSurface:
    """Test the internal mesh creation function."""
    
    def test_basic_surface_creation(self):
        """Test basic surface creation with domain and function."""
        domain = Rectangle(2, 2)
        func = lambda z: z**2
        
        grid, rgb = _create_complex_surface(domain, func, n=50)
        
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
        
        grid, rgb = _create_complex_surface(None, None, z=z, f=f)
        
        assert grid.n_points == 30 * 30
        assert np.allclose(grid["magnitude"], np.abs(f).ravel())
    
    def test_constant_function(self):
        """Test handling of constant functions."""
        domain = Rectangle(2, 2)
        func = lambda z: 1 + 2j
        
        grid, rgb = _create_complex_surface(domain, func, n=20)
        
        # All magnitudes should be sqrt(5)
        expected_mag = np.sqrt(5)
        assert np.allclose(grid["magnitude"], expected_mag)
    
    def test_logarithmic_scaling(self):
        """Test logarithmic height scaling."""
        domain = Rectangle(2, 2)
        func = lambda z: z
        
        grid_linear, _ = _create_complex_surface(domain, func, n=20, log_z=False)
        grid_log, _ = _create_complex_surface(domain, func, n=20, log_z=True)
        
        # Heights should be different
        z_linear = grid_linear.points[:, 2]
        z_log = grid_log.points[:, 2]
        assert not np.allclose(z_linear, z_log)
    
    def test_masked_domain(self):
        """Test handling of domains with masks (e.g., annulus)."""
        domain = Annulus(0.5, 2)
        func = lambda z: 1/z
        
        grid, rgb = _create_complex_surface(domain, func, n=40)
        
        # Check for NaN values in masked regions
        assert np.any(np.isnan(grid.points[:, 2]))


class TestPlotLandscapePv:
    """Test the main plot_landscape_pv function."""
    
    @pytest.fixture
    def simple_setup(self):
        """Simple domain and function for testing."""
        domain = Rectangle(2, 2)
        func = lambda z: z
        return domain, func
    
    def test_basic_plot_noninteractive(self, simple_setup):
        """Test basic non-interactive plotting."""
        domain, func = simple_setup
        
        # Should not raise any errors
        plot_landscape_pv(domain, func, n=20, interactive=False)
    
    def test_plot_with_edges(self, simple_setup):
        """Test plotting with edges shown."""
        domain, func = simple_setup
        
        plot_landscape_pv(
            domain, func, n=20, 
            interactive=False,
            show_edges=True,
            edge_color='black'
        )
    
    def test_plot_with_custom_colormap(self, simple_setup):
        """Test plotting with custom colormap."""
        domain, func = simple_setup
        cmap = Chessboard(spacing=0.5)
        
        plot_landscape_pv(
            domain, func, n=20,
            interactive=False,
            cmap=cmap
        )
    
    def test_plot_with_file_output(self, simple_setup, tmp_path):
        """Test saving plot to file."""
        domain, func = simple_setup
        filename = tmp_path / "test_plot.png"
        
        plot_landscape_pv(
            domain, func, n=20,
            interactive=False,
            filename=str(filename)
        )
        
        assert filename.exists()
    
    def test_orientation_axes(self, simple_setup):
        """Test orientation axes widget display."""
        domain, func = simple_setup
        
        # Test with orientation axes enabled (default)
        plot_landscape_pv(
            domain, func, n=20,
            interactive=False,
            show_orientation=True
        )
        
        # Test with orientation axes disabled
        plot_landscape_pv(
            domain, func, n=20,
            interactive=False,
            show_orientation=False
        )
    
    def test_return_plotter(self, simple_setup):
        """Test returning plotter object."""
        domain, func = simple_setup
        
        plotter = plot_landscape_pv(
            domain, func, n=20,
            interactive=False,
            return_plotter=True
        )
        
        assert isinstance(plotter, pyvista.Plotter)
        plotter.close()
    
    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        # No domain or z
        with pytest.raises(ValueError, match="both domain and z parameters cannot be None"):
            plot_landscape_pv(None, lambda z: z)
        
        # No function or f
        domain = Rectangle(2, 2)
        with pytest.raises(ValueError, match="both f and func parameters cannot be None"):
            plot_landscape_pv(domain, None)


class TestPairPlotLandscapePv:
    """Test the pair_plot_landscape_pv function."""
    
    def test_basic_pair_plot(self):
        """Test basic pair plot functionality."""
        domain = Rectangle(2, 2)
        func = lambda z: z**2
        
        pair_plot_landscape_pv(
            domain, func, n=20,
            interactive=False
        )
    
    def test_pair_plot_orientation_axes(self):
        """Test pair plot with orientation axes."""
        domain = Rectangle(2, 2)
        func = lambda z: z**2
        
        # Test with orientation axes enabled
        pair_plot_landscape_pv(
            domain, func, n=20,
            interactive=False,
            show_orientation=True
        )
        
        # Test with orientation axes disabled
        pair_plot_landscape_pv(
            domain, func, n=20,
            interactive=False,
            show_orientation=False
        )
    
    def test_pair_plot_with_linking(self):
        """Test pair plot with linked views."""
        domain = Disk(2)
        func = lambda z: (z - 1) / (z + 1)
        
        pair_plot_landscape_pv(
            domain, func, n=30,
            interactive=False,
            link_views=True
        )
    
    def test_pair_plot_file_output(self, tmp_path):
        """Test saving pair plot to file."""
        domain = Rectangle(3, 3)
        func = lambda z: np.sin(z)
        filename = tmp_path / "pair_plot.png"
        
        pair_plot_landscape_pv(
            domain, func, n=20,
            interactive=False,
            filename=str(filename)
        )
        
        assert filename.exists()


class TestRiemannPv:
    """Test the riemann_pv function."""
    
    def test_basic_riemann_sphere(self):
        """Test basic Riemann sphere visualization."""
        func = lambda z: z
        
        # Should not raise any errors with default rectangular mesh
        riemann_pv(func, n_theta=50, n_phi=50, interactive=False)
    
    def test_constant_scaling(self):
        """Test traditional Riemann sphere with constant radius."""
        func = lambda z: (z - 1) / (z + 1)
        
        riemann_pv(
            func,
            n_theta=50,
            n_phi=50,
            scaling='constant',
            scaling_params={'radius': 1.5},
            interactive=False
        )
    
    def test_icosahedral_mesh(self):
        """Test Riemann sphere with icosahedral mesh."""
        func = lambda z: z**2
        
        riemann_pv(
            func,
            mesh_type='icosahedral',
            n_subdivisions=2,
            interactive=False
        )
    
    def test_uv_mesh(self):
        """Test Riemann sphere with UV mesh."""
        func = lambda z: 1 / z
        
        riemann_pv(
            func,
            mesh_type='uv',
            n_theta=50,
            n_phi=50,
            interactive=False
        )
    
    def test_arctan_scaling(self):
        """Test Riemann sphere with arctan modulus scaling."""
        func = lambda z: z**2
        
        riemann_pv(
            func,
            n_theta=50,
            n_phi=50,
            scaling='arctan',
            scaling_params={'r_min': 0.3, 'r_max': 1.2},
            interactive=False
        )
    
    def test_with_grid_lines(self):
        """Test Riemann sphere with latitude/longitude grid."""
        func = lambda z: 1 / z
        
        riemann_pv(
            func,
            n_theta=50,
            n_phi=50,
            show_grid=True,
            interactive=False
        )
    
    def test_file_output(self, tmp_path):
        """Test saving Riemann sphere to file."""
        func = lambda z: np.sin(z)
        filename = tmp_path / "riemann_sphere.png"
        
        riemann_pv(
            func,
            n_theta=50,
            n_phi=50,
            interactive=False,
            filename=str(filename)
        )
        
        assert filename.exists()
    
    def test_invalid_scaling(self):
        """Test error handling for invalid scaling method."""
        func = lambda z: z
        
        with pytest.raises(ValueError, match="Unknown scaling method"):
            riemann_pv(func, scaling='invalid', interactive=False)
    
    def test_orientation_axes(self):
        """Test orientation axes widget display."""
        func = lambda z: (z - 1) / (z + 1)
        
        # Test with orientation axes enabled (default)
        riemann_pv(
            func,
            n_theta=50,
            n_phi=50,
            show_orientation=True,
            interactive=False
        )
        
        # Test with orientation axes disabled
        riemann_pv(
            func,
            n_theta=50,
            n_phi=50,
            show_orientation=False,
            interactive=False
        )
    
    def test_quality_settings(self):
        """Test high quality and anti-aliasing settings."""
        func = lambda z: z**2
        
        # Test with quality settings disabled
        riemann_pv(
            func,
            n_theta=30,
            n_phi=30,
            high_quality=False,
            anti_aliasing=False,
            interactive=False
        )
        
        # Test with quality settings enabled
        riemann_pv(
            func,
            n_theta=30,
            n_phi=30,
            high_quality=True,
            anti_aliasing=True,
            interactive=False
        )


# Fixtures for PyVista configuration
@pytest.fixture(autouse=True)
def configure_pyvista():
    """Configure PyVista for testing."""
    # Use null display for CI environments
    if not hasattr(sys, 'ps1'):  # Not in interactive mode
        pyvista.OFF_SCREEN = True
    
    # Set theme
    pyvista.global_theme.window_size = [400, 300]
    
    yield
    
    # Cleanup
    pyvista.close_all()