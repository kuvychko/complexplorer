"""
Unit tests for complexplorer 3D plotting functions.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from complexplorer import (
    Rectangle, Disk, Phase, Chessboard,
    plot_landscape, pair_plot_landscape, riemann
)


class TestPlotLandscape:
    """Test the plot_landscape() function."""
    
    def test_plot_landscape_basic(self):
        """Test basic plot_landscape functionality."""
        domain = Rectangle(4, 4)
        func = lambda z: z**2
        
        # Should create 3D plot without errors
        ax = plot_landscape(domain, func, n=30)
        
        # Check that we have a 3D axes
        assert ax is not None
        assert hasattr(ax, 'plot_surface')
        plt.close()
    
    def test_plot_landscape_with_cmap(self):
        """Test plot_landscape with different color maps."""
        domain = Disk(2)
        func = lambda z: (z - 1) / (z**2 + z + 1)
        
        cmaps = [
            Phase(6),
            Phase(12, enhance_phase=True),
            Chessboard(period=0.5),
        ]
        
        for cmap in cmaps:
            ax = plot_landscape(domain, func, cmap=cmap, n=30)
            assert ax is not None
            plt.close()
    
    def test_plot_landscape_logarithmic_z(self):
        """Test plot_landscape with logarithmic z-axis."""
        domain = Rectangle(4, 4)
        func = lambda z: np.exp(z)
        
        # Test with logarithmic z-axis
        ax = plot_landscape(domain, func, zaxis_log=True, n=30)
        assert ax is not None
        plt.close()
    
    def test_plot_landscape_z_max(self):
        """Test plot_landscape with z_max clipping."""
        domain = Rectangle(2, 2)
        func = lambda z: z**4  # Can get large values
        
        # Test with z_max
        ax = plot_landscape(domain, func, z_max=10, n=30)
        assert ax is not None
        plt.close()
        
        # Test invalid z_max
        with pytest.raises(ValueError, match="z_max must be positive"):
            plot_landscape(domain, func, z_max=-1)
    
    def test_plot_landscape_with_arrays(self):
        """Test plot_landscape with pre-computed arrays."""
        x = np.linspace(-2, 2, 30)
        y = np.linspace(-2, 2, 30)
        X, Y = np.meshgrid(x, y)
        z = X + 1j * Y
        f = z**2 - 1
        
        ax = plot_landscape(z=z, f=f)
        assert ax is not None
        plt.close()
    
    def test_plot_landscape_antialiased(self):
        """Test plot_landscape with antialiasing."""
        domain = Rectangle(2, 2)
        func = lambda z: np.sin(z)
        
        # Test with antialiasing on
        ax = plot_landscape(domain, func, antialiased=True, n=20)
        assert ax is not None
        plt.close()
    
    def test_plot_landscape_with_existing_axes(self):
        """Test plot_landscape with existing 3D axes."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        domain = Rectangle(2, 2)
        func = lambda z: z**2
        
        returned_ax = plot_landscape(domain, func, ax=ax, n=20)
        
        # Should return the same axes
        assert returned_ax is ax
        plt.close()


class TestPairPlotLandscape:
    """Test the pair_plot_landscape() function."""
    
    def test_pair_plot_landscape_basic(self):
        """Test basic pair_plot_landscape functionality."""
        domain = Rectangle(4, 4)
        func = lambda z: z**2
        
        pair_plot_landscape(domain, func, n=30)
        
        # Should create figure with 2 3D subplots
        fig = plt.gcf()
        assert len(fig.axes) == 2
        assert all(hasattr(ax, 'plot_surface') for ax in fig.axes)
        plt.close()
    
    def test_pair_plot_landscape_with_options(self):
        """Test pair_plot_landscape with various options."""
        domain = Disk(2)
        func = lambda z: 1 / (z**2 + 1)
        
        # Test with custom color map and log z-axis
        cmap = Phase(12, enhance_phase=True)
        pair_plot_landscape(
            domain, func, 
            cmap=cmap, 
            zaxis_log=True,
            z_max=5,
            n=30
        )
        
        fig = plt.gcf()
        assert len(fig.axes) == 2
        plt.close()
    
    def test_pair_plot_landscape_figsize(self):
        """Test pair_plot_landscape with custom figure size."""
        domain = Rectangle(2, 2)
        func = lambda z: z**3 - z
        
        pair_plot_landscape(domain, func, figsize=(12, 6), n=20)
        
        fig = plt.gcf()
        assert fig.get_figwidth() == 12
        assert fig.get_figheight() == 6
        plt.close()
    
    def test_pair_plot_landscape_title(self):
        """Test pair_plot_landscape with title."""
        domain = Rectangle(3, 3)
        func = lambda z: np.exp(z)
        
        pair_plot_landscape(domain, func, title="Exponential Function", n=20)
        
        fig = plt.gcf()
        assert fig._suptitle is not None
        assert fig._suptitle.get_text() == "Exponential Function"
        plt.close()


class TestRiemann:
    """Test the riemann() function."""
    
    def test_riemann_basic(self):
        """Test basic riemann functionality."""
        func = lambda z: (z - 1) / (z + 1)
        
        riemann(func, n=30)
        
        # Should create 3D plot
        fig = plt.gcf()
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        assert hasattr(ax, 'plot_surface')
        plt.close()
    
    def test_riemann_with_cmap(self):
        """Test riemann with different color maps."""
        func = lambda z: z**2
        
        # Test with enhanced phase portrait
        cmap = Phase(12, enhance_phase=True, enhance_modulus=True)
        riemann(func, cmap=cmap, n=30)
        plt.close()
        
        # Test with basic phase portrait
        cmap = Phase(6, modulus_saturation=0.8)
        riemann(func, cmap=cmap, n=30)
        plt.close()
    
    def test_riemann_projection_direction(self):
        """Test riemann with different projection directions."""
        func = lambda z: z**3 - 1
        
        # Default projection (from south)
        riemann(func, project_from_north=False, n=30)
        plt.close()
        
        # Projection from north
        riemann(func, project_from_north=True, n=30)
        plt.close()
    
    def test_riemann_with_title(self):
        """Test riemann with title."""
        func = lambda z: np.sin(z)
        
        riemann(func, title="Sine on Riemann Sphere", n=30)
        
        fig = plt.gcf()
        ax = fig.axes[0]
        assert ax.get_title() == "Sine on Riemann Sphere"
        plt.close()
    
    def test_riemann_with_existing_axes(self):
        """Test riemann with existing 3D axes."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        func = lambda z: 1 / (z**2 + 1)
        
        returned_ax = riemann(func, ax=ax, n=30)
        
        # Should return the same axes
        assert returned_ax is ax
        plt.close()


class Test3DPlottingEdgeCases:
    """Test edge cases for 3D plotting functions."""
    
    def test_plot_landscape_with_singularities(self):
        """Test 3D plotting of functions with singularities."""
        domain = Rectangle(4, 4)
        
        # Function with poles
        func = lambda z: 1 / (z**2 - 1)
        ax = plot_landscape(domain, func, n=30)
        assert ax is not None
        plt.close()
        
        # Function with essential singularity
        func = lambda z: np.exp(1/z) if z != 0 else np.inf
        ax = plot_landscape(domain, func, zaxis_log=True, n=30)
        assert ax is not None
        plt.close()
    
    def test_plot_landscape_constant_function(self):
        """Test 3D plotting of constant function."""
        domain = Rectangle(2, 2)
        func = lambda z: 2 + 3j  # Constant
        
        ax = plot_landscape(domain, func, n=20)
        assert ax is not None
        
        # Z-coordinate should be constant (|2+3j| = sqrt(13))
        expected_height = np.sqrt(13)
        # Note: Can't easily check the actual surface data from ax
        plt.close()
    
    def test_riemann_with_poles_at_infinity(self):
        """Test riemann sphere with functions having poles at infinity."""
        # Polynomial has pole at infinity
        func = lambda z: z**5
        
        riemann(func, n=30)
        plt.close()
        
        # Rational function with pole at infinity
        func = lambda z: (z**3 + 1) / z if z != 0 else np.inf
        riemann(func, n=30)
        plt.close()
    
    def test_3d_plots_with_nan_values(self):
        """Test 3D plots handling NaN values gracefully."""
        domain = Rectangle(2, 2)
        
        # Function that returns NaN for some inputs
        def func(z):
            result = np.log(z)  # NaN for negative real values
            return result
        
        # Should handle NaN values without crashing
        ax = plot_landscape(domain, func, n=30)
        assert ax is not None
        plt.close()
        
        # Also test with riemann
        riemann(func, n=30)
        plt.close()


class Test3DPlotSaving:
    """Test saving 3D plots to files."""
    
    def test_plot_landscape_save(self, tmp_path):
        """Test saving plot_landscape to file."""
        domain = Rectangle(2, 2)
        func = lambda z: z**2
        
        filename = tmp_path / "landscape.png"
        ax = plot_landscape(domain, func, n=20)
        plt.savefig(filename)
        
        assert filename.exists()
        assert filename.stat().st_size > 0
        plt.close()
    
    def test_pair_plot_landscape_save(self, tmp_path):
        """Test saving pair_plot_landscape to file."""
        domain = Disk(1.5)
        func = lambda z: z**3 - z
        
        filename = tmp_path / "pair_landscape.png"
        pair_plot_landscape(domain, func, filename=str(filename), n=20)
        
        assert filename.exists()
        assert filename.stat().st_size > 0
        plt.close()
    
    def test_riemann_save(self, tmp_path):
        """Test saving riemann plot to file."""
        func = lambda z: (z**2 - 1) / (z**2 + 1)
        
        filename = tmp_path / "riemann.png"
        riemann(func, filename=str(filename), n=30)
        
        assert filename.exists()
        assert filename.stat().st_size > 0
        plt.close()


class TestMemoryAndPerformance:
    """Test memory and performance aspects of 3D plotting."""
    
    def test_large_mesh_handling(self):
        """Test that large meshes are handled appropriately."""
        domain = Rectangle(10, 10)
        func = lambda z: z
        
        # Test with moderately large mesh
        # Should complete without memory errors
        ax = plot_landscape(domain, func, n=100)
        assert ax is not None
        plt.close()
    
    def test_multiple_3d_plots(self):
        """Test creating multiple 3D plots in sequence."""
        domain = Rectangle(2, 2)
        funcs = [
            lambda z: z,
            lambda z: z**2,
            lambda z: z**3,
        ]
        
        # Create multiple plots and ensure proper cleanup
        for func in funcs:
            ax = plot_landscape(domain, func, n=20)
            assert ax is not None
            plt.close()
        
        # All figures should be closed
        assert len(plt.get_fignums()) == 0