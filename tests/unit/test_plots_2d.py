"""
Unit tests for complexplorer 2D plotting functions.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from complexplorer import (
    Rectangle, Disk, Phase, Chessboard,
    plot, pair_plot, riemann_chart, riemann_hemispheres
)


class TestPlotFunction:
    """Test the plot() function."""
    
    def test_plot_basic(self):
        """Test basic plot functionality."""
        domain = Rectangle(4, 4)
        func = lambda z: z**2
        
        # Should not raise any errors
        plot(domain, func, n=50)
        plt.close()
    
    def test_plot_with_custom_cmap(self):
        """Test plot with different color maps."""
        domain = Disk(2)
        func = lambda z: (z - 1) / (z**2 + z + 1)
        
        # Test with different color maps
        cmaps = [
            Phase(6),
            Phase(12, enhance_phase=True),
            Chessboard(period=0.5),
        ]
        
        for cmap in cmaps:
            plot(domain, func, cmap=cmap, n=50)
            plt.close()
    
    def test_plot_with_arrays(self):
        """Test plot with pre-computed arrays."""
        x = np.linspace(-2, 2, 50)
        y = np.linspace(-2, 2, 50)
        X, Y = np.meshgrid(x, y)
        z = X + 1j * Y
        f = z**2
        
        plot(z=z, f=f)
        plt.close()
    
    def test_plot_with_title_and_filename(self):
        """Test plot with title and filename."""
        domain = Rectangle(2, 2)
        func = lambda z: np.sin(z)
        
        # Create plot with title
        plot(domain, func, title="Test Plot", n=30)
        
        # Get current figure and check title
        fig = plt.gcf()
        assert fig._suptitle is not None
        assert fig._suptitle.get_text() == "Test Plot"
        plt.close()
    
    @pytest.mark.parametrize("figsize", [(8, 8), (10, 6), (5, 5)])
    def test_plot_figsize(self, figsize):
        """Test plot with different figure sizes."""
        domain = Rectangle(2, 2)
        func = lambda z: z
        
        plot(domain, func, figsize=figsize, n=30)
        
        fig = plt.gcf()
        assert fig.get_figwidth() == figsize[0]
        assert fig.get_figheight() == figsize[1]
        plt.close()


class TestPairPlot:
    """Test the pair_plot() function."""
    
    def test_pair_plot_basic(self):
        """Test basic pair_plot functionality."""
        domain = Rectangle(4, 4)
        func = lambda z: z**2
        
        pair_plot(domain, func, n=50)
        
        # Should create figure with 2 subplots
        fig = plt.gcf()
        assert len(fig.axes) == 2
        plt.close()
    
    def test_pair_plot_titles(self):
        """Test pair_plot with titles."""
        domain = Disk(2)
        func = lambda z: 1/z
        
        pair_plot(domain, func, title="Test Pair Plot", n=30)
        
        fig = plt.gcf()
        assert fig._suptitle is not None
        assert fig._suptitle.get_text() == "Test Pair Plot"
        
        # Check subplot titles
        assert fig.axes[0].get_title() == "z"
        assert fig.axes[1].get_title() == "f(z)"
        plt.close()
    
    def test_pair_plot_with_arrays(self):
        """Test pair_plot with pre-computed arrays."""
        z = np.linspace(-1, 1, 30) + 1j * np.linspace(-1, 1, 30)[:, np.newaxis]
        f = z**3 - 1
        
        pair_plot(z=z, f=f)
        
        fig = plt.gcf()
        assert len(fig.axes) == 2
        plt.close()


class TestRiemannChart:
    """Test the riemann_chart() function."""
    
    def test_riemann_chart_basic(self):
        """Test basic riemann_chart functionality."""
        func = lambda z: (z - 1) / (z + 1)
        
        riemann_chart(func, n=40)
        
        # Should create figure with single axis
        fig = plt.gcf()
        assert len(fig.axes) == 1
        plt.close()
    
    def test_riemann_chart_coloring(self):
        """Test riemann_chart with different color maps."""
        func = lambda z: z**2
        
        # Test with enhanced phase portrait
        cmap = Phase(12, enhance_phase=True, enhance_modulus=True)
        riemann_chart(func, cmap=cmap, n=30)
        plt.close()
    
    def test_riemann_chart_title(self):
        """Test riemann_chart with title."""
        func = lambda z: np.exp(z)
        
        riemann_chart(func, title="Exponential on Riemann Sphere", n=30)
        
        fig = plt.gcf()
        ax = fig.axes[0]
        assert ax.get_title() == "Exponential on Riemann Sphere"
        plt.close()


class TestRiemannHemispheres:
    """Test the riemann_hemispheres() function."""
    
    def test_riemann_hemispheres_basic(self):
        """Test basic riemann_hemispheres functionality."""
        func = lambda z: z**3 - 1
        
        riemann_hemispheres(func, n=40)
        
        # Should create figure with 2 subplots
        fig = plt.gcf()
        assert len(fig.axes) == 2
        plt.close()
    
    def test_riemann_hemispheres_titles(self):
        """Test riemann_hemispheres with title."""
        func = lambda z: (z**2 - 1) / (z**2 + 1)
        
        riemann_hemispheres(func, title="Rational Function", n=30)
        
        fig = plt.gcf()
        assert fig._suptitle is not None
        assert fig._suptitle.get_text() == "Rational Function"
        
        # Check subplot titles
        assert "Upper" in fig.axes[0].get_title()
        assert "Lower" in fig.axes[1].get_title()
        plt.close()
    
    def test_riemann_hemispheres_different_cmaps(self):
        """Test riemann_hemispheres with different color maps."""
        func = lambda z: np.sin(z)
        
        # Test with basic phase portrait
        cmap1 = Phase(6, enhance_phase=False)
        riemann_hemispheres(func, cmap=cmap1, n=30)
        plt.close()
        
        # Test with enhanced phase portrait
        cmap2 = Phase(12, enhance_phase=True, n_modulus=6, enhance_modulus=True)
        riemann_hemispheres(func, cmap=cmap2, n=30)
        plt.close()


class TestPlottingEdgeCases:
    """Test edge cases for plotting functions."""
    
    def test_plot_with_singularities(self):
        """Test plotting functions with singularities."""
        domain = Rectangle(4, 4)
        
        # Function with pole at origin
        func = lambda z: 1/z if z != 0 else np.inf
        plot(domain, func, n=50)
        plt.close()
        
        # Function with essential singularity
        func = lambda z: np.exp(1/z) if z != 0 else np.inf
        plot(domain, func, n=50)
        plt.close()
    
    def test_plot_with_branch_cuts(self):
        """Test plotting functions with branch cuts."""
        domain = Rectangle(4, 4)
        
        # Square root has branch cut along negative real axis
        func = lambda z: np.sqrt(z)
        plot(domain, func, n=50)
        plt.close()
        
        # Logarithm has branch cut along negative real axis
        func = lambda z: np.log(z)
        plot(domain, func, n=50)
        plt.close()
    
    def test_plot_empty_domain(self):
        """Test plotting with domain that excludes all points."""
        # Create impossible domain
        domain = Domain(
            infunc=lambda z: False,  # No points satisfy this
            window_real=(-1, 1),
            window_imag=(-1, 1)
        )
        
        func = lambda z: z
        
        # Should handle gracefully
        plot(domain, func, n=20)
        plt.close()
    
    def test_riemann_chart_with_constant_function(self):
        """Test riemann_chart with constant function."""
        func = lambda z: 1 + 2j  # Constant function
        
        riemann_chart(func, n=30)
        
        # Should create valid plot even though function is constant
        fig = plt.gcf()
        assert len(fig.axes) == 1
        plt.close()


class TestPlotSaving:
    """Test saving plots to files."""
    
    def test_plot_save_to_file(self, tmp_path):
        """Test saving plot to file."""
        domain = Rectangle(2, 2)
        func = lambda z: z**2
        
        filename = tmp_path / "test_plot.png"
        plot(domain, func, filename=str(filename), n=30)
        
        # Check file was created
        assert filename.exists()
        assert filename.stat().st_size > 0
        plt.close()
    
    def test_pair_plot_save_to_file(self, tmp_path):
        """Test saving pair_plot to file."""
        domain = Disk(1.5)
        func = lambda z: z**3 - z
        
        filename = tmp_path / "test_pair_plot.png"
        pair_plot(domain, func, filename=str(filename), n=30)
        
        assert filename.exists()
        assert filename.stat().st_size > 0
        plt.close()