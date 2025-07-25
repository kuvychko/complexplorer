"""Tests for 2D matplotlib plotting functions."""

import numpy as np
import pytest
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from complexplorer.plotting.matplotlib.plot_2d import (
    Matplotlib2DPlotter, plot, pair_plot, riemann_chart, riemann_hemispheres
)
from complexplorer.core.domain import Rectangle, Disk
from complexplorer.core.colormap import Phase, Chessboard
from complexplorer.utils.validation import ValidationError


class TestMatplotlib2DPlotter:
    """Test Matplotlib2DPlotter class."""
    
    def test_plot_single(self):
        """Test single plot generation."""
        plotter = Matplotlib2DPlotter()
        domain = Rectangle(4, 4)
        func = lambda z: z**2
        colormap = Phase(n_phi=6)
        
        ax = plotter.plot_single(domain, func, colormap, resolution=50)
        
        assert isinstance(ax, Axes)
        assert ax.get_xlabel() == "Re(z)"
        assert ax.get_ylabel() == "Im(z)"
        
        plt.close('all')
    
    def test_plot_single_with_ax(self):
        """Test plotting on provided axes."""
        plotter = Matplotlib2DPlotter()
        domain = Rectangle(2, 2)
        func = lambda z: z
        colormap = Chessboard()
        
        fig, ax = plt.subplots()
        result_ax = plotter.plot_single(domain, func, colormap, 
                                       resolution=30, ax=ax, title="Test")
        
        assert result_ax is ax
        assert ax.get_title() == "Test"
        
        plt.close('all')
    
    def test_plot_pair(self):
        """Test pair plot generation."""
        plotter = Matplotlib2DPlotter()
        domain = Rectangle(3, 3)
        func = lambda z: (z - 1) / (z + 1)
        colormap = Phase()
        
        fig = plotter.plot_pair(domain, func, colormap, resolution=40,
                               figsize=(8, 4), title="Möbius")
        
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 2
        assert fig._suptitle.get_text() == "Möbius"
        
        plt.close('all')


class TestPlotFunction:
    """Test the plot() convenience function."""
    
    def test_with_domain_and_func(self):
        """Test plotting with domain and function."""
        domain = Rectangle(2, 2)
        func = lambda z: z**3 - 1
        
        # Should not raise
        plot(domain=domain, func=func, n=50)
        plt.close('all')
    
    def test_with_arrays(self):
        """Test plotting with pre-computed arrays."""
        x = np.linspace(-1, 1, 50)
        y = np.linspace(-1, 1, 50)
        X, Y = np.meshgrid(x, y)
        z = X + 1j * Y
        f = z**2
        
        # Should not raise
        plot(z=z, f=f)
        plt.close('all')
    
    def test_missing_domain_and_z(self):
        """Test error when both domain and z are missing."""
        with pytest.raises(ValidationError, match="Either domain or z"):
            plot(func=lambda z: z)
    
    def test_missing_func_and_f(self):
        """Test error when both func and f are missing."""
        domain = Rectangle(2, 2)
        with pytest.raises(ValidationError, match="Either f or func"):
            plot(domain=domain)
    
    def test_with_custom_colormap(self):
        """Test with custom colormap."""
        domain = Disk(2)
        func = lambda z: np.exp(z)
        cmap = Chessboard(spacing=0.5)
        
        plot(domain=domain, func=func, cmap=cmap, n=60)
        plt.close('all')
    
    def test_with_axes(self):
        """Test plotting on provided axes."""
        domain = Rectangle(1, 1)
        func = lambda z: z
        
        fig, ax = plt.subplots()
        result = plot(domain=domain, func=func, ax=ax, title="Identity")
        
        assert result is ax
        assert ax.get_title() == "Identity"
        plt.close('all')
    
    def test_save_to_file(self, tmp_path):
        """Test saving to file."""
        domain = Rectangle(2, 2)
        func = lambda z: z**2
        filename = tmp_path / "test_plot.png"
        
        plot(domain=domain, func=func, filename=str(filename))
        
        assert filename.exists()
        plt.close('all')


class TestPairPlot:
    """Test the pair_plot() function."""
    
    def test_basic_pair_plot(self):
        """Test basic pair plot."""
        domain = Rectangle(3, 2)
        func = lambda z: (z**2 - 1) / (z**2 + 1)
        
        fig = pair_plot(domain=domain, func=func, n=40)
        
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 2
        assert "Domain" in fig.axes[0].get_title()
        assert "Codomain" in fig.axes[1].get_title()
        
        plt.close('all')
    
    def test_pair_plot_with_arrays(self):
        """Test pair plot with arrays."""
        z = Rectangle(2, 2).mesh(30)
        f = np.sin(z)
        
        fig = pair_plot(z=z, f=f, title="Sine function")
        
        assert fig._suptitle.get_text() == "Sine function"
        plt.close('all')
    
    def test_custom_figsize(self):
        """Test custom figure size."""
        domain = Disk(1.5)
        func = lambda z: z**3
        
        fig = pair_plot(domain=domain, func=func, figsize=(12, 6))
        
        assert fig.get_figwidth() == 12
        assert fig.get_figheight() == 6
        plt.close('all')


class TestRiemannChart:
    """Test riemann_chart() function."""
    
    def test_south_hemisphere(self):
        """Test south hemisphere visualization."""
        func = lambda z: (z - 1) / (z + 1)
        
        ax = riemann_chart(func, n=50, show_south_hemisphere=True)
        
        assert isinstance(ax, Axes)
        assert "Re(z)" in ax.get_xlabel()
        assert "Im(z)" in ax.get_ylabel()
        
        plt.close('all')
    
    def test_north_hemisphere(self):
        """Test north hemisphere visualization."""
        func = lambda z: z**2
        
        ax = riemann_chart(func, n=50, show_south_hemisphere=False)
        
        assert "Re(1/z)" in ax.get_xlabel()
        assert "Im(1/z)" in ax.get_ylabel()
        
        # Check tick labels
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert '∞' in labels
        
        plt.close('all')
    
    def test_margin_validation(self):
        """Test margin parameter validation."""
        func = lambda z: z
        
        with pytest.raises(ValidationError, match="non-negative"):
            riemann_chart(func, margin=-0.1)
        
        with pytest.raises(ValidationError, match="exceed 0.5"):
            riemann_chart(func, margin=0.6)
    
    def test_with_domain_mask(self):
        """Test with domain mask."""
        func = lambda z: 1/z
        domain = Disk(3)  # Will be used for masking
        
        ax = riemann_chart(func, domain=domain, n=40)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_constant_function(self):
        """Test handling of constant functions."""
        func = lambda z: 1 + 2j
        
        # Should handle scalar output gracefully
        ax = riemann_chart(func, n=30)
        
        assert isinstance(ax, Axes)
        plt.close('all')


class TestRiemannHemispheres:
    """Test riemann_hemispheres() function."""
    
    def test_basic_hemispheres(self):
        """Test basic hemisphere pair plot."""
        func = lambda z: (z**2 + 1) / (z**2 - 1)
        
        fig = riemann_hemispheres(func, n=50)
        
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 2
        assert "South" in fig.axes[0].get_title()
        assert "North" in fig.axes[1].get_title()
        
        plt.close('all')
    
    def test_with_title(self):
        """Test with overall title."""
        func = lambda z: np.exp(z)
        
        fig = riemann_hemispheres(func, title="Exponential", n=40)
        
        assert fig._suptitle.get_text() == "Exponential"
        plt.close('all')
    
    def test_save_to_file(self, tmp_path):
        """Test saving to file."""
        func = lambda z: z**3 - z
        filename = tmp_path / "riemann.png"
        
        fig = riemann_hemispheres(func, n=30, filename=str(filename))
        
        assert filename.exists()
        plt.close('all')


class TestIntegration:
    """Integration tests with various domains and functions."""
    
    def test_rectangle_domain(self):
        """Test with rectangular domain."""
        domain = Rectangle(re_length=4, im_length=2, center=1+0j)
        func = lambda z: np.sin(z) * np.cos(z)
        
        plot(domain=domain, func=func, n=60)
        plt.close('all')
    
    def test_disk_domain(self):
        """Test with disk domain."""
        domain = Disk(radius=2.5, center=-1+1j)
        func = lambda z: (z**4 - 1) / (z**4 + 1)
        
        plot(domain=domain, func=func, n=80)
        plt.close('all')
    
    def test_complex_function(self):
        """Test with more complex function."""
        domain = Rectangle(6, 6)
        
        def func(z):
            # Weierstrass function approximation
            result = np.zeros_like(z, dtype=complex)
            for n in range(5):
                result += np.cos(3**n * np.pi * z) / 2**n
            return result
        
        fig = pair_plot(domain=domain, func=func, n=100,
                       title="Weierstrass approximation")
        
        assert isinstance(fig, Figure)
        plt.close('all')