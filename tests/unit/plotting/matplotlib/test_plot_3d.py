"""Tests for 3D matplotlib plotting functions."""

import numpy as np
import pytest
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from complexplorer.plotting.matplotlib.plot_3d import (
    Matplotlib3DPlotter, plot_landscape, pair_plot_landscape, riemann
)
from complexplorer.core.domain import Rectangle, Disk
from complexplorer.core.colormap import Phase, Chessboard
from complexplorer.utils.validation import ValidationError


class TestMatplotlib3DPlotter:
    """Test Matplotlib3DPlotter class."""
    
    def test_plot_landscape(self):
        """Test basic landscape plot."""
        plotter = Matplotlib3DPlotter()
        domain = Rectangle(4, 4)
        func = lambda z: z**2
        colormap = Phase()
        
        ax = plotter.plot_landscape(domain, func, colormap, resolution=30)
        
        assert isinstance(ax, Axes3D)
        assert ax.get_xlabel() == "Re(z)"
        assert ax.get_ylabel() == "Im(z)"
        
        plt.close('all')
    
    def test_plot_landscape_with_log(self):
        """Test landscape with logarithmic z-axis."""
        plotter = Matplotlib3DPlotter()
        domain = Rectangle(2, 2)
        func = lambda z: 1 / (z - 0.5)
        colormap = Phase()
        
        ax = plotter.plot_landscape(domain, func, colormap, 
                                   resolution=40, zaxis_log=True)
        
        assert isinstance(ax, Axes3D)
        plt.close('all')
    
    def test_plot_landscape_with_z_max(self):
        """Test landscape with z_max clipping."""
        plotter = Matplotlib3DPlotter()
        domain = Disk(2)
        func = lambda z: z**3
        colormap = Chessboard()
        
        ax = plotter.plot_landscape(domain, func, colormap,
                                   resolution=30, z_max=5.0)
        
        assert isinstance(ax, Axes3D)
        plt.close('all')
    
    def test_invalid_z_max(self):
        """Test error for invalid z_max."""
        plotter = Matplotlib3DPlotter()
        domain = Rectangle(1, 1)
        func = lambda z: z
        colormap = Phase()
        
        with pytest.raises(ValidationError, match="positive"):
            plotter.plot_landscape(domain, func, colormap, 
                                 resolution=20, z_max=-1.0)


class TestPlotLandscape:
    """Test plot_landscape() function."""
    
    def test_with_domain_and_func(self):
        """Test with domain and function."""
        domain = Rectangle(3, 3)
        func = lambda z: (z - 1) / (z + 1)
        
        plot_landscape(domain=domain, func=func, n=40)
        plt.close('all')
    
    def test_with_arrays(self):
        """Test with pre-computed arrays."""
        x = np.linspace(-2, 2, 50)
        y = np.linspace(-2, 2, 50)
        X, Y = np.meshgrid(x, y)
        z = X + 1j * Y
        f = np.sin(z) * np.exp(-np.abs(z)/3)
        
        plot_landscape(z=z, f=f)
        plt.close('all')
    
    def test_missing_inputs(self):
        """Test error handling for missing inputs."""
        with pytest.raises(ValidationError, match="domain or z"):
            plot_landscape(func=lambda z: z)
        
        with pytest.raises(ValidationError, match="f or func"):
            plot_landscape(domain=Rectangle(1, 1))
    
    def test_constant_function(self):
        """Test handling of constant functions."""
        domain = Rectangle(2, 2)
        func = lambda z: 2 + 3j
        
        # Should handle scalar output
        plot_landscape(domain=domain, func=func, n=30)
        plt.close('all')
    
    def test_with_axes(self):
        """Test plotting on provided axes."""
        domain = Rectangle(2, 2)
        func = lambda z: z**2 - 1
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        result = plot_landscape(domain=domain, func=func, ax=ax, n=30)
        
        assert result is ax
        plt.close('all')
    
    def test_antialiased(self):
        """Test antialiased rendering."""
        domain = Disk(1.5)
        func = lambda z: np.exp(z)
        
        plot_landscape(domain=domain, func=func, n=40, antialiased=True)
        plt.close('all')
    
    def test_custom_colormap(self):
        """Test with custom colormap."""
        domain = Rectangle(4, 2)
        func = lambda z: z**3 - z
        cmap = Chessboard(spacing=0.5)
        
        plot_landscape(domain=domain, func=func, cmap=cmap, n=50)
        plt.close('all')


class TestPairPlotLandscape:
    """Test pair_plot_landscape() function."""
    
    def test_basic_pair(self):
        """Test basic landscape pair."""
        domain = Rectangle(3, 3)
        func = lambda z: (z**2 + 1) / (z**2 - 1)
        
        fig = pair_plot_landscape(domain=domain, func=func, n=40)
        
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 2
        assert all(isinstance(ax, Axes3D) for ax in fig.axes)
        
        plt.close('all')
    
    def test_with_title(self):
        """Test with title."""
        domain = Disk(2)
        func = lambda z: np.log(z + 1)
        
        fig = pair_plot_landscape(domain=domain, func=func, 
                                 title="Logarithm", n=30)
        
        assert fig._suptitle is not None
        assert fig._suptitle.get_text() == "Logarithm"
        plt.close('all')
    
    def test_with_log_scale(self):
        """Test with logarithmic z-axis."""
        domain = Rectangle(4, 4)
        func = lambda z: 1 / (z**2 + 1)
        
        fig = pair_plot_landscape(domain=domain, func=func,
                                 zaxis_log=True, n=50)
        
        assert len(fig.axes) == 2
        plt.close('all')
    
    def test_save_to_file(self, tmp_path):
        """Test saving to file."""
        domain = Rectangle(2, 2)
        func = lambda z: z**4 - 1
        filename = tmp_path / "landscape.png"
        
        fig = pair_plot_landscape(domain=domain, func=func, n=30,
                                 filename=str(filename))
        
        assert filename.exists()
        plt.close('all')


class TestRiemann:
    """Test riemann() function."""
    
    def test_basic_riemann(self):
        """Test basic Riemann sphere plot."""
        func = lambda z: (z - 1) / (z + 1)
        
        riemann(func, n=50)
        plt.close('all')
    
    def test_with_axes(self):
        """Test plotting on provided axes."""
        func = lambda z: z**2
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        result = riemann(func, n=40, ax=ax)
        
        assert result is ax
        plt.close('all')
    
    def test_project_from_north(self):
        """Test projection from north pole."""
        func = lambda z: 1 / z
        
        riemann(func, n=40, project_from_north=True)
        plt.close('all')
    
    def test_with_title(self):
        """Test with title."""
        func = lambda z: (z**2 - 1) / (z**2 + 1)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        riemann(func, n=30, ax=ax, title="Rational Function")
        
        assert ax.get_title() == "Rational Function"
        plt.close('all')
    
    def test_custom_colormap(self):
        """Test with custom colormap."""
        func = lambda z: np.sin(z)
        cmap = Phase(n_phi=8, v_base=0.7)
        
        riemann(func, n=60, cmap=cmap)
        plt.close('all')
    
    def test_save_to_file(self, tmp_path):
        """Test saving to file."""
        func = lambda z: z**3 - z
        filename = tmp_path / "riemann.png"
        
        riemann(func, n=40, filename=str(filename))
        
        assert filename.exists()
        plt.close('all')
    
    def test_essential_singularity(self):
        """Test function with essential singularity."""
        def func(z):
            # exp(1/z) with safety for z=0
            with np.errstate(divide='ignore', invalid='ignore'):
                result = np.exp(1/z)
            return np.where(np.isfinite(result), result, 0)
        
        riemann(func, n=60)
        plt.close('all')


class TestIntegration3D:
    """Integration tests for 3D plotting."""
    
    def test_poles_and_zeros(self):
        """Test visualization of poles and zeros."""
        domain = Rectangle(4, 4)
        func = lambda z: (z**2 - 1) / (z**2 + 2*z + 2)
        
        # Should show poles and zeros clearly with log scale
        plot_landscape(domain=domain, func=func, n=60, 
                      zaxis_log=True, z_max=10)
        plt.close('all')
    
    def test_branch_cut(self):
        """Test function with branch cut."""
        domain = Rectangle(4, 4, center=0.5+0j)
        func = lambda z: np.sqrt(z)
        
        plot_landscape(domain=domain, func=func, n=80)
        plt.close('all')
    
    def test_oscillating_function(self):
        """Test rapidly oscillating function."""
        domain = Rectangle(10, 10)
        func = lambda z: np.cos(z) * np.sin(z)
        
        fig = pair_plot_landscape(domain=domain, func=func, n=100,
                                 title="Oscillating Function")
        
        assert isinstance(fig, Figure)
        plt.close('all')