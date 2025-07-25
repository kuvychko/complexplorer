"""Tests for modulus scaling in 3D matplotlib plotting functions."""

import numpy as np
import pytest
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from complexplorer.plotting.matplotlib.plot_3d import plot_landscape, pair_plot_landscape
from complexplorer.core.domain import Rectangle, Disk
from complexplorer.core.colormap import Phase
from complexplorer.utils.validation import ValidationError


class TestModulusScaling:
    """Test modulus scaling functionality in plot_landscape."""
    
    def test_default_mode(self):
        """Test default mode (none) preserves original moduli."""
        domain = Rectangle(2, 2)
        func = lambda z: z
        
        ax = plot_landscape(domain=domain, func=func, n=20)
        assert isinstance(ax, Axes3D)
        plt.close('all')
    
    def test_constant_mode(self):
        """Test constant mode creates flat surface."""
        domain = Rectangle(2, 2)
        func = lambda z: z**2
        
        ax = plot_landscape(domain=domain, func=func, n=20, 
                          modulus_mode='constant')
        assert isinstance(ax, Axes3D)
        plt.close('all')
    
    def test_linear_mode(self):
        """Test linear scaling mode."""
        domain = Rectangle(2, 2)
        func = lambda z: z
        
        ax = plot_landscape(domain=domain, func=func, n=20,
                          modulus_mode='linear',
                          modulus_params={'scale': 0.5})
        assert isinstance(ax, Axes3D)
        plt.close('all')
    
    def test_arctan_mode(self):
        """Test arctan bounded scaling."""
        domain = Rectangle(3, 3)
        func = lambda z: 1/z  # Has pole at origin
        
        ax = plot_landscape(domain=domain, func=func, n=30,
                          modulus_mode='arctan')
        assert isinstance(ax, Axes3D)
        plt.close('all')
    
    def test_logarithmic_mode(self):
        """Test logarithmic scaling."""
        domain = Disk(2)
        func = lambda z: np.exp(z)
        
        ax = plot_landscape(domain=domain, func=func, n=30,
                          modulus_mode='logarithmic')
        assert isinstance(ax, Axes3D)
        plt.close('all')
    
    def test_custom_mode(self):
        """Test custom scaling function."""
        domain = Rectangle(2, 2)
        func = lambda z: z**2 - 1
        
        def custom_scale(moduli):
            return np.sqrt(moduli)
        
        ax = plot_landscape(domain=domain, func=func, n=25,
                          modulus_mode='custom',
                          modulus_params={'scaling_func': custom_scale})
        assert isinstance(ax, Axes3D)
        plt.close('all')
    
    def test_custom_mode_missing_func(self):
        """Test error when custom mode lacks scaling function."""
        domain = Rectangle(2, 2)
        func = lambda z: z
        
        with pytest.raises(ValidationError, match="Custom mode requires"):
            plot_landscape(domain=domain, func=func,
                         modulus_mode='custom',
                         modulus_params={})
    
    def test_unknown_mode(self):
        """Test error for unknown scaling mode."""
        domain = Rectangle(2, 2)
        func = lambda z: z
        
        with pytest.raises(ValidationError, match="Unknown scaling mode"):
            plot_landscape(domain=domain, func=func,
                         modulus_mode='invalid_mode')
    
    def test_with_z_max(self):
        """Test modulus scaling with z_max clipping."""
        domain = Rectangle(2, 2)
        func = lambda z: z**3
        
        ax = plot_landscape(domain=domain, func=func, n=20,
                          modulus_mode='linear',
                          modulus_params={'scale': 2.0},
                          z_max=5.0)
        assert isinstance(ax, Axes3D)
        plt.close('all')
    
    def test_with_zaxis_log(self):
        """Test modulus scaling with logarithmic z-axis."""
        domain = Rectangle(3, 3)
        func = lambda z: (z-1)/(z+1)
        
        ax = plot_landscape(domain=domain, func=func, n=25,
                          modulus_mode='arctan',
                          zaxis_log=True)
        assert isinstance(ax, Axes3D)
        plt.close('all')


class TestPairPlotModulus:
    """Test modulus scaling in pair_plot_landscape."""
    
    def test_pair_plot_with_modulus(self):
        """Test pair plot with modulus scaling."""
        domain = Rectangle(2, 2)
        func = lambda z: z**2
        
        fig = pair_plot_landscape(domain=domain, func=func, n=20,
                                modulus_mode='arctan')
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 2
        plt.close('all')
    
    def test_pair_plot_custom_params(self):
        """Test pair plot with custom modulus parameters."""
        domain = Disk(1.5)
        func = lambda z: np.sin(z)
        
        fig = pair_plot_landscape(domain=domain, func=func, n=25,
                                modulus_mode='linear',
                                modulus_params={'scale': 0.2})
        assert isinstance(fig, Figure)
        plt.close('all')


class TestIntegrationWithColormaps:
    """Test modulus scaling with various colormaps."""
    
    def test_with_enhanced_phase(self):
        """Test modulus scaling with enhanced phase colormap."""
        domain = Rectangle(3, 3)
        func = lambda z: (z**3 - 1) / (z**3 + 1)
        cmap = Phase(n_phi=6, auto_scale_r=True)
        
        ax = plot_landscape(domain=domain, func=func, cmap=cmap, n=40,
                          modulus_mode='sigmoid',
                          modulus_params={'steepness': 3.0, 'center': 1.0})
        assert isinstance(ax, Axes3D)
        plt.close('all')
    
    def test_adaptive_scaling(self):
        """Test adaptive scaling based on percentiles."""
        domain = Rectangle(4, 4)
        
        def func(z):
            # Function with wide range of values
            return np.where(np.abs(z) < 0.5, 100.0, 1/z)
        
        ax = plot_landscape(domain=domain, func=func, n=50,
                          modulus_mode='adaptive',
                          modulus_params={'low_percentile': 20, 
                                        'high_percentile': 80})
        assert isinstance(ax, Axes3D)
        plt.close('all')


class TestEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_constant_function(self):
        """Test modulus scaling with constant function."""
        domain = Rectangle(2, 2)
        func = lambda z: 2 + 3j
        
        ax = plot_landscape(domain=domain, func=func, n=20,
                          modulus_mode='linear')
        assert isinstance(ax, Axes3D)
        plt.close('all')
    
    def test_infinite_values(self):
        """Test handling of infinite values with scaling."""
        domain = Rectangle(2, 2)
        
        def func(z):
            # Create some infinities
            result = 1/z
            result[5, 5] = np.inf
            return result
        
        # Should handle infinities gracefully
        ax = plot_landscape(domain=domain, func=func, n=20,
                          modulus_mode='arctan')
        assert isinstance(ax, Axes3D)
        plt.close('all')
    
    def test_default_params(self):
        """Test that default parameters are applied correctly."""
        domain = Rectangle(2, 2)
        func = lambda z: z**2
        
        # Should use default parameters for each mode
        for mode in ['linear', 'arctan', 'logarithmic', 'power']:
            ax = plot_landscape(domain=domain, func=func, n=20,
                              modulus_mode=mode)
            assert isinstance(ax, Axes3D)
            plt.close('all')