"""End-to-end integration tests for complexplorer workflows."""

import pytest
import numpy as np
import tempfile
import os

# Import from new API structure
from complexplorer.core.domain import Rectangle, Disk, Annulus
from complexplorer.core.colormap import Phase, Chessboard, PolarChessboard, LogRings
from complexplorer.plotting.matplotlib.plot_2d import plot, pair_plot
from complexplorer.plotting.matplotlib.plot_3d import plot_landscape, riemann
from complexplorer.api import quick_plot, analyze_function, Presets
from complexplorer.core.scaling import ModulusScaling, get_scaling_preset

# Check if optional features are available
try:
    from complexplorer.export.stl import create_ornament
    HAS_STL_EXPORT = True
except ImportError:
    HAS_STL_EXPORT = False

try:
    from complexplorer.plotting.pyvista.plot_3d import plot_landscape_pv
    from complexplorer.plotting.pyvista.riemann import riemann_pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False


class TestBasicWorkflows:
    """Test basic end-to-end workflows."""
    
    def test_simple_visualization_pipeline(self):
        """Test a simple visualization from start to finish."""
        # Define function and domain
        func = lambda z: (z**2 - 1) / (z**2 + 1)
        domain = Rectangle(2, 2)
        
        # Create colormap
        cmap = Phase(n_phi=6, auto_scale_r=True)
        
        # Create 2D plot
        ax = plot(domain, func, cmap=cmap, n=50)
        assert ax is not None
        
        # Close plot
        import matplotlib.pyplot as plt
        plt.close('all')
    
    def test_disk_domain_workflow(self):
        """Test workflow with disk domain."""
        # Function with pole at origin
        func = lambda z: 1 / z
        
        # Use annulus to avoid pole
        domain = Annulus(0.5, 2.0)
        
        # Use enhanced phase portrait
        cmap = Phase(n_phi=12, r_linear_step=0.5, v_base=0.4)
        
        # Create plot
        ax = plot(domain, func, cmap=cmap, n=60)
        assert ax is not None
        
        import matplotlib.pyplot as plt
        plt.close('all')
    
    def test_3d_landscape_workflow(self):
        """Test 3D landscape visualization."""
        func = lambda z: np.sin(z)
        domain = Rectangle(4, 4)
        
        # Create 3D plot
        ax = plot_landscape(domain, func=func, n=40)
        assert ax is not None
        
        import matplotlib.pyplot as plt
        plt.close('all')
    
    def test_riemann_sphere_workflow(self):
        """Test Riemann sphere visualization."""
        func = lambda z: z**3 - z
        domain = Rectangle(3, 3)
        
        # Create Riemann sphere plot
        ax = riemann(func=func, n=30)
        assert ax is not None
        
        import matplotlib.pyplot as plt
        plt.close('all')


class TestColormapVariations:
    """Test different colormap configurations."""
    
    def test_all_colormap_types(self):
        """Test all available colormap types."""
        func = lambda z: z**2
        domain = Rectangle(2, 2)
        
        # Test each colormap type
        colormaps = [
            Phase(),
            Phase(n_phi=6, auto_scale_r=True),
            Chessboard(spacing=0.5),
            PolarChessboard(spacing=0.5, n_phi=12),
            LogRings(log_spacing=0.3)
        ]
        
        for cmap in colormaps:
            ax = plot(domain, func, cmap=cmap, n=30)
            assert ax is not None
        
        import matplotlib.pyplot as plt
        plt.close('all')
    
    def test_enhanced_phase_variations(self):
        """Test enhanced phase portrait variations."""
        func = lambda z: (z - 1) * (z + 1) * (z - 1j) * (z + 1j)
        domain = Rectangle(3, 3)
        
        # Different enhanced phase configurations
        configs = [
            {'n_phi': 4, 'r_linear_step': 1.0, 'v_base': 0.5},
            {'n_phi': 8, 'r_linear_step': 0.5, 'v_base': 0.3},
            {'n_phi': 12, 'auto_scale_r': True, 'v_base': 0.4},
            {'n_phi': 6, 'r_log_base': 2.0, 'v_base': 0.5}
        ]
        
        for config in configs:
            cmap = Phase(**config)
            ax = plot(domain, func, cmap=cmap, n=40)
            assert ax is not None
        
        import matplotlib.pyplot as plt
        plt.close('all')


class TestHighLevelAPI:
    """Test high-level API functions."""
    
    def test_quick_plot(self):
        """Test quick_plot convenience function."""
        func = lambda z: z**2 - 1
        
        # 2D quick plot
        ax = quick_plot(func, mode='2d')
        assert ax is not None
        
        # 3D quick plot
        ax = quick_plot(func, mode='3d', n=30)
        assert ax is not None
        
        import matplotlib.pyplot as plt
        plt.close('all')
    
    def test_analyze_function(self):
        """Test analyze_function utility."""
        func = lambda z: (z**2 - 1) / (z**2 + 1)
        
        results = analyze_function(func)
        
        assert 'plot' in results
        assert 'domain' in results
        assert 'function' in results
        assert 'colormap' in results
        
        import matplotlib.pyplot as plt
        plt.close('all')
    
    def test_presets(self):
        """Test preset configurations."""
        func = lambda z: np.exp(z)
        
        # Publication preset
        preset = Presets.publication_ready()
        ax = plot(Rectangle(2, 2), func, **preset)
        assert ax is not None
        
        # High contrast preset
        preset = Presets.high_contrast()
        ax = plot(Rectangle(2, 2), func, **preset)
        assert ax is not None
        
        import matplotlib.pyplot as plt
        plt.close('all')


class TestModulusScaling:
    """Test modulus scaling functionality."""
    
    def test_scaling_methods(self):
        """Test different modulus scaling methods."""
        # Get scaling presets
        presets = ['balanced', 'detail_near_zero', 'auto', 'high_contrast', 'poles_emphasis']
        
        for preset_name in presets:
            preset = get_scaling_preset(preset_name)
            assert isinstance(preset, dict)
            assert 'method' in preset
            assert 'params' in preset
    
    def test_custom_scaling(self):
        """Test custom modulus scaling."""
        # Test direct scaling methods
        r = np.linspace(0, 10, 100)
        
        # Test constant scaling
        scaled = ModulusScaling.constant(r)
        assert scaled.shape == r.shape
        assert np.all(scaled == 1.0)
        
        # Test linear scaling
        scaled = ModulusScaling.linear(r)
        assert scaled.shape == r.shape
        assert np.all(np.isfinite(scaled))
        
        # Test arctan scaling
        scaled = ModulusScaling.arctan(r)
        assert scaled.shape == r.shape
        assert np.all(np.isfinite(scaled))
        
        # Test logarithmic scaling
        scaled = ModulusScaling.logarithmic(r)
        assert scaled.shape == r.shape
        assert np.all(np.isfinite(scaled))
        
        # Test sigmoid scaling
        scaled = ModulusScaling.sigmoid(r)
        assert scaled.shape == r.shape
        assert np.all(np.isfinite(scaled))


@pytest.mark.skipif(not HAS_STL_EXPORT, reason="PyVista not installed")
class TestSTLExportWorkflow:
    """Test STL export workflow."""
    
    def test_basic_stl_export(self):
        """Test basic STL export pipeline."""
        func = lambda z: z**3 - 1
        
        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp:
            filename = tmp.name
        
        try:
            # Create ornament
            saved = create_ornament(
                func, 
                filename,
                size_mm=50,
                resolution=30,
                verbose=False
            )
            
            assert os.path.exists(saved)
            assert os.path.getsize(saved) > 0
            
        finally:
            if os.path.exists(filename):
                os.unlink(filename)
    
    def test_stl_with_custom_parameters(self):
        """Test STL export with custom parameters."""
        func = lambda z: (z**2 - 1) / (z**2 + 1)
        
        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp:
            filename = tmp.name
        
        try:
            # Use custom colormap and scaling
            cmap = Phase(n_phi=12, auto_scale_r=True)
            
            saved = create_ornament(
                func,
                filename,
                size_mm=60,
                resolution=40,
                scaling='adaptive',
                cmap=cmap,
                verbose=False
            )
            
            assert os.path.exists(saved)
            
        finally:
            if os.path.exists(filename):
                os.unlink(filename)


@pytest.mark.skipif(not HAS_PYVISTA, reason="PyVista not installed")  
class TestPyVistaIntegration:
    """Test PyVista integration."""
    
    def test_pyvista_landscape(self):
        """Test PyVista landscape plotting."""
        func = lambda z: z**2
        domain = Rectangle(2, 2)
        
        # Mock the plotter to avoid display
        from unittest import mock
        with mock.patch('pyvista.Plotter'):
            plot_landscape_pv(domain, func, n=30, show=False)
    
    def test_pyvista_riemann(self):
        """Test PyVista Riemann sphere."""
        func = lambda z: (z - 1) / (z + 1)
        
        # Mock the plotter
        from unittest import mock
        with mock.patch('pyvista.Plotter'):
            riemann_pv(func, n_theta=20, n_phi=20, show=False)


class TestComplexFunctions:
    """Test with various complex function types."""
    
    def test_polynomial_functions(self):
        """Test polynomial functions."""
        polynomials = [
            lambda z: z,
            lambda z: z**2,
            lambda z: z**3 - z,
            lambda z: z**4 - 2*z**2 + 1
        ]
        
        domain = Rectangle(2, 2)
        cmap = Phase(n_phi=6, auto_scale_r=True)
        
        for func in polynomials:
            ax = plot(domain, func, cmap=cmap, n=30)
            assert ax is not None
        
        import matplotlib.pyplot as plt
        plt.close('all')
    
    def test_rational_functions(self):
        """Test rational functions with poles."""
        rationals = [
            lambda z: 1 / z,
            lambda z: (z - 1) / (z + 1),
            lambda z: z / (z**2 + 1),
            lambda z: (z**2 - 1) / (z**2 + 1)
        ]
        
        # Use annulus to avoid poles at origin
        domain = Annulus(0.5, 2.0)
        cmap = Phase(n_phi=8, r_linear_step=0.5, v_base=0.4)
        
        for func in rationals:
            ax = plot(domain, func, cmap=cmap, n=40)
            assert ax is not None
        
        import matplotlib.pyplot as plt
        plt.close('all')
    
    def test_transcendental_functions(self):
        """Test transcendental functions."""
        funcs = [
            lambda z: np.exp(z),
            lambda z: np.sin(z),
            lambda z: np.cos(z),
            lambda z: np.sinh(z)
        ]
        
        domain = Rectangle(3, 3)
        
        for func in funcs:
            ax = plot(domain, func, n=40)
            assert ax is not None
        
        import matplotlib.pyplot as plt
        plt.close('all')


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_constant_function(self):
        """Test constant function visualization."""
        func = lambda z: 1 + 0j
        domain = Rectangle(2, 2)
        
        ax = plot(domain, func, n=20)
        assert ax is not None
        
        import matplotlib.pyplot as plt
        plt.close('all')
    
    def test_very_small_domain(self):
        """Test with very small domain."""
        func = lambda z: z**2
        domain = Rectangle(0.1, 0.1)
        
        ax = plot(domain, func, n=20)
        assert ax is not None
        
        import matplotlib.pyplot as plt
        plt.close('all')
    
    def test_highly_oscillatory_function(self):
        """Test highly oscillatory function."""
        func = lambda z: np.sin(10 * z)
        domain = Rectangle(2, 2)
        
        # Need high resolution
        ax = plot(domain, func, n=100)
        assert ax is not None
        
        import matplotlib.pyplot as plt
        plt.close('all')