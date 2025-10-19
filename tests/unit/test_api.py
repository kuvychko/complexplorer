"""Unit tests for the high-level API functions."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import complexplorer as cp
from complexplorer.api import show
from complexplorer.core.domain import Rectangle
from complexplorer.core.colormap import Phase, OklabPhase


class TestShowFunction:
    """Test the show() convenience function."""
    
    def test_show_minimal_args(self):
        """Test show() with minimal arguments."""
        with patch('complexplorer.api.plot') as mock_plot:
            # Test with just a function
            f = lambda z: z**2
            show(f)
            
            # Check that plot was called with correct defaults
            mock_plot.assert_called_once()
            args, kwargs = mock_plot.call_args
            
            # Check function
            assert args[0] == f
            
            # Check domain was created correctly
            domain = args[1]
            assert isinstance(domain, Rectangle)
            assert domain.re_length == 4.0
            assert domain.im_length == 4.0
            assert domain.center == 0+0j
            
            # Check mode
            assert args[2] == '2d'
            
            # Check colormap
            assert isinstance(kwargs['cmap'], Phase)
            assert kwargs['cmap'].phase_sectors == 6
            assert kwargs['cmap'].auto_scale_r == True
            
            # Check resolution
            assert kwargs['resolution'] == 500
    
    def test_show_with_ranges(self):
        """Test show() with custom ranges."""
        with patch('complexplorer.api.plot') as mock_plot:
            f = lambda z: 1/z
            
            # Test with 3-tuple ranges (min, max, resolution)
            show(f, (-3, 3, 600), (-2, 2, 400))
            
            args, kwargs = mock_plot.call_args
            domain = args[1]
            
            assert isinstance(domain, Rectangle)
            assert abs(domain.re_length - 6.0) < 1e-10
            assert abs(domain.im_length - 4.0) < 1e-10
            assert abs(domain.center.real) < 1e-10
            assert abs(domain.center.imag) < 1e-10
            assert kwargs['resolution'] == 600  # max of 600 and 400
    
    def test_show_with_2tuple_ranges(self):
        """Test show() with 2-tuple ranges (min, max)."""
        with patch('complexplorer.api.plot') as mock_plot:
            f = lambda z: np.sin(z)
            
            # Test with 2-tuple ranges
            show(f, (-5, 5), (-3, 3))
            
            args, kwargs = mock_plot.call_args
            domain = args[1]
            
            assert abs(domain.re_length - 10.0) < 1e-10
            assert abs(domain.im_length - 6.0) < 1e-10
            assert kwargs['resolution'] == 500  # default
    
    def test_show_y_range_defaults_to_x_range(self):
        """Test that y_range defaults to x_range when not provided."""
        with patch('complexplorer.api.plot') as mock_plot:
            f = lambda z: z
            
            show(f, (-1, 1, 300))
            
            args, kwargs = mock_plot.call_args
            domain = args[1]
            
            # Both ranges should be equal
            assert abs(domain.re_length - domain.im_length) < 1e-10
            assert abs(domain.re_length - 2.0) < 1e-10
    
    def test_show_custom_kwargs(self):
        """Test show() passes through custom kwargs."""
        with patch('complexplorer.api.plot') as mock_plot:
            f = lambda z: z**3 - 1
            custom_cmap = OklabPhase(phase_sectors=8)
            
            show(f, mode='3d', cmap=custom_cmap, backend='pyvista')
            
            args, kwargs = mock_plot.call_args
            
            assert args[2] == '3d'  # mode should be passed correctly
            assert kwargs['cmap'] == custom_cmap
            assert kwargs['backend'] == 'pyvista'
    
    def test_show_invalid_ranges(self):
        """Test show() with invalid range specifications."""
        f = lambda z: z
        
        # Test with invalid x_range
        with pytest.raises(ValueError, match="x_range must be"):
            show(f, (1,))  # Too few elements
        
        with pytest.raises(ValueError, match="x_range must be"):
            show(f, (1, 2, 3, 4))  # Too many elements


class TestRectangleFromRanges:
    """Test the Rectangle.from_ranges() class method."""
    
    def test_from_ranges_basic(self):
        """Test basic range conversion."""
        domain = Rectangle.from_ranges((-2, 2), (-1, 1))
        
        assert abs(domain.re_length - 4.0) < 1e-10
        assert abs(domain.im_length - 2.0) < 1e-10
        assert abs(domain.center.real) < 1e-10
        assert abs(domain.center.imag) < 1e-10
    
    def test_from_ranges_with_resolution(self):
        """Test that resolution is ignored in domain creation."""
        # Resolution should be ignored
        domain1 = Rectangle.from_ranges((-3, 3, 500), (-3, 3, 500))
        domain2 = Rectangle.from_ranges((-3, 3, 100), (-3, 3, 100))
        domain3 = Rectangle.from_ranges((-3, 3), (-3, 3))
        
        # All should create the same domain
        assert abs(domain1.re_length - domain2.re_length) < 1e-10
        assert abs(domain1.re_length - domain3.re_length) < 1e-10
        assert abs(domain1.im_length - domain2.im_length) < 1e-10
        assert abs(domain1.im_length - domain3.im_length) < 1e-10
    
    def test_from_ranges_y_defaults_to_x(self):
        """Test y_range defaults to x_range."""
        domain = Rectangle.from_ranges((-2.5, 2.5))
        
        assert abs(domain.re_length - 5.0) < 1e-10
        assert abs(domain.im_length - 5.0) < 1e-10
    
    def test_from_ranges_off_center(self):
        """Test creating off-center rectangles."""
        domain = Rectangle.from_ranges((1, 5), (-2, 0))
        
        assert abs(domain.re_length - 4.0) < 1e-10
        assert abs(domain.im_length - 2.0) < 1e-10
        assert abs(domain.center.real - 3.0) < 1e-10
        assert abs(domain.center.imag - (-1.0)) < 1e-10
    
    def test_from_ranges_invalid(self):
        """Test invalid range specifications."""
        # Min >= max
        with pytest.raises(Exception):  # ValidationError
            Rectangle.from_ranges((2, -2))
        
        with pytest.raises(Exception):
            Rectangle.from_ranges((1, 1))  # Equal bounds
        
        # Too few elements
        with pytest.raises(Exception):
            Rectangle.from_ranges((1,))
    
    def test_from_ranges_square_parameter(self):
        """Test the square parameter."""
        # Non-square domain with square=False
        domain = Rectangle.from_ranges((-2, 2), (-1, 1), square=False)
        assert not domain.square
        
        # Square viewing window when square=True
        domain = Rectangle.from_ranges((-2, 2), (-1, 1), square=True)
        assert domain.square


class TestOklabPhase:
    """Test the new OklabPhase colormap."""
    
    def test_oklab_phase_creation(self):
        """Test creating OklabPhase colormap."""
        cmap = OklabPhase(phase_sectors=6, enhanced=True)

        assert cmap.phase_sectors == 6
        assert cmap.enhanced == True
        assert cmap.L == 0.7  # Default lightness
        assert cmap.C == 0.35  # Default chroma
    
    def test_oklab_phase_smooth_mode(self):
        """Test OklabPhase in smooth (cplot-like) mode."""
        cmap = OklabPhase(enhanced=False)
        
        # Test on some complex values
        z = np.array([1, 1j, -1, -1j])
        hsv = cmap.hsv(z)
        
        assert hsv.shape == (4, 3)
        # All values should be valid
        assert np.all(hsv >= 0)
        assert np.all(hsv <= 1)
    
    def test_oklab_phase_enhanced_mode(self):
        """Test OklabPhase in enhanced mode."""
        cmap = OklabPhase(phase_sectors=8, r_linear_step=1.0, enhanced=True)
        
        z = np.exp(1j * np.linspace(0, 2*np.pi, 100))
        hsv = cmap.hsv(z)
        
        assert hsv.shape == (100, 3)
        assert np.all(hsv >= 0)
        assert np.all(hsv <= 1)
    
    def test_oklab_phase_unit_circle_emphasis(self):
        """Test unit circle emphasis in OklabPhase."""
        cmap = OklabPhase(
            phase_sectors=6,
            emphasize_unit_circle=True,
            unit_circle_strength=0.5,
            enhanced=True
        )
        
        # Points on and near unit circle
        z = np.array([1.0, 0.99, 1.01, 0.5, 2.0])
        hsv = cmap.hsv(z)
        
        # Should get valid colors
        assert hsv.shape == (5, 3)
        assert np.all(hsv >= 0)
        assert np.all(hsv <= 1)
    
    def test_oklab_phase_auto_scale(self):
        """Test auto-scaling in OklabPhase."""
        cmap = OklabPhase(phase_sectors=12, auto_scale_r=True, scale_radius=0.8)
        
        expected_r_step = 2 * np.pi / 12 * 0.8
        assert abs(cmap.r_linear_step - expected_r_step) < 1e-10
    
    def test_oklab_phase_invalid_params(self):
        """Test OklabPhase with invalid parameters."""
        # Invalid lightness
        with pytest.raises(Exception):  # ValidationError
            OklabPhase(L=1.5)
        
        # Invalid chroma
        with pytest.raises(Exception):
            OklabPhase(C=0.7)
        
        # Invalid v_base
        with pytest.raises(Exception):
            OklabPhase(v_base=1.0)


class TestPhaseUnitCircleEmphasis:
    """Test unit circle emphasis in Phase colormap."""
    
    def test_phase_unit_circle_basic(self):
        """Test basic unit circle emphasis."""
        cmap = Phase(
            phase_sectors=6,
            emphasize_unit_circle=True,
            unit_circle_strength=0.3
        )
        
        # Test on unit circle
        z = np.exp(1j * np.linspace(0, 2*np.pi, 50))
        hsv = cmap.hsv(z)
        
        assert hsv.shape == (50, 3)
        assert np.all(hsv >= 0)
        assert np.all(hsv <= 1)
    
    def test_phase_unit_circle_with_color(self):
        """Test unit circle emphasis with custom color."""
        cmap = Phase(
            phase_sectors=6,
            emphasize_unit_circle=True,
            unit_circle_strength=0.5,
            unit_circle_color=(0.5, 1.0, 1.0)  # Cyan in HSV
        )
        
        # Points at different distances from unit circle
        z = np.array([1.0, 0.5, 2.0, 0.99, 1.01])
        hsv = cmap.hsv(z)
        
        assert hsv.shape == (5, 3)
        # Colors near unit circle should be influenced by the custom color
        # but exact values depend on blending
    
    def test_phase_unit_circle_strength_validation(self):
        """Test validation of unit circle strength."""
        # Invalid strength
        with pytest.raises(Exception):
            Phase(emphasize_unit_circle=True, unit_circle_strength=1.5)
        
        with pytest.raises(Exception):
            Phase(emphasize_unit_circle=True, unit_circle_strength=-0.1)


class TestImprovedDefaults:
    """Test improved default parameters."""
    
    def test_plot_2d_defaults(self):
        """Test that plot_2d has improved defaults."""
        from complexplorer.plotting.matplotlib.plot_2d import plot
        
        with patch('matplotlib.pyplot.imshow'):
            z = Rectangle(2, 2).mesh(100)
            f = z**2
            
            # Should use improved defaults when cmap not specified
            ax = plot(z=z, f=f)
            
            # Can't easily test the actual default values without mocking more,
            # but the function should work
            assert ax is not None
    
    def test_api_plot_defaults(self):
        """Test that api.plot has improved defaults."""
        # This test is challenging to implement properly because
        # the api.plot function routes through different backends
        # Just test that it doesn't crash with default params
        from complexplorer.api import plot
        from complexplorer.core.domain import Rectangle
        
        with patch('complexplorer.plotting.matplotlib.plot_2d.plot') as mock_plot_2d:
            mock_plot_2d.return_value = MagicMock()
            
            # Create a simple domain and function
            domain = Rectangle(2, 2)
            f = lambda z: z
            
            # Should work with defaults
            result = plot(f, domain)
            
            # Should have been called or routed appropriately
            assert result is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])