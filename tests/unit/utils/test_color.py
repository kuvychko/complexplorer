"""Unit tests for color export utilities."""

import pytest
import numpy as np
import complexplorer as cp
from complexplorer.utils.color import get_color, get_color_array, interpolate_colormap
from complexplorer.core.colormap import Phase, OklabPhase


class TestGetColor:
    """Test the get_color function."""
    
    def test_get_color_scalar_rgb(self):
        """Test getting RGB color for a scalar complex value."""
        color = get_color(1 + 1j)
        
        assert isinstance(color, tuple)
        assert len(color) == 3
        assert all(0 <= c <= 1 for c in color)
    
    def test_get_color_scalar_hsv(self):
        """Test getting HSV color for a scalar complex value."""
        color = get_color(1j, format='hsv')
        
        assert isinstance(color, tuple)
        assert len(color) == 3
        assert all(0 <= c <= 1 for c in color)
    
    def test_get_color_scalar_hex(self):
        """Test getting hex color for a scalar complex value."""
        color = get_color(-1, format='hex')
        
        assert isinstance(color, str)
        assert color.startswith('#')
        assert len(color) == 7
        # Check valid hex
        int(color[1:], 16)
    
    def test_get_color_array_rgb(self):
        """Test getting RGB colors for an array."""
        z = np.array([1, 1j, -1, -1j])
        colors = get_color(z)
        
        assert isinstance(colors, np.ndarray)
        assert colors.shape == (4, 3)
        assert np.all(colors >= 0)
        assert np.all(colors <= 1)
    
    def test_get_color_array_hex(self):
        """Test getting hex colors for an array."""
        z = np.array([[1, 1j], [-1, -1j]])
        colors = get_color(z, format='hex')
        
        assert isinstance(colors, np.ndarray)
        assert colors.shape == (2, 2)
        assert all(isinstance(c, str) and c.startswith('#') for c in colors.flat)
    
    def test_get_color_custom_colormap(self):
        """Test get_color with custom colormap."""
        cmap = OklabPhase(phase_sectors=8, enhanced=False)
        color = get_color(1 + 1j, cmap=cmap)
        
        assert isinstance(color, tuple)
        assert len(color) == 3
    
    def test_get_color_invalid_format(self):
        """Test get_color with invalid format."""
        with pytest.raises(ValueError, match="Unknown format"):
            get_color(1j, format='invalid')
    
    def test_get_color_preserves_shape(self):
        """Test that get_color preserves input shape."""
        # 2D array
        z = np.ones((3, 4), dtype=complex)
        colors = get_color(z)
        assert colors.shape == (3, 4, 3)
        
        # 3D array
        z = np.ones((2, 3, 4), dtype=complex)
        colors = get_color(z)
        assert colors.shape == (2, 3, 4, 3)


class TestGetColorArray:
    """Test the get_color_array function."""
    
    def test_get_color_array_basic(self):
        """Test basic color array generation."""
        z = np.random.randn(10, 10) + 1j * np.random.randn(10, 10)
        colors = get_color_array(z)
        
        assert colors.shape == (10, 10, 3)
        assert colors.dtype == np.float64
        assert np.all(colors >= 0)
        assert np.all(colors <= 1)
    
    def test_get_color_array_custom_cmap(self):
        """Test color array with custom colormap."""
        z = np.exp(1j * np.linspace(0, 2*np.pi, 100).reshape(10, 10))
        cmap = Phase(phase_sectors=12, auto_scale_r=True)
        colors = get_color_array(z, cmap)
        
        assert colors.shape == (10, 10, 3)
    
    def test_get_color_array_large(self):
        """Test performance with large arrays."""
        # Should handle reasonably large arrays efficiently
        z = np.random.randn(500, 500) + 1j * np.random.randn(500, 500)
        colors = get_color_array(z)
        
        assert colors.shape == (500, 500, 3)


class TestInterpolateColormap:
    """Test colormap interpolation."""
    
    def test_interpolate_t0(self):
        """Test interpolation at t=0 gives first colormap."""
        z = 1 + 1j
        cmap1 = Phase(phase_sectors=6)
        cmap2 = OklabPhase(phase_sectors=6)
        
        color_interp = interpolate_colormap(z, cmap1, cmap2, t=0)
        color_cmap1 = cmap1.rgb(z)
        
        np.testing.assert_allclose(color_interp, color_cmap1, rtol=1e-10)
    
    def test_interpolate_t1(self):
        """Test interpolation at t=1 gives second colormap."""
        z = 1 + 1j
        cmap1 = Phase(phase_sectors=6)
        cmap2 = OklabPhase(phase_sectors=6, enhanced=False)
        
        color_interp = interpolate_colormap(z, cmap1, cmap2, t=1)
        color_cmap2 = cmap2.rgb(z)
        
        np.testing.assert_allclose(color_interp, color_cmap2, rtol=1e-10)
    
    def test_interpolate_t_half(self):
        """Test interpolation at t=0.5 gives average."""
        z = np.array([1, 1j, -1])
        cmap1 = Phase(phase_sectors=6)
        cmap2 = Phase(phase_sectors=12)
        
        color_interp = interpolate_colormap(z, cmap1, cmap2, t=0.5)
        color1 = cmap1.rgb(z)
        color2 = cmap2.rgb(z)
        expected = 0.5 * color1 + 0.5 * color2
        
        np.testing.assert_allclose(color_interp, expected, rtol=1e-10)
    
    def test_interpolate_invalid_t(self):
        """Test interpolation with invalid t values."""
        z = 1j
        cmap1 = Phase()
        cmap2 = OklabPhase()
        
        with pytest.raises(ValueError):
            interpolate_colormap(z, cmap1, cmap2, t=-0.1)
        
        with pytest.raises(ValueError):
            interpolate_colormap(z, cmap1, cmap2, t=1.1)
    
    def test_interpolate_array_input(self):
        """Test interpolation with array input."""
        z = np.ones((5, 5), dtype=complex)
        cmap1 = Phase(phase_sectors=6)
        cmap2 = OklabPhase(phase_sectors=6)
        
        colors = interpolate_colormap(z, cmap1, cmap2, t=0.3)
        assert colors.shape == (5, 5, 3)
        assert np.all(colors >= 0)
        assert np.all(colors <= 1)


class TestColorConsistency:
    """Test color consistency across different methods."""
    
    def test_consistency_scalar(self):
        """Test that different methods give same result for scalars."""
        z = 1 + 2j
        cmap = Phase(phase_sectors=8, r_linear_step=1.0)
        
        # Using get_color
        color1 = get_color(z, cmap=cmap)
        
        # Using get_color_array (should handle scalars)
        color2 = get_color_array(np.array(z), cmap=cmap)
        
        # Using colormap directly
        color3 = tuple(cmap.rgb(z))
        
        np.testing.assert_allclose(color1, color2, rtol=1e-10)
        np.testing.assert_allclose(color1, color3, rtol=1e-10)
    
    def test_consistency_formats(self):
        """Test format conversions are consistent."""
        z = -1 + 0.5j
        
        # Get RGB
        rgb = get_color(z, format='rgb')
        
        # Get hex and convert back
        hex_color = get_color(z, format='hex')
        r_hex = int(hex_color[1:3], 16) / 255
        g_hex = int(hex_color[3:5], 16) / 255
        b_hex = int(hex_color[5:7], 16) / 255
        
        # Should be close (within quantization error)
        np.testing.assert_allclose(rgb[0], r_hex, atol=1/255)
        np.testing.assert_allclose(rgb[1], g_hex, atol=1/255)
        np.testing.assert_allclose(rgb[2], b_hex, atol=1/255)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])