"""Tests for colormap classes."""

import numpy as np
import pytest
from complexplorer.core.colormap import (
    Colormap, Phase, Chessboard, PolarChessboard, LogRings,
    OUT_OF_DOMAIN_COLOR_HSV, sawtooth, sawtooth_log
)
from complexplorer.utils.validation import ValidationError


class ConcreteColormap(Colormap):
    """Concrete colormap for testing abstract class."""
    
    def hsv_tuple(self, z):
        """Simple implementation returning constant colors."""
        shape = np.asarray(z).shape
        H = np.full(shape, 0.5, dtype=float)
        S = np.full(shape, 1.0, dtype=float)
        V = np.full(shape, 1.0, dtype=float)
        return H, S, V


class TestColormap:
    """Test base Colormap class."""
    
    def test_init(self):
        """Test initialization."""
        cmap = ConcreteColormap()
        assert cmap.out_of_domain_hsv == OUT_OF_DOMAIN_COLOR_HSV
        
        custom_color = (0.1, 0.2, 0.3)
        cmap2 = ConcreteColormap(out_of_domain_hsv=custom_color)
        assert cmap2.out_of_domain_hsv == custom_color
    
    def test_hsv_no_mask(self):
        """Test HSV conversion without mask."""
        cmap = ConcreteColormap()
        z = np.array([1+1j, 2+2j])
        
        hsv = cmap.hsv(z)
        
        assert hsv.shape == (2, 3)
        assert np.all(hsv[:, 0] == 0.5)  # H
        assert np.all(hsv[:, 1] == 1.0)  # S
        assert np.all(hsv[:, 2] == 1.0)  # V
    
    def test_hsv_with_mask(self):
        """Test HSV conversion with out-of-domain mask."""
        cmap = ConcreteColormap()
        z = np.array([1+1j, 2+2j, 3+3j])
        mask = np.array([False, True, False])
        
        hsv = cmap.hsv(z, outmask=mask)
        
        # Check masked value has out-of-domain color
        assert hsv[1, 0] == OUT_OF_DOMAIN_COLOR_HSV[0]
        assert hsv[1, 1] == OUT_OF_DOMAIN_COLOR_HSV[1]
        assert hsv[1, 2] == OUT_OF_DOMAIN_COLOR_HSV[2]
        
        # Check unmasked values are normal
        assert hsv[0, 0] == 0.5
        assert hsv[2, 0] == 0.5
    
    def test_rgb_conversion(self):
        """Test RGB conversion."""
        cmap = ConcreteColormap()
        z = np.array([1+1j])
        
        rgb = cmap.rgb(z)
        
        assert rgb.shape == (1, 3)
        assert 0 <= rgb[0, 0] <= 1  # R
        assert 0 <= rgb[0, 1] <= 1  # G
        assert 0 <= rgb[0, 2] <= 1  # B


class TestPhase:
    """Test Phase colormap."""
    
    def test_init_basic(self):
        """Test basic initialization."""
        cmap = Phase()
        
        assert cmap.n_phi is None
        assert cmap.phi is None
        assert cmap.r_linear_step is None
        assert cmap.r_log_base is None
        assert cmap.v_base == 0.5
        assert cmap.auto_scale_r is False
    
    def test_init_enhanced_phase(self):
        """Test initialization with phase enhancement."""
        cmap = Phase(n_phi=6)
        
        assert cmap.n_phi == 6
        assert cmap.phi == np.pi / 6
    
    def test_init_enhanced_modulus(self):
        """Test initialization with modulus enhancement."""
        cmap = Phase(r_linear_step=0.5)
        assert cmap.r_linear_step == 0.5
        
        cmap2 = Phase(r_log_base=2.0)
        assert cmap2.r_log_base == 2.0
    
    def test_init_auto_scale(self):
        """Test auto-scaling initialization."""
        cmap = Phase(n_phi=6, auto_scale_r=True)
        
        expected_step = 2 * np.pi / 6  # For unit circle
        assert abs(cmap.r_linear_step - expected_step) < 1e-10
    
    def test_init_auto_scale_validation(self):
        """Test auto-scaling validation."""
        # Need n_phi for auto_scale_r
        with pytest.raises(ValidationError):
            Phase(auto_scale_r=True)
        
        # Can't specify both auto_scale_r and r_linear_step
        with pytest.raises(ValidationError):
            Phase(n_phi=6, auto_scale_r=True, r_linear_step=0.5)
    
    def test_v_base_validation(self):
        """Test v_base validation."""
        with pytest.raises(ValidationError):
            Phase(v_base=-0.1)
        
        with pytest.raises(ValidationError):
            Phase(v_base=1.0)
        
        with pytest.raises(ValidationError):
            Phase(v_base=1.5)
    
    def test_hsv_basic(self):
        """Test basic phase coloring."""
        cmap = Phase()
        
        # Test specific phase values
        z = np.array([1+0j, 0+1j, -1+0j, 0-1j])  # 0, π/2, π, -π/2
        H, S, V = cmap.hsv_tuple(z)
        
        # Check hue mapping with phase in [0, 2π)
        assert abs(H[0] - 0.0) < 1e-10      # Phase 0 -> H=0.0
        assert abs(H[1] - 0.25) < 1e-10     # Phase π/2 -> H=0.25
        assert abs(H[2] - 0.5) < 1e-10      # Phase π -> H=0.5
        assert abs(H[3] - 0.75) < 1e-10     # Phase 3π/2 -> H=0.75
        
        # Full saturation and value for basic phase
        assert np.all(S == 1.0)
        assert np.all(V == 1.0)
    
    def test_hsv_enhanced_phase(self):
        """Test enhanced phase coloring."""
        cmap = Phase(n_phi=4, v_base=0.5)
        
        z = np.array([1+0j, np.exp(1j*np.pi/8)])
        H, S, V = cmap.hsv_tuple(z)
        
        # Value should vary with phase sectors
        assert V[0] != V[1]  # Different sectors
        assert 0.5 <= V[0] <= 1.0  # Within range
        assert 0.5 <= V[1] <= 1.0
    
    def test_hsv_enhanced_modulus(self):
        """Test enhanced modulus coloring."""
        cmap = Phase(r_linear_step=1.0, v_base=0.5)
        
        z = np.array([0.5+0j, 1.0+0j, 1.5+0j])
        H, S, V = cmap.hsv_tuple(z)
        
        # Value should vary with modulus
        assert not np.allclose(V[0], V[1])
        assert not np.allclose(V[1], V[2])


class TestChessboard:
    """Test Chessboard colormap."""
    
    def test_init(self):
        """Test initialization."""
        cmap = Chessboard()
        assert cmap.spacing == 1.0
        assert cmap.center == 0+0j
        
        cmap2 = Chessboard(spacing=2.0, center=1+1j)
        assert cmap2.spacing == 2.0
        assert cmap2.center == 1+1j
    
    def test_hsv_pattern(self):
        """Test chessboard pattern."""
        cmap = Chessboard(spacing=1.0)
        
        # Test points in different squares
        z = np.array([
            0.5+0.5j,   # Square (0,0) -> white
            1.5+0.5j,   # Square (1,0) -> black
            0.5+1.5j,   # Square (0,1) -> black
            1.5+1.5j,   # Square (1,1) -> white
        ])
        
        H, S, V = cmap.hsv_tuple(z)
        
        # No color (grayscale)
        assert np.all(H == 0)
        assert np.all(S == 0)
        
        # Check pattern
        assert V[0] == 1.0  # White
        assert V[1] == 0.0  # Black
        assert V[2] == 0.0  # Black
        assert V[3] == 1.0  # White
    
    def test_hsv_with_center(self):
        """Test chessboard with custom center."""
        cmap = Chessboard(center=1+1j)
        
        # Point at (1,1) should now be in square (0,0)
        z = np.array([1+1j])
        H, S, V = cmap.hsv_tuple(z)
        
        assert V[0] == 1.0  # White (origin square)


class TestPolarChessboard:
    """Test PolarChessboard colormap."""
    
    def test_init(self):
        """Test initialization."""
        cmap = PolarChessboard()
        assert cmap.n_phi == 6
        assert cmap.spacing == 1.0
        assert cmap.r_log is None
        
        cmap2 = PolarChessboard(n_phi=8, spacing=0.5, r_log=2.0)
        assert cmap2.n_phi == 8
        assert cmap2.spacing == 0.5
        assert cmap2.r_log == 2.0
    
    def test_hsv_pattern(self):
        """Test polar chessboard pattern."""
        cmap = PolarChessboard(n_phi=4, spacing=1.0)
        
        # Test points at different angles and radii
        z = np.array([
            1.0 * np.exp(0j),          # r=1, θ=0
            1.0 * np.exp(1j*np.pi/2),  # r=1, θ=π/2
            2.0 * np.exp(0j),          # r=2, θ=0
        ])
        
        H, S, V = cmap.hsv_tuple(z)
        
        # No color (grayscale)
        assert np.all(H == 0)
        assert np.all(S == 0)
        
        # Pattern should alternate
        # (Note: exact values depend on floor calculations)
        assert V[0] != V[2]  # Different radius, same angle


class TestLogRings:
    """Test LogRings colormap."""
    
    def test_init(self):
        """Test initialization."""
        cmap = LogRings()
        assert cmap.log_spacing == 0.2
        
        cmap2 = LogRings(log_spacing=0.5)
        assert cmap2.log_spacing == 0.5
    
    def test_hsv_pattern(self):
        """Test logarithmic rings pattern."""
        cmap = LogRings(log_spacing=np.log(2))  # Double radius = next ring
        
        z = np.array([
            1.0+0j,    # r=1
            2.0+0j,    # r=2
            4.0+0j,    # r=4
            0.5+0j,    # r=0.5
        ])
        
        H, S, V = cmap.hsv_tuple(z)
        
        # No color (grayscale)
        assert np.all(H == 0)
        assert np.all(S == 0)
        
        # Check ring pattern
        assert V[0] == V[2]  # Same ring (mod 2)
        assert V[0] != V[1]  # Different rings
    
    def test_hsv_at_origin(self):
        """Test behavior at origin."""
        cmap = LogRings()
        
        z = np.array([0+0j])
        H, S, V = cmap.hsv_tuple(z)
        
        # Should handle log(0) gracefully
        assert V[0] == 1.0  # White at origin


class TestHelperFunctions:
    """Test helper functions."""
    
    def test_sawtooth(self):
        """Test sawtooth function."""
        x = np.array([0, 0.5, 1.0, 1.5, 2.0, -0.5])
        
        result = sawtooth(x)
        
        expected = np.array([0, 0.5, 0, 0.5, 0, 0.5])
        np.testing.assert_allclose(result, expected)
        
        # Test with different period
        result2 = sawtooth(x, period=2.0)
        expected2 = np.array([0, 0.25, 0.5, 0.75, 0, 0.75])
        np.testing.assert_allclose(result2, expected2)
    
    def test_sawtooth_log(self):
        """Test logarithmic sawtooth."""
        r = np.array([1, 2, 4, 8, 0.5])
        
        result = sawtooth_log(r, base=2.0)
        
        # log2(1)=0, log2(2)=1, log2(4)=2, log2(8)=3, log2(0.5)=-1
        # mod 1: 0, 0, 0, 0, 0 (all integer logs)
        expected = np.array([0, 0, 0, 0, 0])
        np.testing.assert_allclose(result, expected)
        
        # Test at origin
        r_with_zero = np.array([0, 1, 2])
        result = sawtooth_log(r_with_zero, base=2.0)
        assert result[0] == 0.0  # Special case


class TestBackwardCompatibility:
    """Test backward compatibility with legacy colormaps."""
    
    def test_phase_compatibility(self):
        """Test Phase matches legacy behavior."""
        # Basic phase
        cmap = Phase()
        z = np.array([1+0j, 1j, -1+0j, -1j])
        
        hsv = cmap.hsv(z)
        
        # Should produce standard phase colors
        assert hsv.shape == (4, 3)
        assert 0 <= hsv.min() <= hsv.max() <= 1
    
    def test_enhanced_phase_compatibility(self):
        """Test enhanced phase matches legacy."""
        cmap = Phase(n_phi=6, r_linear_step=1.0, v_base=0.5)
        
        # Use values that give different sawtooth results
        z = np.array([0.25+0j, 0.75+0j])  # r=0.25 and r=0.75
        hsv = cmap.hsv(z)
        
        # Should have value modulation
        assert hsv[0, 2] != hsv[1, 2]  # Different V values due to different r
        # Verify it's producing enhanced phase portrait
        assert 0.5 <= hsv[0, 2] <= 1.0  # Values are in expected range