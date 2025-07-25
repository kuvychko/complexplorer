"""Tests for mathematical functions."""

import numpy as np
import pytest
from complexplorer.core.functions import (
    phase, sawtooth, sawtooth_log, sawtooth_legacy,
    stereographic_projection, inverse_stereographic
)


class TestPhase:
    """Test phase function."""
    
    def test_phase_scalar(self):
        """Test phase with scalar inputs."""
        # Basic directions
        assert phase(1+0j) == 0.0
        assert abs(phase(0+1j) - np.pi/2) < 1e-10
        assert abs(phase(-1+0j) - np.pi) < 1e-10
        assert abs(phase(0-1j) - 3*np.pi/2) < 1e-10
        
        # Should be in [0, 2π)
        assert phase(-1-1j) > 0
        assert phase(-1-1j) < 2*np.pi
    
    def test_phase_array(self):
        """Test phase with array inputs."""
        z = np.array([1+0j, 1j, -1+0j, -1j])
        phi = phase(z)
        
        expected = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        np.testing.assert_allclose(phi, expected, rtol=1e-10)
        
        # All values should be in [0, 2π)
        assert np.all(phi >= 0)
        assert np.all(phi < 2*np.pi)
    
    def test_phase_preserves_shape(self):
        """Test that phase preserves input shape."""
        z = np.ones((2, 3), dtype=complex)
        phi = phase(z)
        assert phi.shape == (2, 3)


class TestSawtooth:
    """Test sawtooth function."""
    
    def test_sawtooth_basic(self):
        """Test basic sawtooth behavior."""
        # Values in [0, 1)
        assert sawtooth(0.0) == 0.0
        assert sawtooth(0.5) == 0.5
        assert sawtooth(0.99) == 0.99
        
        # Periodic behavior
        assert sawtooth(1.0) == 0.0
        assert sawtooth(1.5) == 0.5
        assert sawtooth(2.0) == 0.0
        
        # Negative values
        assert sawtooth(-0.5) == 0.5
        assert sawtooth(-1.0) == 0.0
    
    def test_sawtooth_period(self):
        """Test sawtooth with different periods."""
        # Period 2
        assert sawtooth(0.5, period=2.0) == 0.25
        assert sawtooth(1.0, period=2.0) == 0.5
        assert sawtooth(2.0, period=2.0) == 0.0
        
        # Period 0.5
        assert sawtooth(0.25, period=0.5) == 0.5
        assert sawtooth(0.5, period=0.5) == 0.0
    
    def test_sawtooth_array(self):
        """Test sawtooth with array input."""
        x = np.array([0, 0.5, 1.0, 1.5, 2.0])
        result = sawtooth(x)
        expected = np.array([0, 0.5, 0.0, 0.5, 0.0])
        np.testing.assert_allclose(result, expected)


class TestSawtoothLog:
    """Test logarithmic sawtooth function."""
    
    def test_sawtooth_log_base_e(self):
        """Test with natural logarithm."""
        # e^0 = 1, log(1) = 0
        assert sawtooth_log(1.0) == 0.0
        
        # e^1 = e, log(e) = 1 -> 0
        assert abs(sawtooth_log(np.e) - 0.0) < 1e-10
        
        # e^0.5, log(e^0.5) = 0.5
        assert abs(sawtooth_log(np.exp(0.5)) - 0.5) < 1e-10
    
    def test_sawtooth_log_base_2(self):
        """Test with base 2."""
        # Powers of 2
        assert sawtooth_log(1.0, base=2.0) == 0.0  # 2^0
        assert sawtooth_log(2.0, base=2.0) == 0.0  # 2^1
        assert sawtooth_log(4.0, base=2.0) == 0.0  # 2^2
        
        # Between powers
        assert abs(sawtooth_log(np.sqrt(2), base=2.0) - 0.5) < 1e-10
    
    def test_sawtooth_log_zero(self):
        """Test behavior at zero."""
        # Should handle log(0) gracefully
        assert sawtooth_log(0.0) == 0.0
        
        # Array with zero
        x = np.array([0.0, 1.0, 2.0])
        result = sawtooth_log(x, base=2.0)
        assert result[0] == 0.0


class TestSawtoothLegacy:
    """Test legacy sawtooth function."""
    
    def test_legacy_formula(self):
        """Test ceil(x) - x formula."""
        # Values in (0, 1]
        assert sawtooth_legacy(0.1) == 0.9  # ceil(0.1) - 0.1 = 1 - 0.1
        assert sawtooth_legacy(0.5) == 0.5  # ceil(0.5) - 0.5 = 1 - 0.5
        assert sawtooth_legacy(1.0) == 0.0  # ceil(1) - 1 = 1 - 1
        
        # Next period
        assert abs(sawtooth_legacy(1.1) - 0.9) < 1e-10  # ceil(1.1) - 1.1 = 2 - 1.1
    
    def test_legacy_with_log(self):
        """Test legacy sawtooth with logarithm."""
        result = sawtooth_legacy(4.0, log_base=2.0)
        # log2(4) = 2, ceil(2) - 2 = 0
        assert abs(result) < 1e-10


class TestStereographicProjection:
    """Test stereographic projection."""
    
    def test_basic_points(self):
        """Test projection of basic points."""
        # Origin maps to (0, 0, -1) but gets flipped to (0, 0, 1) with project_from_north=False
        result = stereographic_projection(0+0j)
        np.testing.assert_allclose(result, [0, 0, 1])
        
        # Unit circle maps to equator
        result = stereographic_projection(1+0j)
        np.testing.assert_allclose(result, [1, 0, 0])
        
        result = stereographic_projection(0+1j)
        np.testing.assert_allclose(result, [0, 1, 0])
        
        result = stereographic_projection(-1+0j)
        np.testing.assert_allclose(result, [-1, 0, 0])
    
    def test_projection_from_north(self):
        """Test projection from north pole."""
        # Origin maps to (0, 0, -1) when projecting from north
        result = stereographic_projection(0+0j, project_from_north=True)
        np.testing.assert_allclose(result, [0, 0, -1])
    
    def test_infinity_behavior(self):
        """Test behavior for large values."""
        # Large values should approach the pole
        z_large = 1000+0j
        result = stereographic_projection(z_large)
        assert abs(result[2] - (-1)) < 0.01  # Close to north pole (z=-1 after flip)
    
    def test_array_input(self):
        """Test with array input."""
        z = np.array([0+0j, 1+0j, 0+1j, 1+1j])
        result = stereographic_projection(z)
        
        assert result.shape == (4, 3)
        # Check first point
        np.testing.assert_allclose(result[0], [0, 0, 1])
    
    def test_preserves_symmetry(self):
        """Test that projection preserves symmetries."""
        # Rotation symmetry
        z1 = 1+0j
        z2 = 0+1j
        z3 = -1+0j
        z4 = 0-1j
        
        r1 = stereographic_projection(z1)
        r2 = stereographic_projection(z2)
        r3 = stereographic_projection(z3)
        r4 = stereographic_projection(z4)
        
        # All should have same z-coordinate (on equator)
        assert abs(r1[2] - r2[2]) < 1e-10
        assert abs(r2[2] - r3[2]) < 1e-10
        assert abs(r3[2] - r4[2]) < 1e-10


class TestInverseStereographic:
    """Test inverse stereographic projection."""
    
    def test_inverse_basic(self):
        """Test basic inverse projection."""
        # Point at (0, 0, 1) maps to origin (because of the flip)
        w = inverse_stereographic(0, 0, 1)
        assert abs(w) < 1e-10
        
        # Equator points
        w = inverse_stereographic(1, 0, 0)
        assert abs(w - (1+0j)) < 1e-10
        
        w = inverse_stereographic(0, 1, 0)
        assert abs(w - (0+1j)) < 1e-10
    
    def test_inverse_north_pole(self):
        """Test inverse from north pole."""
        # Point at (0, 0, -1) maps to infinity (because of the flip)  
        w = inverse_stereographic(0, 0, -1)
        assert np.isinf(np.abs(w))
    
    def test_round_trip(self):
        """Test projection followed by inverse."""
        # Test various points
        z_values = [0.5+0.5j, 1+0j, 2+3j, -1+2j]
        
        for z in z_values:
            xyz = stereographic_projection(z)
            w = inverse_stereographic(xyz[0], xyz[1], xyz[2])
            assert abs(w - z) < 1e-10
    
    def test_array_input(self):
        """Test with array inputs."""
        x = np.array([1, 0, -1, 0])
        y = np.array([0, 1, 0, -1])
        z = np.array([0, 0, 0, 0])
        
        w = inverse_stereographic(x, y, z)
        
        expected = np.array([1+0j, 0+1j, -1+0j, 0-1j])
        np.testing.assert_allclose(w, expected)


class TestBackwardCompatibility:
    """Test backward compatibility with legacy functions."""
    
    def test_phase_compatibility(self):
        """Test phase matches legacy behavior."""
        # Legacy phase returns [0, 2π)
        z = np.array([1+0j, 1j, -1+0j, -1j])
        phi = phase(z)
        
        # All values should be non-negative
        assert np.all(phi >= 0)
        assert np.all(phi < 2*np.pi)
    
    def test_stereographic_alias(self):
        """Test stereographic alias works."""
        from complexplorer.core.functions import stereographic
        
        result1 = stereographic(1+1j)
        result2 = stereographic_projection(1+1j)
        
        np.testing.assert_allclose(result1, result2)