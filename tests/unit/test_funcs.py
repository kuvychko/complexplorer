"""
Unit tests for complexplorer function utilities.
"""

import pytest
import numpy as np
from complexplorer.funcs import phase, sawtooth, stereographic


class TestPhaseFunction:
    """Test the phase() function."""
    
    def test_phase_range(self):
        """Test that phase returns values in [0, 2π)."""
        # Test scalar inputs
        test_values = [1, -1, 1j, -1j, 1+1j, -1-1j, 0.5+0.5j]
        
        for z in test_values:
            phi = phase(z)
            assert 0 <= phi < 2 * np.pi, f"phase({z}) = {phi} not in [0, 2π)"
            assert np.isscalar(phi), f"phase({z}) should return scalar"
        
        # Test array input
        z_array = np.array(test_values)
        phi_array = phase(z_array)
        assert phi_array.shape == z_array.shape
        assert np.all(phi_array >= 0) and np.all(phi_array < 2 * np.pi)
    
    def test_phase_principal_values(self):
        """Test phase for principal complex values."""
        # Positive real axis
        assert np.isclose(phase(1), 0)
        assert np.isclose(phase(2), 0)
        
        # Negative real axis
        assert np.isclose(phase(-1), np.pi)
        assert np.isclose(phase(-2), np.pi)
        
        # Positive imaginary axis
        assert np.isclose(phase(1j), np.pi/2)
        assert np.isclose(phase(2j), np.pi/2)
        
        # Negative imaginary axis
        assert np.isclose(phase(-1j), 3*np.pi/2)
        assert np.isclose(phase(-2j), 3*np.pi/2)
    
    def test_phase_quadrants(self):
        """Test phase in all four quadrants."""
        # First quadrant
        assert 0 < phase(1 + 1j) < np.pi/2
        
        # Second quadrant
        assert np.pi/2 < phase(-1 + 1j) < np.pi
        
        # Third quadrant
        assert np.pi < phase(-1 - 1j) < 3*np.pi/2
        
        # Fourth quadrant
        assert 3*np.pi/2 < phase(1 - 1j) < 2*np.pi
    
    def test_phase_array(self):
        """Test phase with array input."""
        z = np.array([1, -1, 1j, -1j])
        phi = phase(z)
        
        assert phi.shape == z.shape
        assert np.all(phi >= 0) and np.all(phi < 2*np.pi)
        
        # Check specific values
        assert np.isclose(phi[0], 0)        # phase(1)
        assert np.isclose(phi[1], np.pi)    # phase(-1)
        assert np.isclose(phi[2], np.pi/2)  # phase(1j)
        assert np.isclose(phi[3], 3*np.pi/2) # phase(-1j)
    
    def test_phase_branch_cut(self):
        """Test behavior along negative real axis (branch cut)."""
        # Just above negative real axis
        z_above = -1 + 1e-10j
        assert phase(z_above) < np.pi + 0.1
        
        # Just below negative real axis
        z_below = -1 - 1e-10j
        assert phase(z_below) > np.pi - 0.1
        
        # The phases should be close to π but on opposite sides of 2π
        assert abs(phase(z_above) - np.pi) < 0.1
        assert abs(phase(z_below) - np.pi) < 0.1
    
    def test_phase_zero(self):
        """Test phase at origin."""
        # Phase of zero is undefined but should return finite value
        phi = phase(0)
        assert np.isfinite(phi)
        assert 0 <= phi < 2*np.pi
    
    def test_phase_scalar_vs_array(self):
        """Test that phase handles both scalar and array inputs correctly."""
        # Scalar input
        z_scalar = 1 + 1j
        phi_scalar = phase(z_scalar)
        assert np.isscalar(phi_scalar)
        assert isinstance(phi_scalar, (float, np.floating))
        
        # Array input with single element
        z_array_1 = np.array([1 + 1j])
        phi_array_1 = phase(z_array_1)
        assert isinstance(phi_array_1, np.ndarray)
        assert phi_array_1.shape == (1,)
        assert np.isclose(phi_array_1[0], phi_scalar)
        
        # 2D array input
        z_array_2d = np.array([[1, -1], [1j, -1j]])
        phi_array_2d = phase(z_array_2d)
        assert isinstance(phi_array_2d, np.ndarray)
        assert phi_array_2d.shape == (2, 2)


class TestSawtoothFunction:
    """Test the sawtooth() function."""
    
    def test_sawtooth_basic(self):
        """Test basic sawtooth functionality."""
        # Integer inputs should give 0
        assert sawtooth(1) == 0
        assert sawtooth(2) == 0
        assert sawtooth(-1) == 0
        
        # Fractional inputs
        assert np.isclose(sawtooth(1.5), 0.5)
        assert np.isclose(sawtooth(2.25), 0.75)
        assert np.isclose(sawtooth(0.3), 0.7)
    
    def test_sawtooth_periodicity(self):
        """Test that sawtooth is periodic with period 1."""
        x = 0.7
        assert np.isclose(sawtooth(x), sawtooth(x + 1))
        assert np.isclose(sawtooth(x), sawtooth(x + 2))
        assert np.isclose(sawtooth(x), sawtooth(x - 1))
    
    def test_sawtooth_array(self):
        """Test sawtooth with array input."""
        x = np.array([0.1, 0.5, 1.0, 1.5, 2.9])
        result = sawtooth(x)
        
        assert result.shape == x.shape
        assert np.all(result >= 0) and np.all(result < 1)
        
        # Check specific values
        assert np.isclose(result[0], 0.9)  # sawtooth(0.1)
        assert np.isclose(result[1], 0.5)  # sawtooth(0.5)
        assert np.isclose(result[2], 0.0)  # sawtooth(1.0)
        assert np.isclose(result[3], 0.5)  # sawtooth(1.5)
        assert np.isclose(result[4], 0.1)  # sawtooth(2.9)
    
    def test_sawtooth_logarithmic(self):
        """Test sawtooth with logarithmic base."""
        # With log base 2
        x = np.array([1, 2, 4, 8])
        result = sawtooth(x, log_base=2)
        
        # Powers of 2 should give 0
        np.testing.assert_allclose(result, 0, atol=1e-10)
        
        # Test intermediate values
        x = np.array([1.5, 3, 6])
        result = sawtooth(x, log_base=2)
        assert np.all(result > 0) and np.all(result < 1)
    
    def test_sawtooth_special_values(self):
        """Test sawtooth with special values."""
        # Zero
        assert sawtooth(0) == 0
        
        # Negative values
        assert np.isclose(sawtooth(-0.3), 0.3)
        assert np.isclose(sawtooth(-1.7), 0.7)
        
        # With log_base and zero (should handle log(0) gracefully)
        result = sawtooth(0, log_base=2)
        assert np.isfinite(result) or np.isnan(result)


class TestStereographicFunction:
    """Test the stereographic() function."""
    
    def test_stereographic_unit_circle(self):
        """Test stereographic projection of unit circle."""
        # Points on unit circle should map to equator of sphere
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        for angle in angles:
            z = np.exp(1j * angle)
            x, y, z_coord = stereographic(z)
            
            # Check that point is on unit sphere
            assert np.isclose(x**2 + y**2 + z_coord**2, 1)
            
            # Check that z-coordinate is 0 (equator)
            assert np.isclose(z_coord, 0)
    
    def test_stereographic_origin(self):
        """Test stereographic projection of origin."""
        x, y, z = stereographic(0)
        
        # Origin should map to (0, 0, 1) with default projection from south
        # The formula gives z = -1, but project_from_north=False returns -z = 1
        assert np.isclose(x, 0)
        assert np.isclose(y, 0)
        assert np.isclose(z, 1)  # North pole with default projection from south
        
        # Test projection from north
        x, y, z = stereographic(0, project_from_north=True)
        assert np.isclose(x, 0)
        assert np.isclose(y, 0)
        assert np.isclose(z, -1)  # South pole when projecting from north
    
    def test_stereographic_infinity(self):
        """Test stereographic projection approaching infinity."""
        # Large values should approach south pole with default projection from south
        # The formula gives z approaching 1, but project_from_north=False returns -z = -1
        z_large = 1e10
        x, y, z = stereographic(z_large)
        
        assert np.abs(x) < 0.1  # x, y should be small
        assert np.abs(y) < 0.1
        assert z < -0.9  # z should be close to -1 (south pole)
    
    def test_stereographic_real_axis(self):
        """Test stereographic projection along real axis."""
        real_values = np.array([-2, -1, 0, 1, 2])
        
        for r in real_values:
            x, y, z = stereographic(r)
            
            # y-coordinate should be 0 (on xz-plane)
            assert np.isclose(y, 0)
            
            # Point should be on unit sphere
            assert np.isclose(x**2 + y**2 + z**2, 1)
    
    def test_stereographic_array(self):
        """Test stereographic projection with array input."""
        z = np.array([0, 1, -1, 1j, -1j])
        x, y, z_coord = stereographic(z)
        
        assert x.shape == z.shape
        assert y.shape == z.shape
        assert z_coord.shape == z.shape
        
        # All points should be on unit sphere
        for i in range(len(z)):
            assert np.isclose(x[i]**2 + y[i]**2 + z_coord[i]**2, 1)
    
    def test_stereographic_inverse_property(self):
        """Test inverse relationship of stereographic projection."""
        # For a point z, if (x,y,w) is its stereographic projection,
        # then z = (x + iy) / (1 + w) for default projection
        
        test_points = [0.5, -0.5, 0.5j, 1+1j, 2-1j]
        
        for z in test_points:
            x, y, w = stereographic(z)
            
            # Reconstruct z from projection
            if w != -1:  # Avoid division by zero (happens at infinity)
                z_reconstructed = (x + 1j*y) / (1 + w)
                np.testing.assert_allclose(z, z_reconstructed, rtol=1e-10)
    
    def test_stereographic_projection_direction(self):
        """Test projection from north vs south pole."""
        z = 1 + 1j
        
        # Project from south (default)
        x_s, y_s, z_s = stereographic(z, project_from_north=False)
        
        # Project from north
        x_n, y_n, z_n = stereographic(z, project_from_north=True)
        
        # x and y should be the same
        assert np.isclose(x_s, x_n)
        assert np.isclose(y_s, y_n)
        
        # z should have opposite sign
        assert np.isclose(z_s, -z_n)


class TestFunctionEdgeCases:
    """Test edge cases for all functions."""
    
    def test_phase_with_complex_array(self):
        """Test phase with 2D complex array."""
        z = np.array([[1, -1], [1j, -1j]])
        phi = phase(z)
        
        assert phi.shape == z.shape
        assert np.all(phi >= 0) and np.all(phi < 2*np.pi)
    
    def test_sawtooth_with_nan(self):
        """Test sawtooth with NaN values."""
        result = sawtooth(np.nan)
        assert np.isnan(result)
        
        # Array with NaN
        x = np.array([1, np.nan, 2])
        result = sawtooth(x)
        assert np.isfinite(result[0])
        assert np.isnan(result[1])
        assert np.isfinite(result[2])
    
    def test_stereographic_with_complex_infinity(self):
        """Test stereographic with complex infinity patterns."""
        # Different directions to infinity
        directions = [1, -1, 1j, -1j, 1+1j, 1-1j]
        
        for direction in directions:
            z = 1e10 * direction
            x, y, z_coord = stereographic(z)
            
            # Should be close to south pole with default projection
            assert np.sqrt(x**2 + y**2 + z_coord**2) <= 1.0001  # On sphere
            assert z_coord < -0.9  # Near south pole