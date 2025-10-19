"""Unit tests for stereographic projection correctness."""

import numpy as np
import pytest
from complexplorer.utils.mesh import sphere_to_complex, complex_to_sphere


class TestStereographicProjection:
    """Test stereographic projection mappings."""
    
    def test_south_pole_projection_origin(self):
        """Test that projection from south pole maps north pole to origin."""
        # When projecting from south pole, north pole should map to origin
        x, y, z = 0, 0, 1  # North pole
        w = sphere_to_complex(x, y, z, from_north=False)
        assert np.abs(w) < 1e-10, f"North pole should map to origin when projecting from south, got |w|={np.abs(w)}"
        
    def test_south_pole_projection_infinity(self):
        """Test that projection from south pole maps south pole to infinity."""
        # When projecting from south pole, south pole should map to infinity
        x, y, z = 0, 0, -1  # South pole
        w = sphere_to_complex(x, y, z, from_north=False)
        assert np.isinf(np.abs(w)), f"South pole should map to infinity when projecting from south, got w={w}"
    
    def test_north_pole_projection_origin(self):
        """Test that projection from north pole maps south pole to origin."""
        # When projecting from north pole, south pole should map to origin
        x, y, z = 0, 0, -1  # South pole
        w = sphere_to_complex(x, y, z, from_north=True)
        assert np.abs(w) < 1e-10, f"South pole should map to origin when projecting from north, got |w|={np.abs(w)}"
        
    def test_north_pole_projection_infinity(self):
        """Test that projection from north pole maps north pole to infinity."""
        # When projecting from north pole, north pole should map to infinity
        x, y, z = 0, 0, 1  # North pole
        w = sphere_to_complex(x, y, z, from_north=True)
        assert np.isinf(np.abs(w)), f"North pole should map to infinity when projecting from north, got w={w}"
    
    def test_equator_maps_to_unit_circle(self):
        """Test that equator maps to unit circle for both projections."""
        # Points on equator (z=0) should map to unit circle
        theta = np.linspace(0, 2*np.pi, 8)
        for angle in theta:
            x = np.cos(angle)
            y = np.sin(angle)
            z = 0
            
            # From north pole
            w_north = sphere_to_complex(x, y, z, from_north=True)
            assert np.abs(np.abs(w_north) - 1) < 1e-10, \
                f"Equator should map to unit circle from north, got |w|={np.abs(w_north)}"
            
            # From south pole
            w_south = sphere_to_complex(x, y, z, from_north=False)
            assert np.abs(np.abs(w_south) - 1) < 1e-10, \
                f"Equator should map to unit circle from south, got |w|={np.abs(w_south)}"
    
    def test_hemisphere_mapping_from_south(self):
        """Test that northern hemisphere maps to |w| < 1 when projecting from south pole."""
        # Points in northern hemisphere (z > 0) should map inside unit circle
        # when projecting from south pole
        test_points = [
            (0.5, 0, np.sqrt(0.75)),    # Point in northern hemisphere
            (0, 0.5, np.sqrt(0.75)),    # Another point in northern hemisphere
            (0.3, 0.4, np.sqrt(0.75)),  # Another point in northern hemisphere
        ]
        
        for x, y, z in test_points:
            w = sphere_to_complex(x, y, z, from_north=False)
            assert np.abs(w) < 1, \
                f"Northern hemisphere (z={z}) should map inside unit circle from south, got |w|={np.abs(w)}"
    
    def test_hemisphere_mapping_from_north(self):
        """Test that southern hemisphere maps to |w| < 1 when projecting from north pole."""
        # Points in southern hemisphere (z < 0) should map inside unit circle
        # when projecting from north pole
        test_points = [
            (0.5, 0, -np.sqrt(0.75)),    # Point in southern hemisphere
            (0, 0.5, -np.sqrt(0.75)),    # Another point in southern hemisphere
            (0.3, 0.4, -np.sqrt(0.75)),  # Another point in southern hemisphere
        ]
        
        for x, y, z in test_points:
            w = sphere_to_complex(x, y, z, from_north=True)
            assert np.abs(w) < 1, \
                f"Southern hemisphere (z={z}) should map inside unit circle from north, got |w|={np.abs(w)}"
    
    def test_inverse_projection_consistency(self):
        """Test that inverse projection is consistent with forward projection."""
        # Test various complex points
        test_points = [
            0 + 0j,          # Origin
            1 + 0j,          # Real axis
            0 + 1j,          # Imaginary axis
            0.5 + 0.5j,      # First quadrant
            -0.5 + 0.5j,     # Second quadrant
            2 + 3j,          # Outside unit circle
        ]
        
        for w in test_points:
            # Test from north pole
            x, y, z = complex_to_sphere(w, to_north=True)
            w_recovered = sphere_to_complex(x, y, z, from_north=True)
            assert np.abs(w - w_recovered) < 1e-10, \
                f"Inverse projection from north not consistent: {w} != {w_recovered}"
            
            # Test from south pole  
            x, y, z = complex_to_sphere(w, to_north=False)
            w_recovered = sphere_to_complex(x, y, z, from_north=False)
            assert np.abs(w - w_recovered) < 1e-10, \
                f"Inverse projection from south not consistent: {w} != {w_recovered}"
    
    def test_phase_preservation(self):
        """Test that phase (argument) is preserved in stereographic projection."""
        # Points at same angle should maintain that angle after projection
        r_values = [0.5, 1.0, 2.0]  # Different radii
        theta_values = [0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2]  # Different angles
        
        for r in r_values:
            for theta in theta_values:
                w = r * np.exp(1j * theta)
                
                # Project to sphere and back (from south pole)
                x, y, z = complex_to_sphere(w, to_north=False)
                w_proj = sphere_to_complex(x, y, z, from_north=False)
                
                # Check phase is preserved (modulo numerical errors)
                if not np.isinf(w_proj):
                    phase_original = np.angle(w)
                    phase_projected = np.angle(w_proj)
                    phase_diff = np.abs(phase_original - phase_projected)
                    # Handle wraparound at ±π
                    if phase_diff > np.pi:
                        phase_diff = 2*np.pi - phase_diff
                    assert phase_diff < 1e-10, \
                        f"Phase not preserved: original={phase_original}, projected={phase_projected}"


class TestRiemannSphereOrientation:
    """Test Riemann sphere orientation in visualization."""
    
    def test_riemann_pv_uses_correct_projection(self):
        """Verify that riemann_pv uses projection from north pole (standard convention)."""
        # This test verifies the expected behavior in the code
        # riemann_pv should use from_north=True to project from north pole
        # This is the standard convention where:
        # - Origin of complex plane -> south pole of sphere (bottom)
        # - Infinity in complex plane -> north pole of sphere (top)
        
        # Check that the projection is called with from_north=True
        from complexplorer.plotting.pyvista.riemann import riemann_pv
        import inspect
        
        source = inspect.getsource(riemann_pv)
        assert "from_north=True" in source, \
            "riemann_pv should use projection from north pole (from_north=True)"
    
    def test_complex_plane_orientation(self):
        """Test that complex plane orientation matches expected convention."""
        # Standard convention for Riemann sphere (projecting from north pole):
        # - z=0 (origin) should appear at south pole
        # - z=∞ should appear at north pole
        # - Positive real axis points in +x direction
        # - Positive imaginary axis points in +y direction
        
        # Test key points
        test_cases = [
            (0+0j, "south pole (bottom)"),
            (1+0j, "positive real axis on equator"),
            (0+1j, "positive imaginary axis on equator"),
            (-1+0j, "negative real axis on equator"),
            (0-1j, "negative imaginary axis on equator"),
        ]
        
        for z, description in test_cases:
            # When projecting from north pole (standard convention)
            x, y, z_sphere = complex_to_sphere(z, to_north=True)
            
            if z == 0:
                # Origin should map to south pole (z=-1)
                assert np.abs(z_sphere + 1) < 1e-10, \
                    f"Origin should map to south pole, got z={z_sphere}"
            elif np.abs(z) == 1:
                # Unit circle should map to equator (z=0)
                assert np.abs(z_sphere) < 1e-10, \
                    f"Unit circle should map to equator, got z={z_sphere} for {description}"
    
    def test_small_values_near_south_pole(self):
        """Test that small complex values appear near south pole of sphere."""
        # Small values should be close to south pole when projecting from north
        small_values = [0.01+0j, 0+0.01j, 0.01+0.01j, -0.01+0j, 0-0.01j]
        
        for w in small_values:
            x, y, z = complex_to_sphere(w, to_north=True)
            # Should be in lower hemisphere and close to south pole
            assert z < -0.9, f"Small value {w} should map to lower hemisphere, got z={z}"
            assert np.sqrt(x**2 + y**2) < 0.2, \
                f"Small value {w} should be near south pole axis, got x={x}, y={y}"
    
    def test_large_values_near_north_pole(self):
        """Test that large complex values appear near north pole of sphere."""
        # Large values should be close to north pole when projecting from north
        large_values = [10+0j, 0+10j, 10+10j, -10+0j, 0-10j]
        
        for w in large_values:
            x, y, z = complex_to_sphere(w, to_north=True)
            # Should be in upper hemisphere and close to north pole
            assert z > 0.9, f"Large value {w} should map to upper hemisphere, got z={z}"
            assert np.sqrt(x**2 + y**2) < 0.2, \
                f"Large value {w} should be near north pole axis, got x={x}, y={y}"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])