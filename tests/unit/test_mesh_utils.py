"""
Unit tests for mesh utilities.
"""

import pytest
import numpy as np
import pyvista as pv

from complexplorer.mesh_utils import (
    IcosphereGenerator,
    IcosphereData,
    stereographic_projection,
    inverse_stereographic,
    ModulusScaling
)


class TestIcosphereGenerator:
    """Test icosahedral sphere generation."""
    
    def test_basic_generation(self):
        """Test basic icosphere generation."""
        generator = IcosphereGenerator(radius=1.0, subdivisions=0)
        mesh = generator.generate()
        
        assert isinstance(mesh, pv.PolyData)
        assert mesh.n_points == 12  # Base icosahedron
        assert mesh.n_cells == 20   # 20 triangular faces
    
    def test_subdivision_counts(self):
        """Test vertex and face counts after subdivision."""
        # Formula: V = 10 * 4^n + 2, F = 20 * 4^n
        test_cases = [
            (0, 12, 20),      # Base icosahedron
            (1, 42, 80),      # 1 subdivision
            (2, 162, 320),    # 2 subdivisions
            (3, 642, 1280),   # 3 subdivisions
        ]
        
        for subdivs, expected_v, expected_f in test_cases:
            generator = IcosphereGenerator(subdivisions=subdivs)
            mesh = generator.generate()
            assert mesh.n_points == expected_v
            assert mesh.n_cells == expected_f
    
    def test_radius_scaling(self):
        """Test that radius parameter works correctly."""
        radius = 2.5
        generator = IcosphereGenerator(radius=radius, subdivisions=1)
        mesh = generator.generate()
        
        # Check that all points are at the specified radius
        distances = np.linalg.norm(mesh.points, axis=1)
        assert np.allclose(distances, radius)
    
    def test_unit_sphere_normalization(self):
        """Test that vertices lie on unit sphere before scaling."""
        generator = IcosphereGenerator(radius=1.0, subdivisions=2)
        data = generator.get_data()
        
        # Internal vertices should be on unit sphere
        distances = np.linalg.norm(data.vertices, axis=1)
        assert np.allclose(distances, 1.0)
    
    def test_subdivision_limit(self):
        """Test that subdivision level is limited."""
        generator = IcosphereGenerator(subdivisions=10)  # Too high
        assert generator.subdivisions == 8  # Should be clamped


class TestStereographicProjection:
    """Test stereographic projection functions."""
    
    def test_north_pole_projection(self):
        """Test projection from north pole."""
        # Points on sphere
        x = np.array([1, 0, 0, 0])
        y = np.array([0, 1, 0, 0])
        z = np.array([0, 0, 0, 0.999])
        
        w = stereographic_projection(x, y, z, from_north=True)
        
        # Check known mappings
        assert np.isclose(w[0], 1 + 0j)  # (1,0,0) -> 1
        assert np.isclose(w[1], 0 + 1j)  # (0,1,0) -> i
        assert np.isclose(w[2], 0 + 0j)  # (0,0,0) -> 0
        assert np.isinf(w[3])            # Near north pole -> infinity
    
    def test_south_pole_projection(self):
        """Test projection from south pole."""
        x = np.array([1, 0, 0])
        y = np.array([0, 1, 0])
        z = np.array([0, 0, -0.999])
        
        w = stereographic_projection(x, y, z, from_north=False)
        
        # Near south pole should map to infinity
        assert np.isinf(w[2])
    
    def test_inverse_projection(self):
        """Test inverse stereographic projection."""
        # Test points
        w = np.array([0 + 0j, 1 + 0j, 0 + 1j, 2 + 2j])
        
        x, y, z = inverse_stereographic(w, to_north=True)
        
        # Check some properties
        # Origin maps to south pole
        assert np.isclose(z[0], -1)
        
        # All points should be on unit sphere
        distances = np.sqrt(x**2 + y**2 + z**2)
        assert np.allclose(distances, 1.0)
    
    def test_projection_inverse_consistency(self):
        """Test that projection and inverse are consistent."""
        # Random points on sphere (avoiding poles)
        theta = np.random.uniform(0.1, np.pi - 0.1, 10)
        phi = np.random.uniform(0, 2 * np.pi, 10)
        
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        
        # Project to plane and back
        w = stereographic_projection(x, y, z, from_north=True)
        x2, y2, z2 = inverse_stereographic(w, to_north=True)
        
        # Should recover original points
        assert np.allclose(x, x2, atol=1e-10)
        assert np.allclose(y, y2, atol=1e-10)
        assert np.allclose(z, z2, atol=1e-10)


class TestModulusScaling:
    """Test modulus scaling functions."""
    
    def test_constant_scaling(self):
        """Test constant radius scaling."""
        moduli = np.array([0, 1, 5, 10, np.inf])
        radius = 2.0
        
        scaled = ModulusScaling.constant(moduli, radius)
        
        assert np.all(scaled == radius)
    
    def test_arctan_scaling(self):
        """Test arctan scaling properties."""
        moduli = np.array([0, 1, 10, 100, 1000])
        r_min, r_max = 0.2, 1.0
        
        scaled = ModulusScaling.arctan(moduli, r_min, r_max)
        
        # Check bounds
        assert np.all(scaled >= r_min)
        assert np.all(scaled <= r_max)
        
        # Check monotonicity
        assert np.all(np.diff(scaled) > 0)
        
        # Check specific values
        assert np.isclose(scaled[0], r_min)  # 0 maps to r_min
        # Large values approach r_max
        assert scaled[-1] < r_max
        assert scaled[-1] > 0.9 * r_max
    
    def test_logarithmic_scaling(self):
        """Test logarithmic scaling."""
        moduli = np.array([0.1, 1, 10, 100])
        
        scaled = ModulusScaling.logarithmic(moduli)
        
        # Check bounds
        assert np.all(scaled >= 0.2)
        assert np.all(scaled <= 1.0)
        
        # Check that it handles zero properly
        zero_scaled = ModulusScaling.logarithmic(np.array([0]))
        assert np.isfinite(zero_scaled[0])
    
    def test_linear_clamp_scaling(self):
        """Test linear clamping."""
        moduli = np.array([0, 5, 10, 15, 20])
        m_max = 10
        r_min, r_max = 0.2, 1.0
        
        scaled = ModulusScaling.linear_clamp(moduli, m_max, r_min, r_max)
        
        # Check mapping
        assert np.isclose(scaled[0], r_min)  # 0 -> r_min
        assert np.isclose(scaled[1], 0.6)    # 5 -> midpoint
        assert np.isclose(scaled[2], r_max)  # 10 -> r_max
        assert np.isclose(scaled[3], r_max)  # 15 -> clamped to r_max
        assert np.isclose(scaled[4], r_max)  # 20 -> clamped to r_max
    
    def test_custom_scaling(self):
        """Test custom scaling function."""
        moduli = np.array([0, 1, 2, 3, 4])
        
        # Custom function: square root scaling
        def sqrt_scale(x):
            return np.sqrt(x) / 2  # Maps [0, 4] to [0, 1]
        
        scaled = ModulusScaling.custom(moduli, sqrt_scale, 0.3, 0.8)
        
        # Check bounds
        assert np.all(scaled >= 0.3)
        assert np.all(scaled <= 0.8)
        
        # Check that it follows the pattern
        expected = 0.3 + (0.8 - 0.3) * sqrt_scale(moduli)
        assert np.allclose(scaled, expected)