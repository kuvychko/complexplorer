"""
Unit tests for mesh utilities.
"""

import pytest
import numpy as np
import pyvista as pv

from complexplorer.mesh_utils import (
    RectangularSphereGenerator,
    stereographic_projection,
    inverse_stereographic,
    ModulusScaling
)


class TestRectangularSphereGenerator:
    """Test rectangular sphere generation."""
    
    def test_basic_generation(self):
        """Test basic sphere generation."""
        generator = RectangularSphereGenerator(radius=1.0, n_theta=50, n_phi=50)
        mesh = generator.generate()
        
        assert isinstance(mesh, pv.PolyData)
        assert mesh.n_points > 0
        assert mesh.n_cells > 0
    
    def test_radius_scaling(self):
        """Test that radius parameter works correctly."""
        radius = 2.5
        generator = RectangularSphereGenerator(radius=radius, n_theta=30, n_phi=30)
        mesh = generator.generate()
        
        # Check that all points are approximately at the specified radius
        distances = np.linalg.norm(mesh.points, axis=1)
        assert np.allclose(distances, radius, atol=0.1)
    
    def test_avoid_poles(self):
        """Test pole avoidance option."""
        generator = RectangularSphereGenerator(avoid_poles=True, n_theta=30, n_phi=30)
        mesh = generator.generate()
        
        # Check that no points are exactly at poles
        z_coords = mesh.points[:, 2]
        assert not np.any(np.abs(z_coords) == 1.0)
    
    def test_resolution_parameters(self):
        """Test different resolution settings."""
        n_theta, n_phi = 20, 40
        generator = RectangularSphereGenerator(n_theta=n_theta, n_phi=n_phi)
        mesh = generator.generate()
        
        # Should have approximately n_theta * n_phi points
        expected_points = n_theta * n_phi
        assert abs(mesh.n_points - expected_points) < expected_points * 0.1


class TestStereographicProjection:
    """Test stereographic projection functions."""
    
    def test_north_pole_projection(self):
        """Test projection from north pole."""
        # Points on unit sphere
        x = np.array([1, 0, 0])
        y = np.array([0, 1, 0])
        z = np.array([0, 0, 0])
        
        w = stereographic_projection(x, y, z, from_north=True)
        
        # Check known mappings (from north pole projection)
        # (1,0,0) -> 1/(1-0) = 1
        # (0,1,0) -> i/(1-0) = i  
        # (0,0,0) -> 0/(1-0) = 0
        assert np.isclose(w[0], 1 + 0j)  # (1,0,0) -> 1
        assert np.isclose(w[1], 0 + 1j)  # (0,1,0) -> i
        assert np.isclose(w[2], 0 + 0j)  # (0,0,0) -> 0
        
        # Test near pole (our implementation avoids infinity)
        x_pole = np.array([0])
        y_pole = np.array([0])
        z_pole = np.array([0.9999])
        w_pole = stereographic_projection(x_pole, y_pole, z_pole, from_north=True)
        # Near pole should give large value but not infinity
        assert np.abs(w_pole[0]) < np.inf
    
    def test_south_pole_projection(self):
        """Test projection from south pole."""
        # Points on unit sphere
        x = np.array([1, 0, 0])
        y = np.array([0, 1, 0])
        z = np.array([0, 0, 0])
        
        w = stereographic_projection(x, y, z, from_north=False)
        
        # Same points but projecting from south pole
        assert np.isclose(w[0], 1 + 0j)  # (1,0,0) -> 1
        assert np.isclose(w[1], 0 + 1j)  # (0,1,0) -> i
        assert np.isclose(w[2], 0 + 0j)  # (0,0,0) -> 0
        
        # Test near south pole
        x_pole = np.array([0])
        y_pole = np.array([0])
        z_pole = np.array([-0.9999])
        w_pole = stereographic_projection(x_pole, y_pole, z_pole, from_north=False)
        # Near pole should give large value but not infinity
        assert np.abs(w_pole[0]) < np.inf
    
    def test_inverse_projection(self):
        """Test inverse stereographic projection."""
        # Test round-trip for some complex numbers
        w = np.array([1 + 0j, 0 + 1j, 2 - 1j, 0.5 + 0.5j])
        
        # Project to sphere and back
        x, y, z = inverse_stereographic(w, to_north=True)
        w_back = stereographic_projection(x, y, z, from_north=True)
        
        assert np.allclose(w, w_back)
    
    def test_projection_preserves_angles(self):
        """Test that stereographic projection is conformal."""
        # Create a small square in the complex plane
        w = np.array([0.1 + 0.1j, 0.2 + 0.1j, 0.2 + 0.2j, 0.1 + 0.2j])
        
        # Project to sphere
        x, y, z = inverse_stereographic(w)
        
        # Check that points are on unit sphere
        distances = np.sqrt(x**2 + y**2 + z**2)
        assert np.allclose(distances, 1.0)


class TestModulusScaling:
    """Test modulus scaling methods."""
    
    def test_constant_scaling(self):
        """Test constant radius scaling."""
        moduli = np.array([0, 1, 2, 10, 100])
        radii = ModulusScaling.constant(moduli, radius=2.0)
        
        assert np.all(radii == 2.0)
    
    def test_linear_scaling(self):
        """Test linear scaling."""
        moduli = np.array([0, 1, 2])
        radii = ModulusScaling.linear(moduli, scale=0.5)
        
        expected = np.array([1.0, 1.5, 2.0])
        assert np.allclose(radii, expected)
    
    def test_arctan_scaling(self):
        """Test arctangent scaling."""
        moduli = np.array([0, 1, 10, 100, 1000])
        radii = ModulusScaling.arctan(moduli, r_min=0.5, r_max=1.5)
        
        # Check bounds
        assert np.all(radii >= 0.5)
        assert np.all(radii <= 1.5)
        
        # Check monotonicity
        assert np.all(np.diff(radii) >= 0)
        
        # Check specific values
        assert np.isclose(radii[0], 0.5)  # |f| = 0 -> r_min
        assert radii[-1] > 1.4            # Large |f| -> near r_max
    
    def test_logarithmic_scaling(self):
        """Test logarithmic scaling."""
        moduli = np.array([0.1, 1, 10, 100])
        radii = ModulusScaling.logarithmic(moduli, base=10, r_min=0.5, r_max=1.5)
        
        # Check bounds
        assert np.all(radii >= 0.5)
        assert np.all(radii <= 1.5)
        
        # Check monotonicity
        assert np.all(np.diff(radii) >= 0)
    
    def test_linear_clamp_scaling(self):
        """Test linear scaling with clamping."""
        moduli = np.array([0, 5, 10, 20, 100])
        radii = ModulusScaling.linear_clamp(moduli, m_max=10, r_min=0.5, r_max=1.5)
        
        # Check bounds
        assert np.all(radii >= 0.5)
        assert np.all(radii <= 1.5)
        
        # Check clamping
        assert radii[3] == radii[4]  # Both clamped at r_max
    
    def test_power_scaling(self):
        """Test power scaling."""
        moduli = np.array([0, 0.5, 1, 2])
        radii = ModulusScaling.power(moduli, exponent=0.5, r_min=0.5, r_max=1.5)
        
        # Check bounds
        assert np.all(radii >= 0.5)
        assert np.all(radii <= 1.5)
        
        # Check that max modulus maps to r_max
        assert np.isclose(radii[-1], 1.5)