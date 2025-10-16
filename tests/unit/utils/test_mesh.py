"""Tests for mesh generation utilities."""

import numpy as np
import pytest
from complexplorer.utils.mesh import (
    RectangularSphereGenerator, sphere_to_complex, complex_to_sphere,
    HAS_PYVISTA
)
from complexplorer.core.domain import Disk


class TestSphereToComplex:
    """Test sphere to complex plane projection."""
    
    def test_basic_points_from_north(self):
        """Test projection of basic points from north pole."""
        # North pole itself
        w = sphere_to_complex(0, 0, 1, from_north=True)
        assert np.abs(w) > 1e9  # Should be very large (near infinity)
        
        # South pole
        w = sphere_to_complex(0, 0, -1, from_north=True)
        assert np.abs(w) < 1e-10  # Should be origin
        
        # Equator points
        w = sphere_to_complex(1, 0, 0, from_north=True)
        assert np.abs(w - 1) < 1e-10
        
        w = sphere_to_complex(0, 1, 0, from_north=True)
        assert np.abs(w - 1j) < 1e-10
        
        w = sphere_to_complex(-1, 0, 0, from_north=True)
        assert np.abs(w - (-1)) < 1e-10
    
    def test_basic_points_from_south(self):
        """Test projection of basic points from south pole."""
        # South pole itself
        w = sphere_to_complex(0, 0, -1, from_north=False)
        assert np.abs(w) > 1e9  # Should be very large (near infinity)
        
        # North pole
        w = sphere_to_complex(0, 0, 1, from_north=False)
        assert np.abs(w) < 1e-10  # Should be origin
        
        # Equator points
        w = sphere_to_complex(1, 0, 0, from_north=False)
        assert np.abs(w - 1) < 1e-10
    
    def test_array_input(self):
        """Test with array inputs."""
        x = np.array([1, 0, -1, 0])
        y = np.array([0, 1, 0, -1])
        z = np.array([0, 0, 0, 0])
        
        w = sphere_to_complex(x, y, z)
        
        assert len(w) == 4
        # All equator points should map to unit circle
        assert np.allclose(np.abs(w), 1.0)
    
    def test_pole_avoidance(self):
        """Test handling of near-pole points."""
        # Very close to north pole
        w = sphere_to_complex(0, 0, 0.99999, from_north=True)
        assert np.isfinite(w)  # Should not be inf or nan
        
        # Very close to south pole
        w = sphere_to_complex(0, 0, -0.99999, from_north=False)
        assert np.isfinite(w)


class TestComplexToSphere:
    """Test complex plane to sphere projection."""
    
    def test_basic_points_to_north(self):
        """Test inverse projection to north pole projection."""
        # Origin maps to south pole
        x, y, z = complex_to_sphere(0+0j, to_north=True)
        assert abs(x) < 1e-10
        assert abs(y) < 1e-10
        assert abs(z - (-1)) < 1e-10
        
        # Unit circle to equator
        x, y, z = complex_to_sphere(1+0j, to_north=True)
        assert abs(x - 1) < 1e-10
        assert abs(y) < 1e-10
        assert abs(z) < 1e-10
        
        x, y, z = complex_to_sphere(0+1j, to_north=True)
        assert abs(x) < 1e-10
        assert abs(y - 1) < 1e-10
        assert abs(z) < 1e-10
    
    def test_basic_points_to_south(self):
        """Test inverse projection to south pole projection."""
        # Origin maps to north pole
        x, y, z = complex_to_sphere(0+0j, to_north=False)
        assert abs(x) < 1e-10
        assert abs(y) < 1e-10
        assert abs(z - 1) < 1e-10
    
    def test_array_input(self):
        """Test with array inputs."""
        w = np.array([0+0j, 1+0j, 0+1j, 1+1j])
        x, y, z = complex_to_sphere(w)
        
        assert len(x) == 4
        assert len(y) == 4
        assert len(z) == 4
        
        # Check all points are on unit sphere
        radii = np.sqrt(x**2 + y**2 + z**2)
        assert np.allclose(radii, 1.0)
    
    def test_round_trip(self):
        """Test round trip projection."""
        # Start with points on sphere
        x0 = np.array([0.6, 0.8, 0.0])
        y0 = np.array([0.8, 0.0, 0.6])
        z0 = np.array([0.0, 0.6, 0.8])
        
        # Project to plane and back
        w = sphere_to_complex(x0, y0, z0, from_north=True)
        x1, y1, z1 = complex_to_sphere(w, to_north=True)
        
        assert np.allclose(x0, x1)
        assert np.allclose(y0, y1)
        assert np.allclose(z0, z1)


@pytest.mark.skipif(not HAS_PYVISTA, reason="PyVista not installed")
class TestRectangularSphereGenerator:
    """Test rectangular sphere mesh generation."""
    
    def test_basic_generation(self):
        """Test basic sphere generation."""
        gen = RectangularSphereGenerator(radius=2.0, n_theta=20, n_phi=30)
        mesh = gen.generate()
        
        # Check mesh properties
        assert mesh.n_points > 0
        assert mesh.n_cells > 0
        
        # Check radius
        points = mesh.points
        radii = np.sqrt(np.sum(points**2, axis=1))
        assert np.allclose(radii, 2.0, rtol=0.01)
    
    def test_pole_avoidance(self):
        """Test pole avoidance option."""
        gen = RectangularSphereGenerator(n_theta=10, n_phi=10, avoid_poles=True)
        mesh = gen.generate()
        
        # Check no points exactly at poles
        z_coords = mesh.points[:, 2]
        assert np.all(np.abs(z_coords) < 1.0)  # Should be strictly less than 1
    
    def test_with_domain(self):
        """Test generation with domain constraint."""
        # Create a disk domain
        domain = Disk(radius=2.0)
        
        gen = RectangularSphereGenerator(n_theta=20, n_phi=20, domain=domain)
        mesh = gen.generate()
        
        # Project all points to complex plane
        points = mesh.points
        w = sphere_to_complex(points[:, 0], points[:, 1], points[:, 2])
        
        # Most points should be within domain (some boundary effects expected)
        in_domain = domain.contains(w)
        assert np.sum(in_domain) > 0.7 * len(in_domain)


@pytest.mark.skipif(HAS_PYVISTA, reason="PyVista is installed")
class TestWithoutPyVista:
    """Test behavior without PyVista."""
    
    def test_import_error(self):
        """Test that appropriate error is raised."""
        gen = RectangularSphereGenerator()
        
        with pytest.raises(ImportError, match="PyVista is required"):
            gen.generate()