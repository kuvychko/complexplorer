"""
Unit tests for complexplorer domain classes.
"""

import pytest
import numpy as np
from complexplorer import Domain, Rectangle, Disk, Annulus


class TestDomain:
    """Test the base Domain class."""
    
    def test_domain_creation(self):
        """Test Domain base class initialization."""
        # Create a simple domain using the base class
        domain = Domain(
            real=(-2, 2),
            imag=(-2, 2),
            infunc=lambda z: np.abs(z) < 1
        )
        
        assert domain.window_real == (-2, 2)
        assert domain.window_imag == (-2, 2)
        assert callable(domain.infunc)
    
    def test_domain_mesh_generation(self):
        """Test mesh generation for domains."""
        domain = Rectangle(real=4, imag=4, center=0)
        
        # Test different mesh sizes
        for n in [10, 50, 100]:
            mesh = domain.mesh(n)
            assert isinstance(mesh, np.ndarray)
            assert mesh.dtype == complex
            
            # Mesh shape depends on the spacing calculation
            assert mesh.ndim == 2
    
    def test_domain_inmask(self):
        """Test inmask generation."""
        domain = Disk(radius=1, center=0)
        mesh = domain.mesh(50)
        inmask = domain.inmask(50)
        
        assert inmask.shape == mesh.shape
        assert inmask.dtype == bool
        
        # Points inside disk should be True
        inside_points = mesh[inmask]
        assert np.all(np.abs(inside_points) <= 1.0001)  # Small tolerance
        
        # Points outside disk should be False
        outside_points = mesh[~inmask]
        assert np.all(np.abs(outside_points) >= 0.9999)  # Small tolerance
    
    def test_domain_outmask(self):
        """Test outmask generation."""
        domain = Disk(radius=1, center=0)
        inmask = domain.inmask(50)
        outmask = domain.outmask(50)
        
        # outmask should be complement of inmask
        np.testing.assert_array_equal(outmask, ~inmask)
    
    def test_domain_union(self):
        """Test union of two domains."""
        disk1 = Disk(radius=1, center=-0.5)
        disk2 = Disk(radius=1, center=0.5)
        
        union = disk1.union(disk2)
        
        # Test points that should be in the union
        assert union.infunc(0)      # Center (in both)
        assert union.infunc(-0.5)   # Center of disk1
        assert union.infunc(0.5)    # Center of disk2
        assert union.infunc(-1.4)   # Edge of disk1  
        assert union.infunc(1.4)    # Edge of disk2
        
        # Test points outside union
        assert not union.infunc(-2.5)  # Far left
        assert not union.infunc(2.5)   # Far right
    
    def test_domain_intersection(self):
        """Test intersection of two domains."""
        disk1 = Disk(radius=1, center=-0.5)
        disk2 = Disk(radius=1, center=0.5)
        
        intersection = disk1.intersection(disk2)
        
        # Test points in the intersection (lens shape)
        assert intersection.infunc(0)  # Center should be in both
        assert not intersection.infunc(-1.4)  # Far left not in disk2
        assert not intersection.infunc(1.4)   # Far right not in disk1


class TestRectangle:
    """Test the Rectangle domain class."""
    
    def test_rectangle_creation(self):
        """Test Rectangle initialization."""
        rect = Rectangle(real=4, imag=2, center=1+1j)
        
        # Check window values
        # With square=True (default), window is made square based on larger dimension
        assert rect.window_real == (-1, 3)  # center ± real/2
        # Window is expanded to be square
        assert rect.window_imag == (-1, 3)  # Made square
    
    def test_rectangle_contains(self):
        """Test point containment for rectangles."""
        rect = Rectangle(real=2, imag=2, center=0)
        
        # Test corners and center
        assert rect.infunc(0)      # Center
        assert rect.infunc(0.99 + 0.99j)   # Inside
        assert rect.infunc(-0.99 - 0.99j)  # Inside
        
        # Test outside points
        assert not rect.infunc(1.1 + 0j)   # Outside right
        assert not rect.infunc(0 + 1.1j)   # Outside top
        assert not rect.infunc(2 + 2j)     # Far outside
    
    def test_rectangle_edge_cases(self):
        """Test edge cases for rectangles."""
        # Very small but non-zero size
        rect1 = Rectangle(real=0.001, imag=1, square=False)
        assert abs(rect1.window_real[1] - rect1.window_real[0]) < 0.01
        
        # Negative sizes are allowed (takes absolute value)  
        rect2 = Rectangle(real=-2, imag=2, square=False)
        assert rect2.window_real == (-1, 1)
        
        # Test with square=False to avoid window expansion
        rect3 = Rectangle(real=4, imag=2, center=0, square=False)
        assert rect3.window_real == (-2, 2)
        assert rect3.window_imag == (-1, 1)


class TestDisk:
    """Test the Disk domain class."""
    
    def test_disk_creation(self):
        """Test Disk initialization."""
        disk = Disk(radius=2, center=1+1j)
        
        # Check window values
        assert disk.window_real == (-1, 3)  # center ± radius
        assert disk.window_imag == (-1, 3)  # center ± radius
    
    def test_disk_contains(self):
        """Test point containment for disks."""
        disk = Disk(radius=1, center=0)
        
        # Test points on axes
        assert disk.infunc(0)      # Center
        assert disk.infunc(0.5)    # Right of center
        assert disk.infunc(-0.5)   # Left of center
        assert disk.infunc(0.5j)   # Above center
        assert disk.infunc(-0.5j)  # Below center
        
        # Test boundary (with small tolerance)
        boundary_points = np.exp(2j * np.pi * np.linspace(0, 1, 8, endpoint=False))
        for z in boundary_points:
            assert disk.infunc(0.99 * z)   # Just inside
            assert not disk.infunc(1.01 * z)  # Just outside
    
    def test_disk_edge_cases(self):
        """Test edge cases for disks."""
        # Zero radius should raise error
        with pytest.raises(ValueError):
            Disk(radius=0, center=0)
        
        # Negative radius should raise error
        with pytest.raises(ValueError):
            Disk(radius=-1, center=0)


class TestAnnulus:
    """Test the Annulus domain class."""
    
    def test_annulus_creation(self):
        """Test Annulus initialization."""
        annulus = Annulus(radius_inner=1, radius_outer=2, center=0)
        
        # Check window values based on outer radius
        assert annulus.window_real == (-2, 2)
        assert annulus.window_imag == (-2, 2)
    
    def test_annulus_contains(self):
        """Test point containment for annuli."""
        annulus = Annulus(radius_inner=1, radius_outer=2, center=0)
        
        # Test points in annulus
        assert annulus.infunc(1.5)     # Right side
        assert annulus.infunc(-1.5)    # Left side
        assert annulus.infunc(1.5j)    # Top
        assert annulus.infunc(-1.5j)   # Bottom
        
        # Test points outside annulus
        assert not annulus.infunc(0)      # Center (inside inner radius)
        assert not annulus.infunc(0.5)    # Too close to center
        assert not annulus.infunc(2.5)    # Outside outer radius
        assert not annulus.infunc(3j)     # Far outside
    
    def test_annulus_edge_cases(self):
        """Test edge cases for annuli."""
        # Inner radius > outer radius should raise error
        with pytest.raises(ValueError):
            Annulus(radius_inner=2, radius_outer=1, center=0)
        
        # Equal radii should raise error (empty annulus)
        with pytest.raises(ValueError):
            Annulus(radius_inner=1, radius_outer=1, center=0)
        
        # Negative or zero inner radius should raise error
        with pytest.raises(ValueError):
            Annulus(radius_inner=-1, radius_outer=2, center=0)
        
        with pytest.raises(ValueError):
            Annulus(radius_inner=0, radius_outer=2, center=0)
    
    def test_annulus_valid_creation(self):
        """Test valid annulus creation."""
        # Small positive inner radius
        annulus = Annulus(radius_inner=0.1, radius_outer=1, center=0)
        
        # Should work correctly
        assert not annulus.infunc(0)      # Center (inside inner)
        assert annulus.infunc(0.5)        # In annulus
        assert not annulus.infunc(1.5)    # Outside


class TestDomainOperations:
    """Test domain operations and combinations."""
    
    def test_complex_union(self):
        """Test union of multiple domains."""
        rect = Rectangle(real=2, imag=2, center=-1)
        disk = Disk(radius=1, center=1)
        
        union = rect.union(disk)
        
        # Test points in rectangle only
        assert union.infunc(-1.9)
        
        # Test points in disk only  
        assert union.infunc(1.9)
        
        # Test points in neither
        assert not union.infunc(3j)
    
    def test_complex_intersection(self):
        """Test intersection with offset domains."""
        disk1 = Disk(radius=2, center=0)
        rect = Rectangle(real=2, imag=4, center=1)
        
        intersection = disk1.intersection(rect)
        
        # Points in both
        assert intersection.infunc(1)
        assert intersection.infunc(0.5 + 0.5j)
        
        # Points only in disk
        assert not intersection.infunc(-1.5)
        
        # Points only in rectangle
        assert not intersection.infunc(1.9 + 1.9j)
    
    def test_domain_mesh_with_operations(self):
        """Test meshing of combined domains."""
        disk1 = Disk(radius=1, center=-0.5)
        disk2 = Disk(radius=1, center=0.5)
        union = disk1.union(disk2)
        
        mesh = union.mesh(50)
        inmask = union.inmask(50)
        
        # Check that mesh and mask have consistent shapes
        assert mesh.shape == inmask.shape
        
        # Verify points are correctly classified
        inside_points = mesh[inmask]
        for z in inside_points.flat:
            assert disk1.infunc(z) or disk2.infunc(z)