"""Tests for domain classes."""

import numpy as np
import pytest
from complexplorer.core.domain import (
    Domain, Rectangle, Disk, Annulus, CompositeDomain
)
from complexplorer.utils.validation import ValidationError


class TestRectangle:
    """Test Rectangle domain."""
    
    def test_init_basic(self):
        """Test basic rectangle initialization."""
        rect = Rectangle(4, 2)
        
        assert rect.re_length == 4
        assert rect.im_length == 2
        assert rect.center == 0+0j
        assert rect.window_real == (-2, 2)
        # With square=True (default), window is made square
        assert rect.window_imag == (-2, 2)
    
    def test_init_with_center(self):
        """Test rectangle with custom center."""
        rect = Rectangle(2, 2, center=1+1j)
        
        assert rect.center == 1+1j
        assert rect.window_real == (0, 2)
        assert rect.window_imag == (0, 2)
    
    def test_init_square_window(self):
        """Test square viewing window constraint."""
        rect = Rectangle(4, 2, square=True)
        
        # Window should be square (4x4)
        assert rect.window_real == (-2, 2)
        assert rect.window_imag == (-2, 2)
    
    def test_init_non_square_window(self):
        """Test non-square viewing window."""
        rect = Rectangle(4, 2, square=False)
        
        # Window should match rectangle dimensions
        assert rect.window_real == (-2, 2)
        assert rect.window_imag == (-1, 1)
    
    def test_invalid_dimensions(self):
        """Test validation of dimensions."""
        with pytest.raises(ValidationError):
            Rectangle(0, 2)  # Zero width
        
        with pytest.raises(ValidationError):
            Rectangle(2, -1)  # Negative height
    
    def test_contains(self):
        """Test point containment."""
        rect = Rectangle(2, 2)
        
        # Test single points
        assert rect.contains(0+0j)
        assert rect.contains(0.9+0.9j)
        assert not rect.contains(2+0j)
        assert not rect.contains(0+2j)
        
        # Test array
        z = np.array([0+0j, 1+0j, 2+0j, 0+1j, 0+2j])
        mask = rect.contains(z)
        expected = np.array([True, True, False, True, False])
        np.testing.assert_array_equal(mask, expected)
    
    def test_mesh_generation(self):
        """Test mesh generation."""
        rect = Rectangle(2, 2)
        mesh = rect.mesh(n=3)
        
        # Should be 3x3 mesh
        assert mesh.shape == (3, 3)
        assert mesh[0, 0] == -1-1j
        assert mesh[2, 2] == 1+1j
    
    def test_inmask(self):
        """Test inside mask generation."""
        rect = Rectangle(2, 2)
        mask = rect.inmask(n=5)
        
        assert mask.shape == (5, 5)
        assert mask.all()  # All points should be inside
    
    def test_domain_with_nan(self):
        """Test domain mesh with NaN outside."""
        # Create a disk smaller than its viewing window
        disk = Disk(1)  # radius 1
        # Default window is (-1, 1) x (-1, 1)
        # But let's use a larger mesh
        
        domain_mesh = disk.domain(n=5)
        
        # Center should be valid
        center_idx = domain_mesh.shape[0] // 2
        assert not np.isnan(domain_mesh[center_idx, center_idx])
        
        # Corners should be NaN (outside radius 1)
        assert np.isnan(domain_mesh[0, 0])
        assert np.isnan(domain_mesh[-1, -1])


class TestDisk:
    """Test Disk domain."""
    
    def test_init_basic(self):
        """Test basic disk initialization."""
        disk = Disk(2)
        
        assert disk.radius == 2
        assert disk.center == 0+0j
        assert disk.window_real == (-2, 2)
        assert disk.window_imag == (-2, 2)
        assert disk.square is True  # Always square
    
    def test_init_with_center(self):
        """Test disk with custom center."""
        disk = Disk(1, center=2+3j)
        
        assert disk.center == 2+3j
        assert disk.window_real == (1, 3)
        assert disk.window_imag == (2, 4)
    
    def test_invalid_radius(self):
        """Test validation of radius."""
        with pytest.raises(ValidationError):
            Disk(0)  # Zero radius
        
        with pytest.raises(ValidationError):
            Disk(-1)  # Negative radius
    
    def test_contains(self):
        """Test point containment."""
        disk = Disk(1)
        
        # Test single points
        assert disk.contains(0+0j)
        assert disk.contains(0.7+0.7j)
        assert not disk.contains(1+1j)
        assert not disk.contains(2+0j)
        
        # Test array
        z = np.array([0+0j, 0.5+0j, 1+0j, 0+1j, 1+1j])
        mask = disk.contains(z)
        expected = np.array([True, True, True, True, False])
        np.testing.assert_array_equal(mask, expected)
    
    def test_contains_centered(self):
        """Test containment for centered disk."""
        disk = Disk(1, center=1+1j)
        
        assert disk.contains(1+1j)  # Center
        assert disk.contains(2+1j)  # Edge
        assert not disk.contains(0+0j)  # Outside


class TestAnnulus:
    """Test Annulus domain."""
    
    def test_init_basic(self):
        """Test basic annulus initialization."""
        ann = Annulus(1, 2)
        
        assert ann.inner_radius == 1
        assert ann.outer_radius == 2
        assert ann.center == 0+0j
        assert ann.window_real == (-2, 2)
        assert ann.window_imag == (-2, 2)
    
    def test_init_with_center(self):
        """Test annulus with custom center."""
        ann = Annulus(0.5, 1, center=1+2j)
        
        assert ann.center == 1+2j
        assert ann.window_real == (0, 2)
        assert ann.window_imag == (1, 3)
    
    def test_invalid_radii(self):
        """Test validation of radii."""
        with pytest.raises(ValidationError):
            Annulus(0, 1)  # Zero inner radius
        
        with pytest.raises(ValidationError):
            Annulus(-1, 2)  # Negative inner radius
        
        with pytest.raises(ValidationError):
            Annulus(2, 2)  # Equal radii
        
        with pytest.raises(ValidationError):
            Annulus(2, 1)  # Inner > outer
    
    def test_contains(self):
        """Test point containment."""
        ann = Annulus(1, 2)
        
        # Test single points
        assert not ann.contains(0+0j)  # Inside inner
        assert not ann.contains(0.5+0j)  # Inside inner
        assert ann.contains(1.5+0j)  # In annulus
        assert not ann.contains(3+0j)  # Outside outer
        
        # Test array
        z = np.array([0+0j, 1+0j, 1.5+0j, 2+0j, 3+0j])
        mask = ann.contains(z)
        expected = np.array([False, True, True, True, False])
        np.testing.assert_array_equal(mask, expected)


class TestCompositeDomain:
    """Test CompositeDomain for set operations."""
    
    def test_union_basic(self):
        """Test basic union operation."""
        disk1 = Disk(1, center=-1+0j)
        disk2 = Disk(1, center=1+0j)
        
        union = disk1.union(disk2)
        
        assert isinstance(union, CompositeDomain)
        assert union.operation == 'union'
        assert union.window_real == (-2, 2)
        # Both disks have square windows, so union is also square
        assert union.window_imag == (-2, 2)
    
    def test_union_contains(self):
        """Test containment for union."""
        disk1 = Disk(1, center=-1+0j)
        disk2 = Disk(1, center=1+0j)
        union = disk1 | disk2  # Using operator
        
        # Points in either disk
        assert union.contains(-1+0j)  # Center of disk1
        assert union.contains(1+0j)   # Center of disk2
        assert union.contains(0+0j)   # In both
        assert not union.contains(3+0j)  # In neither
    
    def test_intersection_basic(self):
        """Test basic intersection operation."""
        disk1 = Disk(2)
        rect = Rectangle(2, 2)
        
        intersection = disk1.intersection(rect)
        
        assert isinstance(intersection, CompositeDomain)
        assert intersection.operation == 'intersection'
    
    def test_intersection_contains(self):
        """Test containment for intersection."""
        disk = Disk(2)
        rect = Rectangle(2, 2, square=False)  # Non-square to get actual 2x2 rect
        intersection = disk & rect  # Using operator
        
        # Points in both domains
        assert intersection.contains(0+0j)
        assert intersection.contains(0.7+0.7j)
        
        # Point at corner of rect (1,1) is at distance sqrt(2) ≈ 1.41 from origin
        # This is inside disk of radius 2, so it's in the intersection
        assert intersection.contains(1+1j)
        
        # Point outside disk radius
        assert not intersection.contains(2.5+0j)
        
        # Points in neither
        assert not intersection.contains(3+3j)
    
    def test_invalid_operation(self):
        """Test invalid operation."""
        disk1 = Disk(1)
        disk2 = Disk(2)
        
        with pytest.raises(ValidationError):
            CompositeDomain(disk1, disk2, 'subtract')
    
    def test_nested_operations(self):
        """Test nested set operations."""
        disk1 = Disk(1, center=-1+0j)
        disk2 = Disk(1, center=1+0j)
        rect = Rectangle(4, 1)
        
        # (disk1 | disk2) & rect
        union = disk1 | disk2
        final = union & rect
        
        # Should contain points in either disk AND in rectangle
        assert final.contains(0+0j)
        assert not final.contains(0+2j)  # Outside rect


class TestDomainMethods:
    """Test common domain methods."""
    
    def test_spacing(self):
        """Test spacing calculation."""
        rect = Rectangle(4, 2)
        
        # For n=4, spacing should be 1 (4/4)
        assert rect.spacing(4) == 1.0
        
        # For n=8, spacing should be 0.5
        assert rect.spacing(8) == 0.5
    
    def test_mesh_shape(self):
        """Test mesh generation shape."""
        rect = Rectangle(4, 2, square=False)
        mesh = rect.mesh(n=10)
        
        # Should have 10 points on longer axis (real)
        assert mesh.shape[1] == 10
        # Shorter axis should have proportionally fewer
        assert mesh.shape[0] == 5
    
    def test_mesh_square_window(self):
        """Test mesh with square window."""
        rect = Rectangle(4, 2, square=True)
        mesh = rect.mesh(n=10)
        
        # Both dimensions should be equal for square window
        assert mesh.shape[0] == mesh.shape[1]
    
    def test_outmask(self):
        """Test outside mask generation."""
        disk = Disk(1)
        disk.window_real = (-2, 2)
        disk.window_imag = (-2, 2)
        
        outmask = disk.outmask(n=5)
        inmask = disk.inmask(n=5)
        
        # Masks should be complementary
        np.testing.assert_array_equal(outmask, ~inmask)
    
    def test_resolution_validation(self):
        """Test resolution parameter validation."""
        rect = Rectangle(2, 2)
        
        # Should work with valid resolution
        mesh = rect.mesh(n=100)
        assert mesh.size > 0
        
        # Edge cases
        mesh_small = rect.mesh(n=2)  # Minimum allowed
        assert mesh_small.size > 0
        
        # Invalid resolutions should raise errors
        with pytest.raises(ValidationError):
            rect.mesh(n=1)  # Too small
        
        with pytest.raises(ValidationError):
            rect.mesh(n=0)
        
        with pytest.raises(ValidationError):
            rect.mesh(n=-10)
        
        with pytest.raises(ValidationError):
            rect.mesh(n=10001)  # Too large


class TestBackwardCompatibility:
    """Test backward compatibility features."""
    
    def test_rectangle_compatibility(self):
        """Test Rectangle matches legacy behavior."""
        # New API with non-square window to match legacy
        rect_new = Rectangle(4, 2, center=1+1j, square=False)
        
        # Should have same bounds
        assert rect_new.window_real == (rect_new.center.real - 2, rect_new.center.real + 2)
        assert rect_new.window_imag == (rect_new.center.imag - 1, rect_new.center.imag + 1)
    
    def test_disk_compatibility(self):
        """Test Disk matches legacy behavior."""
        disk = Disk(2, center=1+1j)
        
        # Test contains matches legacy infunc
        z = np.array([1+1j, 3+1j, 1+3j])
        mask = disk.contains(z)
        expected = np.abs(z - disk.center) <= disk.radius
        np.testing.assert_array_equal(mask, expected)
    
    def test_annulus_compatibility(self):
        """Test Annulus matches legacy behavior."""
        ann = Annulus(1, 2, center=1+1j)
        
        # Test contains matches legacy infunc
        z = np.array([1+1j, 2+1j, 3+1j])
        mask = ann.contains(z)
        dist = np.abs(z - ann.center)
        expected = (dist >= ann.inner_radius) & (dist <= ann.outer_radius)
        np.testing.assert_array_equal(mask, expected)