"""Tests for STL export utility functions."""

import numpy as np
import pytest

# Check if PyVista is available
try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

from complexplorer.export.stl.utils import (
    check_pyvista_available, validate_printability,
    scale_to_size, center_mesh
)


class TestCheckPyVistaAvailable:
    """Test PyVista availability check."""
    
    @pytest.mark.skipif(HAS_PYVISTA, reason="PyVista is installed")
    def test_raises_when_not_available(self):
        """Test error when PyVista not installed."""
        with pytest.raises(ImportError, match="PyVista is required"):
            check_pyvista_available()
    
    @pytest.mark.skipif(not HAS_PYVISTA, reason="PyVista not installed")
    def test_passes_when_available(self):
        """Test no error when PyVista installed."""
        check_pyvista_available()  # Should not raise


@pytest.mark.skipif(not HAS_PYVISTA, reason="PyVista not installed")
class TestValidatePrintability:
    """Test mesh validation for 3D printing."""
    
    def test_watertight_sphere(self):
        """Test validation of watertight sphere."""
        sphere = pv.Sphere(radius=1.0)
        
        results = validate_printability(sphere, verbose=False)
        
        assert results['is_watertight'] is True
        assert results['is_manifold'] is True
        assert results['n_boundary_edges'] == 0
        assert results['n_non_manifold_edges'] == 0
        assert results['volume'] > 0
        assert results['surface_area'] > 0
    
    def test_open_mesh(self):
        """Test validation of mesh with holes."""
        # Create cylinder without caps (open mesh)
        cylinder = pv.Cylinder(height=2.0, radius=0.5, capping=False)
        
        results = validate_printability(cylinder, verbose=False)
        
        assert results['is_watertight'] is False
        assert results['n_boundary_edges'] > 0
    
    def test_size_validation(self):
        """Test size-based validation."""
        sphere = pv.Sphere(radius=1.0)  # 2 units diameter
        
        # Test with small size (should fail wall thickness)
        results = validate_printability(sphere, size_mm=10, verbose=False)
        
        assert 'wall_thickness_ok' in results
        assert 'estimated_min_wall_mm' in results
        assert 'recommended_size_mm' in results
        
        # For a 10mm sphere with 30% minimum radius, wall is ~3mm at thinnest
        # This should be OK
        assert results['wall_thickness_ok'] is True
        
        # Test with very small size
        results = validate_printability(sphere, size_mm=2, verbose=False)
        assert results['wall_thickness_ok'] is False
        assert results['recommended_size_mm'] > 2
    
    def test_verbose_output(self, capsys):
        """Test verbose output."""
        sphere = pv.Sphere()
        
        validate_printability(sphere, size_mm=50, verbose=True)
        
        captured = capsys.readouterr()
        assert "Mesh Validation Results" in captured.out
        assert "Watertight: True" in captured.out
        assert "ready for 3D printing" in captured.out


@pytest.mark.skipif(not HAS_PYVISTA, reason="PyVista not installed")
class TestScaleToSize:
    """Test mesh scaling function."""
    
    def test_scale_max_dimension(self):
        """Test scaling by maximum dimension."""
        # Create box with known dimensions (2x1x0.5)
        box = pv.Box(bounds=[0, 2, 0, 1, 0, 0.5])
        
        scaled = scale_to_size(box, target_size_mm=100, axis='max')
        
        bounds = scaled.bounds
        dims = [bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]]
        
        # Max dimension should be 100
        assert abs(max(dims) - 100) < 0.01
        # Proportions should be maintained
        assert abs(dims[0] - 100) < 0.01  # x was max
        assert abs(dims[1] - 50) < 0.01   # y was half of x
        assert abs(dims[2] - 25) < 0.01   # z was quarter of x
    
    def test_scale_specific_axis(self):
        """Test scaling specific axes."""
        box = pv.Box(bounds=[0, 2, 0, 1, 0, 0.5])
        
        # Scale Y axis to 100mm
        scaled = scale_to_size(box, target_size_mm=100, axis='y')
        
        bounds = scaled.bounds
        y_size = bounds[3] - bounds[2]
        
        assert abs(y_size - 100) < 0.01
    
    def test_invalid_axis(self):
        """Test error for invalid axis."""
        sphere = pv.Sphere()
        
        with pytest.raises(ValueError, match="Invalid axis"):
            scale_to_size(sphere, 50, axis='invalid')


@pytest.mark.skipif(not HAS_PYVISTA, reason="PyVista not installed")
class TestCenterMesh:
    """Test mesh centering function."""
    
    def test_center_offset_mesh(self):
        """Test centering an offset mesh."""
        # Create sphere offset from origin
        sphere = pv.Sphere(center=(5, 3, -2))
        
        # Original center should not be at origin
        assert not np.allclose(sphere.center, [0, 0, 0])
        
        centered = center_mesh(sphere)
        
        # Centered mesh should be at origin
        assert np.allclose(centered.center, [0, 0, 0], atol=1e-10)
    
    def test_already_centered(self):
        """Test centering already centered mesh."""
        sphere = pv.Sphere(center=(0, 0, 0))
        
        centered = center_mesh(sphere)
        
        # Should still be at origin
        assert np.allclose(centered.center, [0, 0, 0])
    
    def test_preserves_shape(self):
        """Test that centering preserves mesh shape."""
        # Create asymmetric mesh
        box = pv.Box(bounds=[1, 3, 2, 5, -1, 1])
        
        original_dims = [
            box.bounds[1] - box.bounds[0],
            box.bounds[3] - box.bounds[2],
            box.bounds[5] - box.bounds[4]
        ]
        
        centered = center_mesh(box)
        
        centered_dims = [
            centered.bounds[1] - centered.bounds[0],
            centered.bounds[3] - centered.bounds[2],
            centered.bounds[5] - centered.bounds[4]
        ]
        
        # Dimensions should be preserved
        assert np.allclose(original_dims, centered_dims)