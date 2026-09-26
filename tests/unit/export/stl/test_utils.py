"""Tests for STL export utility functions."""

import numpy as np
import pytest
import pyvista as pv

from complexplorer.exceptions import ValidationError
from complexplorer.export.stl.utils import (
    center_mesh,
    count_edges,
    scale_to_size,
    validate_printability,
)


class TestValidatePrintability:
    """Test mesh validation for 3D printing."""

    def test_watertight_sphere(self):
        """Test validation of watertight sphere."""
        sphere = pv.Sphere(radius=1.0)

        results = validate_printability(sphere, verbose=False)

        assert results["is_watertight"] is True
        assert results["is_manifold"] is True
        assert results["n_boundary_edges"] == 0
        assert results["n_non_manifold_edges"] == 0
        assert results["volume"] > 0
        assert results["surface_area"] > 0

    def test_open_mesh(self):
        """Test validation of mesh with holes."""
        # Create cylinder without caps (open mesh)
        cylinder = pv.Cylinder(height=2.0, radius=0.5, capping=False)

        results = validate_printability(cylinder, verbose=False)

        assert results["is_watertight"] is False
        assert results["n_boundary_edges"] > 0

    def test_closed_mesh_with_creases_is_watertight(self):
        """The reported bug: a cube is closed and manifold, and was called neither.

        ``extract_feature_edges`` enables all four edge classes by default, so asking for boundary
        edges alone returned every crease as well. A sphere is smooth enough to have no creases,
        which is why the bug survived a sphere-only test.
        """
        cube = pv.Cube().triangulate()
        assert cube.n_open_edges == 0  # the ground truth, from PyVista itself

        results = validate_printability(cube, size_mm=50, verbose=False)

        assert results["is_watertight"] is True
        assert results["is_manifold"] is True
        assert results["n_boundary_edges"] == 0
        assert results["n_non_manifold_edges"] == 0
        assert results["volume"] > 0
        # The creases are still there; they are simply not boundary edges.
        assert count_edges(cube, feature_edges=True) == 12

    def test_counts_reject_an_unknown_edge_class(self):
        with pytest.raises(ValidationError, match="Unknown edge class"):
            count_edges(pv.Sphere(), sharp_edges=True)

    def test_radial_extent_replaces_wall_thickness(self):
        """Radii from the origin are reported; the constant-False wall test is gone."""
        sphere = pv.Sphere(radius=1.0)

        results = validate_printability(sphere, size_mm=10, verbose=False)

        assert results["min_radius_mm"] == pytest.approx(1.0, rel=1e-3)
        assert results["max_radius_mm"] == pytest.approx(1.0, rel=1e-3)
        for gone in ("wall_thickness_ok", "estimated_min_wall_mm", "recommended_size_mm"):
            assert gone not in results

    def test_radii_are_measured_from_the_origin(self):
        """Which is why validation runs before centring: centring moves the star centre away."""
        offset = pv.Sphere(radius=1.0, center=(5, 0, 0))

        results = validate_printability(offset, verbose=False)

        # From the origin, not from the mesh's own centre: 4 to 6, not 1 to 1.
        assert results["min_radius_mm"] == pytest.approx(4.0, rel=1e-3)
        assert results["max_radius_mm"] == pytest.approx(6.0, rel=1e-3)

    def test_verbose_output(self, capsys):
        """Test verbose output."""
        sphere = pv.Sphere()

        validate_printability(sphere, size_mm=50, verbose=True)

        captured = capsys.readouterr()
        assert "Mesh Validation Results" in captured.out
        assert "Watertight: True" in captured.out
        assert "ready for 3D printing" in captured.out
        assert "Radius from origin" in captured.out
        # The quantity that actually fails on an FDM printer is named rather than substituted for.
        assert "Ridge width between adjacent features is not measured" in captured.out
        assert "wall" not in captured.out.lower()


class TestScaleToSize:
    """Test mesh scaling function."""

    def test_scale_max_dimension(self):
        """Test scaling by maximum dimension."""
        # Create box with known dimensions (2x1x0.5)
        box = pv.Box(bounds=[0, 2, 0, 1, 0, 0.5])

        scaled = scale_to_size(box, target_size_mm=100, axis="max")

        bounds = scaled.bounds
        dims = [bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]]

        # Max dimension should be 100
        assert abs(max(dims) - 100) < 0.01
        # Proportions should be maintained
        assert abs(dims[0] - 100) < 0.01  # x was max
        assert abs(dims[1] - 50) < 0.01  # y was half of x
        assert abs(dims[2] - 25) < 0.01  # z was quarter of x

    def test_scale_specific_axis(self):
        """Test scaling specific axes."""
        box = pv.Box(bounds=[0, 2, 0, 1, 0, 0.5])

        # Scale Y axis to 100mm
        scaled = scale_to_size(box, target_size_mm=100, axis="y")

        bounds = scaled.bounds
        y_size = bounds[3] - bounds[2]

        assert abs(y_size - 100) < 0.01

    def test_invalid_axis(self):
        """Test error for invalid axis."""
        sphere = pv.Sphere()

        with pytest.raises(ValueError, match="Invalid axis"):
            scale_to_size(sphere, 50, axis="invalid")


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
            box.bounds[5] - box.bounds[4],
        ]

        centered = center_mesh(box)

        centered_dims = [
            centered.bounds[1] - centered.bounds[0],
            centered.bounds[3] - centered.bounds[2],
            centered.bounds[5] - centered.bounds[4],
        ]

        # Dimensions should be preserved
        assert np.allclose(original_dims, centered_dims)
