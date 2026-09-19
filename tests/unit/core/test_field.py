"""Unit tests for the backend-agnostic ComplexField sampling (core/field.py)."""

import numpy as np
import pytest

from complexplorer.core.domain import Disk, Rectangle
from complexplorer.core.field import (
    ComplexField,
    sample,
    sample_sphere,
    sphere_coordinates,
)
from complexplorer.utils.validation import ValidationError


class TestPlanarSample:
    def test_values_modulus_phase(self):
        fld = sample(lambda z: z**2, Rectangle(4, 4), resolution=20)
        assert fld.kind == "planar"
        assert fld.w.shape == (20, 20)
        np.testing.assert_allclose(fld.modulus, np.abs(fld.w))
        np.testing.assert_allclose(fld.phase, np.angle(fld.w))
        assert fld.z is not None and fld.sphere_xyz is None

    def test_domain_mask_matches_outmask(self):
        dom = Disk(1.5)
        fld = sample(lambda z: z, dom, resolution=25)
        np.testing.assert_array_equal(fld.mask, dom.outmask(25))
        assert fld.mask.any()  # corners of the bounding grid are outside the disk

    def test_explicit_z_grid_no_domain(self):
        z = np.array([[0 + 0j, 1 + 0j], [0 + 1j, 1 + 1j]])
        fld = sample(lambda z: z + 1, z=z)
        assert fld.mask is None
        np.testing.assert_allclose(fld.w, z + 1)

    def test_requires_domain_or_z(self):
        with pytest.raises(ValidationError):
            sample(lambda z: z)

    def test_records_nonfinite_inside_domain(self):
        # 1/z has a pole at the origin (a grid node at this resolution)
        fld = sample(lambda z: 1 / z, Rectangle(3, 3), resolution=25)
        assert fld.nonfinite.sum() >= 1


class TestSphereSample:
    def test_shapes_and_kind(self):
        sf = sample_sphere(lambda z: z, resolution=30)
        assert sf.kind == "sphere"
        assert sf.sphere_xyz.shape == (30, 30, 3)
        assert sf.w.shape == (30, 30)
        assert sf.z is None

    def test_canonical_projection_zero_at_south_pole(self):
        # Canonical convention: z = 0 maps to the south pole (Cartesian z = -1).
        x, y, z = sphere_coordinates(60)
        south = np.unravel_index(np.argmin(z), z.shape)
        sf = sample_sphere(lambda w: w, resolution=60)  # identity -> w itself
        # near the south pole the projected value is ~0 (offset by avoid_poles)
        assert np.abs(sf.w[south]) < 0.02
        # and large near the north pole (z = +1)
        north = np.unravel_index(np.argmax(z), z.shape)
        assert np.abs(sf.w[north]) > 50

    def test_domain_mask(self):
        sf = sample_sphere(lambda z: z, resolution=40, domain=Disk(2.0))
        assert sf.mask is not None and sf.mask.any()

    def test_unit_sphere_coordinates(self):
        x, y, z = sphere_coordinates(20)
        r = np.sqrt(x**2 + y**2 + z**2)
        np.testing.assert_allclose(r, 1.0, atol=1e-12)


def test_field_is_pyvista_free():
    """core.field must import without PyVista (it is part of the 2D/core path)."""
    import sys

    assert "complexplorer.core.field" in sys.modules or ComplexField is not None
    import complexplorer.core.field as field_mod

    src = field_mod.__file__
    with open(src, encoding="utf-8") as fh:
        text = fh.read()
    assert "import pyvista" not in text and "pv." not in text
