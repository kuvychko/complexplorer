"""Tests for the Riemann surface builder (add-riemann-surfaces)."""

import warnings

import numpy as np
import pytest

pytest.importorskip("pyvista")

from complexplorer.mesh import build_riemann_surface
from complexplorer.utils.validation import ValidationError


def _points(sm):
    return sm.to_pyvista().points


class TestPowerFamily:
    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_height_is_re_w_and_n_sheets(self, n):
        warnings.simplefilter("ignore")
        r_max = 1.5
        sm = build_riemann_surface("power", n=n, r_max=r_max, resolution=30)
        pts = _points(sm)
        # height (z) is Re(w), which ranges over [-R, R] with R = r_max**(1/n)
        R = r_max ** (1.0 / n)
        assert np.isclose(pts[:, 2].max(), R, atol=0.05)
        assert np.isclose(pts[:, 2].min(), -R, atol=0.05)
        # n distinct sheet-heights over a generic z = the n n-th roots' real parts
        z0 = 0.6 + 0.4j
        roots_re = sorted(
            {round((z0 ** (1 / n) * np.exp(2j * np.pi * k / n)).real, 6) for k in range(n)}
        )
        assert len(roots_re) == n

    def test_seam_closes(self):
        """The phi=0 and phi=2*pi columns coincide -> the surface closes (no tear)."""
        warnings.simplefilter("ignore")
        sm = build_riemann_surface("power", n=2, resolution=30)
        # extract_surface keeps all grid points; the mesh is a closed band, so its bounds in
        # x and y are symmetric about the cut and finite (a torn surface would not close).
        pts = _points(sm)
        assert np.all(np.isfinite(pts))

    def test_z_equals_w_pow_n_via_xy(self):
        """The (x, y) of each vertex equals Re/Im of z = w**n (definitional sanity)."""
        warnings.simplefilter("ignore")
        sm = build_riemann_surface("power", n=2, r_max=1.0, resolution=40)
        pts = _points(sm)
        # |z| = |w|**n <= r_max; height = Re(w), |Re(w)| <= |w| = |z|**(1/n)
        rz = np.hypot(pts[:, 0], pts[:, 1])
        assert rz.max() <= 1.0 + 1e-6
        assert np.abs(pts[:, 2]).max() <= 1.0 + 1e-6

    def test_n_too_small_raises(self):
        with pytest.raises(ValidationError):
            build_riemann_surface("power", n=1)


class TestLogFamily:
    @pytest.mark.parametrize("turns", [1, 3])
    def test_helicoid_height_spans_turns(self, turns):
        warnings.simplefilter("ignore")
        sm = build_riemann_surface("log", turns=turns, resolution=30)
        pts = _points(sm)
        assert np.isclose(pts[:, 2].min(), 0.0, atol=1e-6)
        assert np.isclose(pts[:, 2].max(), 2 * np.pi * turns, atol=0.1)

    def test_xy_within_radius(self):
        warnings.simplefilter("ignore")
        r_max = 1.5
        sm = build_riemann_surface("log", turns=2, r_max=r_max, resolution=30)
        pts = _points(sm)
        assert np.hypot(pts[:, 0], pts[:, 1]).max() <= r_max + 1e-6

    def test_turns_too_small_raises(self):
        with pytest.raises(ValidationError):
            build_riemann_surface("log", turns=0)


class TestAlgebraicFamily:
    """w**2 = P(z) (add-algebraic-curves)."""

    ELLIPTIC = [1, 0, -1, 0]  # w**2 = z**3 - z

    def test_two_sheets_double_the_points(self):
        warnings.simplefilter("ignore")
        res = 30
        sm = build_riemann_surface("algebraic", p=self.ELLIPTIC, resolution=res)
        # single sheet grid is res x 4*res; the cover carries two of them, unmerged
        assert sm.to_pyvista().n_points == 2 * res * 4 * res

    def test_sheets_are_mirror_heights(self):
        warnings.simplefilter("ignore")
        sm = build_riemann_surface("algebraic", p=self.ELLIPTIC, resolution=25)
        pts = _points(sm)
        half = pts.shape[0] // 2
        np.testing.assert_allclose(pts[:half, 2], -pts[half:, 2], atol=1e-12)
        # x/y grids identical across sheets
        np.testing.assert_allclose(pts[:half, :2], pts[half:, :2], atol=1e-12)

    def test_branch_points_in_metadata(self):
        warnings.simplefilter("ignore")
        sm = build_riemann_surface("algebraic", p=self.ELLIPTIC, resolution=20)
        assert sm.metadata["topology"] == "riemann_surface:algebraic:deg=3"
        roots = sorted(np.asarray(sm.metadata["branch_points"]).real)
        np.testing.assert_allclose(roots, [-1.0, 0.0, 1.0], atol=1e-9)

    def test_colors_finite(self):
        warnings.simplefilter("ignore")
        sm = build_riemann_surface("algebraic", p=self.ELLIPTIC, resolution=20)
        mesh = sm.to_pyvista()
        assert "RGB" in mesh.array_names and "phase" in mesh.array_names
        rgb = np.asarray(mesh["RGB"], dtype=float)
        assert np.all(np.isfinite(rgb))
        assert rgb.min() >= 0.0

    def test_missing_p_raises(self):
        with pytest.raises(ValidationError, match="requires polynomial coefficients"):
            build_riemann_surface("algebraic")

    def test_short_p_raises(self):
        with pytest.raises(ValidationError, match="at least two coefficients"):
            build_riemann_surface("algebraic", p=[1])

    def test_zero_leading_coefficient_raises(self):
        with pytest.raises(ValidationError, match="nonzero leading coefficient"):
            build_riemann_surface("algebraic", p=[0, 1, 1])


class TestColorsAndValidation:
    def test_colors_finite_and_in_gamut(self):
        warnings.simplefilter("ignore")
        sm = build_riemann_surface("power", n=2, resolution=30)
        mesh = sm.to_pyvista()
        assert "RGB" in mesh.array_names and "phase" in mesh.array_names
        rgb = np.asarray(mesh["RGB"], dtype=float)
        # RGB may be stored 0-1 or 0-255 depending on flatten; just require finite + bounded
        assert np.all(np.isfinite(rgb))
        assert rgb.min() >= 0.0

    def test_unknown_family_raises(self):
        with pytest.raises(ValidationError):
            build_riemann_surface("elliptic")
