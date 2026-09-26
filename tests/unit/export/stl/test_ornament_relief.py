"""Tests for the ornament relief: normalization, tip shape, and bulk contrast.

These exercise the generator end to end at a coarse resolution. The mathematics of the constant and
of the transfer is tested in ``tests/unit/core/test_normalization.py``; what matters here is that the
generator wires them up -- that the shape stops depending on the arbitrary scalar in front of ``f``,
that ``pointiness`` reaches the transfer, and that ``contrast`` is off unless asked for.
"""

import numpy as np
import pytest

pytest.importorskip("pyvista")

import pyvista as pv

from complexplorer.core.scaling import normalization_constant
from complexplorer.exceptions import ValidationError
from complexplorer.export.stl.ornament_generator import OrnamentGenerator
from complexplorer.export.stl.utils import count_edges

RES = 60


def radii(mesh) -> np.ndarray:
    return np.linalg.norm(np.asarray(mesh.points), axis=1)


def ornament(func, **kwargs):
    gen = OrnamentGenerator(func, resolution=kwargs.pop("resolution", RES), **kwargs)
    return gen, gen.generate_ornament(verbose=False)


class TestNormalization:
    def test_a_rescaled_function_gives_the_same_ornament(self):
        """The point of normalizing: 100*f is the same shape as f, not a different one.

        Sea level sits at |f| = 1 for every self-dual transfer, so without this the arbitrary
        constant in front of a function changes the geometry rather than only its labels.
        """
        func = lambda z: z / (z**10 - 1)  # noqa: E731

        _, plain = ornament(func)
        _, scaled = ornament(lambda z: 100.0 * func(z))

        np.testing.assert_allclose(radii(plain), radii(scaled), rtol=1e-9)

    def test_the_constant_matches_the_closed_form(self):
        gen, _ = ornament(lambda z: z / (z**10 - 1), resolution=200)

        exact = normalization_constant([0.0], np.exp(2j * np.pi * np.arange(10) / 10))
        assert gen.applied_normalization == pytest.approx(exact, rel=5e-3)

    def test_an_explicit_constant_is_used_as_given(self):
        gen, mesh = ornament(lambda z: z / (z**3 - 1), normalize=2.5)

        assert gen.applied_normalization == 2.5
        # And it is actually applied: the same function without it is a different shape.
        _, plain = ornament(lambda z: z / (z**3 - 1), normalize=None)
        assert not np.allclose(radii(mesh), radii(plain))

    def test_normalization_off_reproduces_the_unnormalized_mapping(self):
        """``normalize=None`` is the documented way back to the pre-3.1 mapping."""
        func = lambda z: z / (z**3 - 1)  # noqa: E731

        gen, off = ornament(func, normalize=None)
        assert gen.applied_normalization is None

        # With the constant pinned to 1.0 the geometry must be identical to switching it off.
        _, unit = ornament(func, normalize=1.0)
        np.testing.assert_allclose(radii(off), radii(unit), rtol=1e-12)

    def test_the_magnitude_scalar_stays_raw(self):
        """Normalization is a statement about geometry, not about the function."""
        func = lambda z: 100.0 * z / (z**3 - 1)  # noqa: E731

        gen, mesh = ornament(func)

        assert gen.applied_normalization != pytest.approx(1.0)
        # magnitude is |f| itself, so it still reports the factor of 100 the caller wrote.
        assert np.nanmax(np.asarray(mesh["magnitude"])) > 10.0
        # radius, by contrast, is bounded by the transfer.
        assert radii(mesh).max() <= 1.0 + 1e-9

    def test_the_median_statistic_is_selectable(self):
        """More robust where a high-order feature at infinity drags sea level off the structure."""
        func = lambda z: z / (z**10 - 1)  # noqa: E731

        geometric, _ = ornament(func, normalize="geometric")
        median, _ = ornament(func, normalize="median")

        assert geometric.applied_normalization != pytest.approx(median.applied_normalization)
        assert median.applied_normalization > 0

    def test_a_non_positive_explicit_constant_is_rejected(self):
        gen = OrnamentGenerator(lambda z: z, resolution=20, normalize=-1.0)
        with pytest.raises(ValidationError, match="must be positive"):
            gen.generate_ornament(verbose=False)

    def test_an_unknown_statistic_is_rejected(self):
        gen = OrnamentGenerator(lambda z: z, resolution=20, normalize="mean")
        with pytest.raises(ValidationError, match="Unknown statistic"):
            gen.generate_ornament(verbose=False)


class TestTipShape:
    def test_the_default_transfer_accepts_a_scale(self):
        """arctan has no scale, so pointiness would have nothing to set."""
        gen = OrnamentGenerator(lambda z: z, resolution=20)

        assert gen.scaling == "logarithmic"
        assert gen.scaling_params["base"] == pytest.approx(np.exp(2.0))

    def test_the_default_depth_matches_the_named_presets(self):
        assert OrnamentGenerator(lambda z: z, resolution=20).scaling_params["r_min"] == 0.2

    def test_pointiness_sharpens_tips(self):
        """A sharper tip reaches its maximum radius over a smaller neighbourhood of the feature."""
        func = lambda z: z / (z**5 - 1)  # noqa: E731

        _, blunt = ornament(func, pointiness=0.75)
        _, sharp = ornament(func, pointiness=3.0)

        # Share of the surface within the top tenth of the radial range: smaller means the approach
        # to the tip is confined to a smaller neighbourhood.
        def near_tip(mesh):
            r = radii(mesh)
            span = r.max() - r.min()
            return float(np.mean(r > r.max() - 0.1 * span))

        assert near_tip(sharp) < near_tip(blunt)

    def test_feature_order_is_compensated(self):
        """A double pole at pole_order=2 approaches like a simple pole at pole_order=1."""
        simple = OrnamentGenerator(lambda z: 1 / (z - 2), resolution=20, pole_order=1)
        double = OrnamentGenerator(lambda z: 1 / (z - 2) ** 2, resolution=20, pole_order=2)

        # The tip exponent is mu / k, and k = pointiness * pole_order, so both are 1 / pointiness.
        assert double.sharpness == pytest.approx(2 * simple.sharpness)
        assert 2 / double.sharpness == pytest.approx(1 / simple.sharpness)

    def test_the_scale_can_be_set_directly(self):
        """ln(10) makes one unit of relief a decade of gain -- 20 dB -- which is not a tip exponent."""
        gen = OrnamentGenerator(
            lambda z: z, resolution=20, sharpness=float(np.log(10.0)), pointiness=99.0, pole_order=7
        )

        assert gen.sharpness == pytest.approx(np.log(10.0))
        assert gen.scaling_params["base"] == pytest.approx(10.0)

    def test_explicit_scaling_params_win(self):
        """So a caller can pin anything the pointiness machinery would otherwise derive."""
        gen = OrnamentGenerator(
            lambda z: z, resolution=20, scaling_params={"base": 3.0, "r_min": 0.45}
        )

        assert gen.scaling_params["base"] == 3.0
        assert gen.scaling_params["r_min"] == 0.45

    def test_a_non_positive_scale_is_rejected(self):
        with pytest.raises(ValidationError, match="must be positive"):
            OrnamentGenerator(lambda z: z, resolution=20, pointiness=0.0)

    def test_the_old_look_is_still_reachable(self):
        """The migration route: arctan at the old depth, with normalization off."""
        gen = OrnamentGenerator(
            lambda z: z,
            resolution=20,
            scaling="arctan",
            scaling_params={"r_min": 0.5, "r_max": 1.0},
            normalize=None,
        )

        assert gen.scaling == "arctan"
        assert gen.scaling_params == {"r_min": 0.5, "r_max": 1.0}


class TestContrast:
    def test_it_is_off_by_default(self):
        """A single logistic unless asked: applied globally it degrades tip-led pieces."""
        gen = OrnamentGenerator(lambda z: z, resolution=20)

        assert gen.contrast is None
        assert gen.scaling == "logarithmic"
        assert "boost" not in gen.scaling_params

    def test_requesting_it_selects_the_two_scale_transfer(self):
        gen = OrnamentGenerator(lambda z: z, resolution=20, contrast=(4.0, 0.6))

        assert gen.scaling == "log_mixture"
        assert gen.scaling_params["boost"] == 4.0
        assert gen.scaling_params["weight"] == 0.6
        assert gen.scaling_params["scale"] == pytest.approx(2.0)

    def test_it_spreads_area_away_from_mid_radius(self):
        """The whole point: a body a single logistic leaves nearly spherical gets sculpted."""
        # Few, widely separated features, so the body is otherwise smooth.
        func = lambda z: (z**2 + 1) / (z**2 - 1)  # noqa: E731

        _, plain = ornament(func)
        _, boosted = ornament(func, contrast=(4.0, 0.6))

        def spread(mesh):
            r = radii(mesh)
            return float(np.abs(r - 0.5 * (r.max() + r.min())).mean())

        assert spread(boosted) > spread(plain)

    def test_the_mixture_stays_self_dual_end_to_end(self):
        """The relief of 1/f mirrors that of f about sea level, sample for sample."""
        func = lambda z: z / (z**3 - 1)  # noqa: E731
        kwargs = {"contrast": (3.0, 0.5), "resolution": 40}

        _, forward = ornament(func, **kwargs)
        _, inverse = ornament(lambda z: 1 / func(z), **kwargs)

        r_min, r_max = 0.2, 1.0
        # Mirroring about sea level means the two radii sum to r_min + r_max at each sample.
        np.testing.assert_allclose(
            np.sort(radii(forward)) + np.sort(radii(inverse))[::-1],
            r_min + r_max,
            atol=5e-3,
        )


class TestSavedMesh:
    def test_normals_are_consistent_and_outward(self, tmp_path):
        """So the facet normals written into the STL mean something to a consumer that reads them."""
        out = tmp_path / "ornament.stl"
        OrnamentGenerator(lambda z: z / (z**3 - 1), resolution=RES).generate_and_save(
            str(out), size_mm=40, verbose=False
        )

        saved = pv.read(str(out)).clean(tolerance=1e-9)
        with_normals = saved.compute_normals(
            consistent_normals=True, auto_orient_normals=True, inplace=False
        )
        cell_normals = np.asarray(with_normals.cell_normals)
        centres = np.asarray(with_normals.cell_centers().points)
        centres = centres - centres.mean(axis=0)

        # Outward: the normal agrees with the outward radial direction at essentially every facet.
        outward = centres / np.linalg.norm(centres, axis=1, keepdims=True)
        assert float(np.mean(np.einsum("ij,ij->i", cell_normals, outward) > 0)) > 0.99

    def test_the_saved_solid_is_closed(self, tmp_path):
        """Under the corrected check -- which is what made this assertion possible to write."""
        out = tmp_path / "closed.stl"
        OrnamentGenerator(lambda z: z / (z**3 - 1), resolution=RES).generate_and_save(
            str(out), size_mm=40, verbose=False
        )

        saved = pv.read(str(out)).clean(tolerance=1e-9)
        assert count_edges(saved, boundary_edges=True) == 0
        assert count_edges(saved, non_manifold_edges=True) == 0
        assert saved.volume > 0

    def test_validation_sees_the_uncentred_scaled_mesh(self, tmp_path):
        """Radii are measured from the star centre, so validation must precede centring."""
        # A lopsided piece: its star centre and its bounding-box centre are far apart.
        gen = OrnamentGenerator(lambda z: 1 / (z - 1.5), resolution=RES)
        gen.generate_ornament(verbose=False)
        out = tmp_path / "lopsided.stl"
        gen.save_stl(str(out), size_mm=40, verbose=False)

        results = gen.validate_mesh(size_mm=40, verbose=False)
        assert results["min_radius_mm"] < results["max_radius_mm"]
