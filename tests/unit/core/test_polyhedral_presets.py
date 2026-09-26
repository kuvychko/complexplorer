"""Tests for the polyhedral ornament family in ``cp.catalog``.

The mathematics of the invariants is tested in ``test_polyhedral.py``; these check that the presets
carry it faithfully — the right divisor, the relief settings the pieces need, and an expression string
that really is the function.
"""

import numpy as np
import pytest

import complexplorer as cp
from complexplorer.core.expression import compile_expression
from complexplorer.core.polyhedral import polyhedral_features

FAMILY = {
    "tetrahedral_dual": {"pole_order": 1, "resolution": 250, "zeros": 4, "poles": 4},
    "octahedral_crown": {"pole_order": 2, "resolution": 300, "zeros": 12, "poles": 5},
    "cube_octahedron_dual": {"pole_order": 3, "resolution": 300, "zeros": 5, "poles": 8},
    "icosahedral_crown": {"pole_order": 5, "resolution": 400, "zeros": 30, "poles": 11},
    "dodecahedron_icosahedron_dual": {
        "pole_order": 3,
        "resolution": 400,
        "zeros": 11,
        "poles": 20,
    },
    "icosidodecahedral_star": {"pole_order": 2, "resolution": 400, "zeros": 20, "poles": 30},
}

# Off any root, so the ratios are finite and well conditioned.
PROBE = np.array([0.37 + 0.21j, 1.27 - 0.63j, 0.83 + 1.41j, -1.9 + 0.29j])


class TestTheFamilyIsRegistered:
    def test_all_six_are_present(self):
        assert set(FAMILY) <= set(cp.catalog.list())

    def test_the_family_is_discoverable_by_tag(self):
        tagged = {p.id for p in cp.catalog.filter("polyhedral")}
        assert tagged == set(FAMILY)

    @pytest.mark.parametrize("preset_id", list(FAMILY))
    def test_each_is_tagged_as_a_printable_ornament(self, preset_id):
        assert "ornament" in cp.catalog.get(preset_id).tags


class TestReliefSettings:
    @pytest.mark.parametrize("preset_id", list(FAMILY))
    def test_the_feature_order_and_resolution_are_recorded(self, preset_id):
        preset = cp.catalog.get(preset_id)
        expected = FAMILY[preset_id]

        assert preset.pole_order == expected["pole_order"]
        assert preset.resolution == expected["resolution"]

    @pytest.mark.parametrize("preset_id", list(FAMILY))
    def test_they_are_serialized(self, preset_id):
        record = cp.catalog.get(preset_id).to_dict()

        assert record["pole_order"] == FAMILY[preset_id]["pole_order"]
        assert record["resolution"] == FAMILY[preset_id]["resolution"]

    @pytest.mark.parametrize("preset_id", list(FAMILY))
    def test_the_sphere_is_not_clipped_to_the_viewing_window(self, preset_id):
        """Four of the six have a real feature at the north pole; clipping would cut it off.

        The domain these carry is a 2D viewing window for the portrait, and applying it to a sphere
        sample deletes cells — which for these pieces deletes geometry and lets the repair pass fill a
        flat cap over what should be a spike or a pit.
        """
        preset = cp.catalog.get(preset_id)

        assert preset.clip_ornament_to_domain is False
        assert preset.to_dict()["clip_ornament_to_domain"] is False

    def test_the_existing_presets_are_untouched(self):
        """The new fields are absent rather than defaulted into something that changes geometry."""
        plain = cp.catalog.get("pole_flower_10")

        assert plain.pole_order is None
        assert plain.resolution is None
        assert plain.clip_ornament_to_domain is True
        record = plain.to_dict()
        for key in ("pole_order", "resolution", "clip_ornament_to_domain"):
            assert key not in record, f"{key} must not appear for a preset that does not set it"


class TestDivisors:
    """Counts, multiplicities, and the features that sit at infinity."""

    @pytest.mark.parametrize("preset_id", list(FAMILY))
    def test_the_counts_match(self, preset_id):
        stats = cp.catalog.get(preset_id).answer_key_stats()
        expected = FAMILY[preset_id]

        assert stats["count_by_type"].get("zero", 0) == expected["zeros"]
        assert stats["count_by_type"].get("pole", 0) == expected["poles"]

    @pytest.mark.parametrize(
        ("preset_id", "kind", "at_infinity"),
        [
            ("octahedral_crown", "pole", "one of six octahedron vertices"),
            ("cube_octahedron_dual", "zero", "one of six octahedron vertices"),
            ("icosahedral_crown", "pole", "one of twelve icosahedron vertices"),
            ("dodecahedron_icosahedron_dual", "zero", "one of twelve icosahedron vertices"),
        ],
    )
    def test_a_feature_at_infinity_is_described_not_recorded(self, preset_id, kind, at_infinity):
        """A location is a finite pair, so the one at the north pole cannot be a record."""
        preset = cp.catalog.get(preset_id)

        # Nothing claims a finite location for it...
        assert all(np.isfinite(record["at"]).all() for record in preset.singularities)
        # ...and the story says it is there, so the key does not read as complete.
        assert "infinity" in preset.story

    def test_the_star_has_no_feature_at_infinity(self):
        """Both of its forms are full degree, so its key IS complete."""
        preset = cp.catalog.get("icosidodecahedral_star")

        assert preset.answer_key_stats()["count"] == 50  # 20 zeros + 30 poles
        assert "no feature at infinity" in preset.story

    @pytest.mark.parametrize("preset_id", list(FAMILY))
    def test_every_recorded_location_is_a_singularity_of_the_function(self, preset_id):
        """The divisor is checked against the callable rather than trusted."""
        preset = cp.catalog.get(preset_id)
        with np.errstate(all="ignore"):
            for record in preset.singularities:
                at = complex(*record["at"])
                value = np.abs(preset.func(np.array([at])))[0]
                if record["type"] == "zero":
                    assert value < 1e-3, f"{preset_id}: no zero at {at}"
                else:
                    assert not np.isfinite(value) or value > 1e3, f"{preset_id}: no pole at {at}"

    def test_the_star_divisor_is_the_expected_geometry(self):
        """Zeros on the dodecahedron, poles at the icosahedron's edge midpoints."""

        def ordered(values):
            # Complex numbers have no ordering, so sort on the pair.
            return sorted(values, key=lambda w: (round(w.real, 8), round(w.imag, 8)))

        preset = cp.catalog.get("icosidodecahedral_star")
        zeros = ordered(complex(*r["at"]) for r in preset.singularities if r["type"] == "zero")
        poles = ordered(complex(*r["at"]) for r in preset.singularities if r["type"] == "pole")

        np.testing.assert_allclose(
            zeros, ordered(polyhedral_features("icosahedron", "faces")), atol=1e-9
        )
        np.testing.assert_allclose(
            poles, ordered(polyhedral_features("icosahedron", "edges")), atol=1e-9
        )

    def test_min_separation_is_the_ridge_proxy(self):
        """It is the tightest peak-next-to-pit distance, which is what binds at a given print size.

        The star is the densest piece in the family, so it must have the smallest separation.
        """
        separations = {
            pid: cp.catalog.get(pid).answer_key_stats()["min_separation"] for pid in FAMILY
        }

        assert separations["icosidodecahedral_star"] == min(separations.values())
        assert separations["tetrahedral_dual"] == max(separations.values())


class TestExpressionsAreContracts:
    @pytest.mark.parametrize("preset_id", sorted(cp.catalog.list()))
    def test_every_preset_expression_evaluates_to_its_callable(self, preset_id):
        """Checked for the whole catalog, not only the new family -- it is a contract for all."""
        preset = cp.catalog.get(preset_id)
        evaluated = compile_expression(preset.expression)

        with np.errstate(all="ignore"):
            from_func = np.asarray(preset.func(PROBE), dtype=complex)
            from_expression = np.asarray(evaluated(PROBE), dtype=complex)

        np.testing.assert_allclose(
            from_expression, from_func, rtol=1e-8, atol=1e-10, equal_nan=True
        )

    @pytest.mark.parametrize("preset_id", list(FAMILY))
    def test_the_expression_names_no_helper(self, preset_id):
        """It is written out in full: a preset record has to stand alone."""
        expression = cp.catalog.get(preset_id).expression

        assert "z" in expression
        for helper in ("icosahedral", "octahedral", "tetrahedral", "cube_vertex", "polyhedral"):
            assert helper not in expression
