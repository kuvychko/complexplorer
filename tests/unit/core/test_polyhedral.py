"""Tests for the Klein relative invariants.

These assert the mathematics rather than any output, because the failure mode being guarded against
is a silent one: the forms quoted in the literature cohere as a set only for one orientation of the
solid, and the commonly-remembered signs mix orientations. A mixed set is three root sets that are
each *individually* a perfect solid while being rotated relative to one another, so every per-form
geometry check passes while the ratios stop being rotation-invariant.

Klein's syzygy is the check that catches it, and ``test_the_wrong_sign_convention_fails_the_syzygy``
exists to show the check discriminates rather than passing vacuously.
"""

import numpy as np
import pytest

from complexplorer.core.functions import inverse_stereographic, stereographic_projection
from complexplorer.core.polyhedral import (
    cube_vertex,
    icosahedral_edge,
    icosahedral_hessian,
    icosahedral_vertex,
    octahedral_edge,
    octahedral_vertex,
    polyhedral_features,
    tetrahedral_dual_vertex,
    tetrahedral_vertex,
)
from complexplorer.exceptions import ValidationError

# Sample points away from any root, so a ratio is finite and well conditioned.
PROBE = np.array([0.4 + 0.2j, 1.3 - 0.7j, 0.9 + 1.4j, -2.0 + 0.3j, 0.13 - 1.9j])


def rotation(axis, angle: float) -> np.ndarray:
    """A rotation matrix, by Rodrigues' formula."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    cross = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], dtype=float
    )
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)


def rotate_in_plane(w: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Rotate the sphere under a plane point: project up, rotate, project back.

    Going through the sphere rather than through the equivalent Moebius transformation keeps the test
    free of any convention the forms might disagree with -- it uses the library's own projection.
    """
    xyz = np.asarray(stereographic_projection(w, project_from_north=True))
    moved = xyz @ matrix.T
    return np.asarray(
        inverse_stereographic(moved[..., 0], moved[..., 1], moved[..., 2], project_from_north=True)
    )


def icosahedron() -> np.ndarray:
    """The icosahedron in the orientation the forms are written for."""
    height, radius = 1.0 / np.sqrt(5.0), 2.0 / np.sqrt(5.0)
    turn = 2.0 * np.pi * np.arange(5) / 5.0
    return np.vstack(
        [
            [[0.0, 0.0, 1.0]],
            np.stack([radius * np.cos(turn), radius * np.sin(turn), np.full(5, height)], axis=1),
            np.stack(
                [
                    radius * np.cos(turn + np.pi / 5),
                    radius * np.sin(turn + np.pi / 5),
                    np.full(5, -height),
                ],
                axis=1,
            ),
            [[0.0, 0.0, -1.0]],
        ]
    )


def preserves(matrix: np.ndarray, vertices: np.ndarray) -> bool:
    """Whether a rotation maps a vertex set onto itself."""
    moved = vertices @ matrix.T
    distance = np.linalg.norm(moved[:, None, :] - vertices[None, :, :], axis=-1)
    return bool(np.all(distance.min(axis=1) < 1e-9))


class TestKleinsSyzygy:
    """`H^3 - T^2 == 1728 V^5` ties the three icosahedral forms to each other."""

    def test_it_holds(self):
        left = icosahedral_hessian(PROBE) ** 3 - icosahedral_edge(PROBE) ** 2
        right = 1728 * icosahedral_vertex(PROBE) ** 5

        np.testing.assert_allclose(left, right, rtol=1e-12)

    def test_the_wrong_sign_convention_fails_the_syzygy(self):
        """The point of the check: a mixed-orientation set is rejected loudly.

        `z(z^10 + 11z^5 - 1)` is a perfectly good icosahedral vertex form -- for a *different*
        orientation. Paired with this Hessian and edge form it is wrong, and only the syzygy says so.
        """
        wrong = PROBE * (PROBE**10 + 11 * PROBE**5 - 1)

        left = icosahedral_hessian(PROBE) ** 3 - icosahedral_edge(PROBE) ** 2
        right = 1728 * wrong**5

        relative = np.abs((left - right) / right)
        assert relative.max() > 1.0, "the syzygy must reject the mixed-orientation set"

    def test_the_tetrahedra_multiply_to_the_cube(self):
        """The two antipodal tetrahedra together are the cube."""
        np.testing.assert_allclose(
            tetrahedral_vertex(PROBE) * tetrahedral_dual_vertex(PROBE),
            cube_vertex(PROBE),
            rtol=1e-12,
        )


class TestRotationInvariance:
    """What a mixed-orientation set loses, and what makes these reliefs symmetric."""

    RATIOS = {
        "H^3 / T^2 (icosidodecahedral star)": lambda z: (
            icosahedral_hessian(z) ** 3 / icosahedral_edge(z) ** 2
        ),
        "T^2 / V^5 (icosahedral crown)": lambda z: (
            icosahedral_edge(z) ** 2 / icosahedral_vertex(z) ** 5
        ),
        "V^5 / H^3 (dodecahedral dual)": lambda z: (
            icosahedral_vertex(z) ** 5 / icosahedral_hessian(z) ** 3
        ),
    }

    def axes(self):
        """A 5-fold, a 3-fold and a 2-fold axis of the icosahedral group."""
        solid = icosahedron()
        face = solid[0] + solid[1] + solid[2]
        edge = solid[1] + solid[2]
        return {
            "5-fold (vertex axis)": rotation([0, 0, 1], 2 * np.pi / 5),
            "3-fold (face axis)": rotation(face / np.linalg.norm(face), 2 * np.pi / 3),
            "2-fold (edge axis)": rotation(edge / np.linalg.norm(edge), np.pi),
        }

    def test_the_axes_are_really_symmetries(self):
        """Otherwise the invariance test below would be vacuous."""
        solid = icosahedron()
        for name, matrix in self.axes().items():
            assert preserves(matrix, solid), f"{name} does not preserve the icosahedron"

    @pytest.mark.parametrize("ratio_name", list(RATIOS))
    def test_each_ratio_is_invariant(self, ratio_name):
        ratio = self.RATIOS[ratio_name]
        for axis_name, matrix in self.axes().items():
            moved = rotate_in_plane(PROBE, matrix)
            before, after = np.abs(ratio(PROBE)), np.abs(ratio(moved))
            np.testing.assert_allclose(
                after, before, rtol=1e-9, err_msg=f"{ratio_name} not invariant under {axis_name}"
            )

    def test_an_unequal_degree_ratio_is_not_invariant(self):
        """Why equal binary degree is a requirement and not a stylistic preference.

        `H / T` is degree 20 over 30: the automorphy factors do not cancel, so it renders but is not
        symmetric.
        """
        lopsided = lambda z: icosahedral_hessian(z) / icosahedral_edge(z)  # noqa: E731

        moved = rotate_in_plane(PROBE, self.axes()["2-fold (edge axis)"])
        ratio = np.abs(lopsided(moved)) / np.abs(lopsided(PROBE))

        assert np.abs(ratio - 1).max() > 1e-3


class TestFeaturesMatchTheForms:
    """The projected geometry is the source the coefficients were derived from."""

    CASES = {
        "tetrahedron vertices": ("tetrahedron", "vertices", tetrahedral_vertex, 4),
        "dual tetrahedron vertices": ("tetrahedron_dual", "vertices", tetrahedral_dual_vertex, 4),
        "octahedron vertices": ("octahedron", "vertices", octahedral_vertex, 5),
        "cube vertices": ("octahedron", "faces", cube_vertex, 8),
        "octahedron edge midpoints": ("octahedron", "edges", octahedral_edge, 12),
        "icosahedron vertices": ("icosahedron", "vertices", icosahedral_vertex, 11),
        "dodecahedron vertices": ("icosahedron", "faces", icosahedral_hessian, 20),
        "icosahedron edge midpoints": ("icosahedron", "edges", icosahedral_edge, 30),
    }

    @pytest.mark.parametrize("case", list(CASES))
    def test_the_monic_polynomial_reproduces_the_form(self, case):
        """Task 3.2's check, and the one that would have caught a numerically solved key.

        Compared as a ratio rather than coefficient by coefficient, because the forms are monic
        already: ``form(z) / monic(z)`` must be the constant 1.
        """
        solid, kind, form, _ = self.CASES[case]
        roots = polyhedral_features(solid, kind)

        monic = np.poly(roots)
        ratio = form(PROBE) / np.polyval(monic, PROBE)

        np.testing.assert_allclose(ratio, 1.0, rtol=1e-8)

    @pytest.mark.parametrize("case", list(CASES))
    def test_the_finite_feature_count_is_right(self, case):
        solid, kind, _, expected = self.CASES[case]

        assert len(polyhedral_features(solid, kind)) == expected

    def test_a_vertex_at_the_north_pole_is_omitted(self):
        """It projects to infinity, so it has no finite location to record.

        Both vertex forms are one degree short as polynomials for exactly this reason, and a relief
        built over them carries a real feature at the north pole that no `[re, im]` pair can express.
        """
        # 6 octahedron vertices, 12 icosahedron vertices, one of each at the pole.
        assert len(polyhedral_features("octahedron", "vertices")) == 5
        assert len(polyhedral_features("icosahedron", "vertices")) == 11
        # Face and edge sets have nothing at the pole, so they come back complete.
        assert len(polyhedral_features("octahedron", "faces")) == 8
        assert len(polyhedral_features("icosahedron", "edges")) == 30

    def test_the_roots_are_roots(self):
        """Scale-free: each location is a root of its form, relative to the form's local size."""
        for solid, kind, form, _ in self.CASES.values():
            roots = polyhedral_features(solid, kind)
            # Compare |form(root)| against |form| a little way off the root.
            nearby = np.abs(form(roots + 0.05))
            assert np.all(np.abs(form(roots)) < 1e-6 * np.maximum(nearby, 1.0))

    def test_the_order_is_canonical_and_stable(self):
        """Preset records are serialized into a byte-compared manifest, so order must not wobble."""
        first = polyhedral_features("icosahedron", "edges")
        second = polyhedral_features("icosahedron", "edges")

        np.testing.assert_array_equal(first, second)
        moduli = np.abs(first)
        assert np.all(np.diff(np.round(moduli, 9)) >= 0), "not sorted by modulus"

    def test_dodecahedron_vertices_are_the_icosahedron_face_centres(self):
        """Which is what makes the dual pieces dual."""
        dodeca = polyhedral_features("icosahedron", "faces")
        assert len(dodeca) == 20
        assert np.abs(icosahedral_hessian(dodeca)).max() < 1e-4

    @pytest.mark.parametrize(
        ("solid", "kind"), [("dodecahedron", "vertices"), ("icosahedron", "corners")]
    )
    def test_unknown_inputs_are_rejected(self, solid, kind):
        with pytest.raises(ValidationError):
            polyhedral_features(solid, kind)
