"""Klein relative invariants of the polyhedral rotation groups.

These are the building blocks of a relief that carries the full symmetry of a Platonic solid. Each
form is a polynomial whose roots are one class of feature of a solid — its vertices, its face centres
or its edge midpoints — stereographically projected onto the complex plane.

**A ratio of invariants must have equal binary degree, or it is not invariant.** A rational map of
degree ``d`` has exactly ``d`` zeros and ``d`` poles on the sphere, so a function with features at one
solid's vertices *and nowhere else* does not exist. What does exist is a ratio of two relative
invariants: each picks up an automorphy factor under a rotation of the sphere, and the factors cancel
only when the two forms have the same **binary** degree, at which point ``|f|`` descends to a genuine
invariant function. That is why the reliefs built from these are ratios like ``H**3 / T**2`` (60 over
60) rather than single forms. A ratio of unequal degree fails silently: it renders, it looks
plausible, and it is not symmetric.

Binary degree is not always the polynomial degree. A form whose solid has a feature at the **north
pole** is one degree short, because that feature projects to infinity and so contributes no finite
root: :func:`octahedral_vertex` is binary degree 6 as a polynomial of degree 5, and
:func:`icosahedral_vertex` is binary degree 12 as a polynomial of degree 11. That missing root is a
real feature of the relief — a zero or pole at infinity, at the north pole of the sphere — and it is
invisible to anything that only reads the polynomial, including
:func:`~complexplorer.core.scaling.normalization_constant`, which takes finite divisors only.

**On the coefficients.** They are not transcribed from a table, and that matters. The forms quoted in
the literature cohere as a set only for one particular orientation of the solid, and the
commonly-remembered signs mix orientations: pairing ``z*(z**10 + 11*z**5 - 1)`` with the usual
Hessian gives three root sets that are each *individually* a perfect icosahedron, dodecahedron and
edge set while being rotated relative to one another. The ratios then stop being rotation-invariant
by a factor of 1e3 to 1e5, and checking each form's roots against its own solid does not notice.

So the coefficients here are derived: :func:`polyhedral_features` builds each solid explicitly and
projects it, and the test suite asserts that the monic polynomial over those projected roots
reproduces the integer coefficients below, that the icosahedral trio satisfies Klein's syzygy
``H**3 - T**2 == 1728 * V**5``, and that each documented ratio is invariant under a rotation from its
group. The syzygy is the check worth running on any set you transcribe yourself: it fails by orders
of magnitude for a mixed-orientation set, where a per-form geometry check passes.

The classical single-letter names (``V``, ``F``, ``E``, ``H``, ``T``, ``Phi``, ``Psi``) are given in
each docstring; the exported names are spelled out, because single letters collide with everything.
"""

from __future__ import annotations

import itertools
from functools import cache

import numpy as np

from ..exceptions import ValidationError
from .functions import inverse_stereographic

__all__ = [
    "cube_vertex",
    "icosahedral_edge",
    "icosahedral_hessian",
    "icosahedral_vertex",
    "octahedral_edge",
    "octahedral_vertex",
    "polyhedral_features",
    "tetrahedral_dual_vertex",
    "tetrahedral_vertex",
]

_ROOT_THREE = np.sqrt(3.0)


# --------------------------------------------------------------------------------------
# The forms
# --------------------------------------------------------------------------------------


def tetrahedral_vertex(z):
    """Tetrahedral vertex form (classically ``Phi``), binary degree 4.

    Roots: the 4 vertices of a tetrahedron inscribed in the cube. Its coefficients are complex —
    this orientation of the tetrahedron is not symmetric about the real axis — which is expected
    rather than a transcription slip.
    """
    return z**4 + 2j * _ROOT_THREE * z**2 + 1


def tetrahedral_dual_vertex(z):
    """The antipodal tetrahedron's vertex form (classically ``Psi``), binary degree 4.

    Roots: the other 4 cube vertices. ``tetrahedral_vertex * tetrahedral_dual_vertex == cube_vertex``
    exactly: the two tetrahedra together are the cube.
    """
    return z**4 - 2j * _ROOT_THREE * z**2 + 1


def octahedral_vertex(z):
    """Octahedral vertex form (classically ``V``), binary degree 6.

    Only degree 5 as a polynomial: the sixth vertex sits at the north pole and so projects to
    infinity, contributing no finite root. Roots: the 6 octahedron vertices.
    """
    return z * (z**4 - 1)


def cube_vertex(z):
    """Cube vertex form = octahedral face form (classically ``F``), binary degree 8.

    Roots: the 8 cube vertices, which are the face centres of the octahedron.
    """
    return z**8 + 14 * z**4 + 1


def octahedral_edge(z):
    """Octahedral edge form (classically ``E``), binary degree 12.

    Roots: the 12 octahedron edge midpoints, which are the vertices of a cuboctahedron.
    """
    return z**12 - 33 * z**8 - 33 * z**4 + 1


def icosahedral_vertex(z):
    """Icosahedral vertex form (classically ``V``), binary degree 12.

    Only degree 11 as a polynomial: the twelfth vertex sits at the north pole and projects to
    infinity. Roots: the 12 icosahedron vertices.

    Note the **minus** 11. The ``+11`` variant is a different orientation of the icosahedron; paired
    with the Hessian and edge form below it fails Klein's syzygy by a factor of about 20 and destroys
    the rotation invariance of every ratio built from it.
    """
    return z * (z**10 - 11 * z**5 - 1)


def icosahedral_hessian(z):
    """Icosahedral Hessian (classically ``H``), binary degree 20.

    Roots: the 20 dodecahedron vertices, which are the face centres of the icosahedron.
    """
    return z**20 + 228 * z**15 + 494 * z**10 - 228 * z**5 + 1


def icosahedral_edge(z):
    """Icosahedral edge form (classically ``T``), binary degree 30.

    Roots: the 30 icosahedron edge midpoints, which are the vertices of an icosidodecahedron.

    Note the signs on the 522 terms run minus then plus, the reverse of the variant usually quoted.
    """
    return z**30 - 522 * z**25 - 10005 * z**20 - 10005 * z**10 + 522 * z**5 + 1


# --------------------------------------------------------------------------------------
# The geometry the coefficients come from
# --------------------------------------------------------------------------------------

_SOLIDS = ("tetrahedron", "tetrahedron_dual", "octahedron", "icosahedron")
_KINDS = ("vertices", "faces", "edges")


def _unit(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    return points / np.linalg.norm(points, axis=1, keepdims=True)


@cache
def _solid_vertices(solid: str) -> tuple[tuple[float, float, float], ...]:
    """The unit-sphere vertices of a solid, in the orientation the forms above are written for."""
    if solid == "tetrahedron":
        # The tetrahedron whose vertex form is `tetrahedral_vertex`; its antipode is the dual.
        points = _unit([[-1, -1, -1], [-1, 1, 1], [1, -1, 1], [1, 1, -1]])
    elif solid == "tetrahedron_dual":
        points = _unit([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
    elif solid == "octahedron":
        # A vertex at each pole, so one vertex projects to infinity.
        points = np.array(
            [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float
        )
    elif solid == "icosahedron":
        # A vertex at the north pole, with rings of five at z = +-1/sqrt(5): the orientation that
        # makes the icosahedral trio cohere. The ring radius follows from |v| = 1.
        height = 1.0 / np.sqrt(5.0)
        radius = 2.0 / np.sqrt(5.0)
        turn = 2.0 * np.pi * np.arange(5) / 5.0
        upper = np.stack([radius * np.cos(turn), radius * np.sin(turn), np.full(5, height)], axis=1)
        lower = np.stack(
            [
                radius * np.cos(turn + np.pi / 5.0),
                radius * np.sin(turn + np.pi / 5.0),
                np.full(5, -height),
            ],
            axis=1,
        )
        points = np.vstack([[[0.0, 0.0, 1.0]], upper, lower, [[0.0, 0.0, -1.0]]])
    else:
        raise ValidationError(f"Unknown solid: {solid}. Available: {', '.join(_SOLIDS)}")
    return tuple(tuple(float(c) for c in point) for point in points)


def _faces_and_edges(vertices: np.ndarray) -> tuple[list[tuple[int, ...]], list[tuple[int, int]]]:
    """Triangular faces and edges of a deltahedron, from nearest-neighbour distance.

    Only used for the tetrahedron, octahedron and icosahedron, whose faces are all triangles -- the
    cube and dodecahedron are reached as the *face centres* of their duals instead.
    """
    distance = np.linalg.norm(vertices[:, None, :] - vertices[None, :, :], axis=-1)
    edge_length = np.sort(distance[0])[1]
    adjacent = np.isclose(distance, edge_length, atol=1e-9)
    indices = range(len(vertices))
    edges = [(i, j) for i, j in itertools.combinations(indices, 2) if adjacent[i, j]]
    faces = [
        triple
        for triple in itertools.combinations(indices, 3)
        if adjacent[triple[0], triple[1]]
        and adjacent[triple[1], triple[2]]
        and adjacent[triple[0], triple[2]]
    ]
    return faces, edges


def polyhedral_features(solid: str, kind: str = "vertices") -> np.ndarray:
    """The **finite** projected locations of one class of a solid's features.

    Builds the solid explicitly and stereographically projects the requested features onto the
    complex plane, using the library's canonical convention -- the south pole maps to ``0`` and the
    north pole to infinity, matching
    :func:`~complexplorer.core.field.sample_sphere`. These are the roots of the corresponding form
    above, and the source those forms' coefficients were derived from.

    Constructed rather than solved, deliberately. Solving a degree-30 polynomial numerically gives
    roots that agree to perhaps ten significant digits between linear-algebra implementations, which
    is not enough for a preset record that is serialized into a byte-compared manifest. Projected
    geometry depends only on ``sqrt`` and trigonometry, which agree to within one unit in the last
    place everywhere.

    Parameters
    ----------
    solid : {'tetrahedron', 'tetrahedron_dual', 'octahedron', 'icosahedron'}
        Which solid. The cube and the dodecahedron are the ``faces`` of the octahedron and the
        icosahedron respectively.
    kind : {'vertices', 'faces', 'edges'}, default='vertices'
        Which class of feature: vertices, face centres, or edge midpoints, each projected onto the
        unit sphere first.

    Returns
    -------
    np.ndarray
        Complex locations, in a canonical order (by modulus, then by argument) so the result is
        reproducible. A feature **at the north pole is omitted**, because it projects to infinity and
        has no finite location: that happens for the ``vertices`` of the octahedron and the
        icosahedron, each of which has one there. Every other combination returns the full set.

    Raises
    ------
    ValidationError
        If ``solid`` or ``kind`` is not recognized.

    Examples
    --------
    >>> import numpy as np
    >>> dodecahedron = polyhedral_features("icosahedron", "faces")
    >>> len(dodecahedron)
    20
    >>> bool(np.abs(icosahedral_hessian(dodecahedron)).max() < 1e-9)
    True
    """
    if kind not in _KINDS:
        raise ValidationError(f"Unknown feature kind: {kind}. Available: {', '.join(_KINDS)}")
    vertices = np.asarray(_solid_vertices(solid), dtype=float)

    if kind == "vertices":
        points = vertices
    else:
        faces, edges = _faces_and_edges(vertices)
        groups = faces if kind == "faces" else edges
        points = _unit([vertices[list(group)].mean(axis=0) for group in groups])

    # Drop anything at the north pole: it projects to infinity.
    finite = points[points[:, 2] < 1.0 - 1e-12]
    located = np.asarray(
        inverse_stereographic(finite[:, 0], finite[:, 1], finite[:, 2], project_from_north=True)
    ).ravel()

    order = np.lexsort((np.round(np.angle(located), 9), np.round(np.abs(located), 9)))
    return located[order]
