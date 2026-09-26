"""Function preset registry (``cp.catalog``).

A curated, metadata-rich, **serializable** description of complex functions, designed to be
the single source consumed by the gallery, the CLI and the STL object cards. The records are
self-contained enough that an independent implementation can rebuild the same mathematics and
be checked against the same exact answer keys.

This module is deliberately **PyVista-free** (presets are data, not rendering) and imports
only the core/data layer. Distinct from ``complexplorer.PlotPresets`` (render settings):
this is the *function* registry, exposed as ``cp.catalog``.

A preset carries:

- ``func`` — the callable Complexplorer renders with (NOT serialized),
- ``expression`` — a string like ``"z / (z**10 - 1)"`` (the function, parseable),
- ``domain_spec`` / ``cmap_spec`` / ``scaling_spec`` — plain dicts whose keys mirror the
  target constructor kwargs; every complex value is an ``[re, im]`` pair,
- ``singularities`` — hand-authored, exact answer keys (one record per location),
- ``id`` / ``title`` / ``story`` / ``tags``.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..typing import CmapSpec, ComplexFunction, DomainSpec, ScalingSpec, SingularityRecord
from ..utils.validation import ValidationError
from .colormap import Chessboard, Colormap, LogRings, Phase, PolarChessboard
from .domain import Annulus, Disk, Domain, Rectangle
from .polyhedral import (
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
from .scaling import get_scaling_preset

SINGULARITY_TYPES = frozenset({"zero", "pole", "essential", "branch_point"})


# --------------------------------------------------------------------------------------
# Complex <-> [re, im] (JSON has no complex type)
# --------------------------------------------------------------------------------------


def _complex_from_pair(pair: Any) -> complex:
    return complex(float(pair[0]), float(pair[1]))


def _pair(z: complex) -> list[float]:
    z = complex(z)
    return [float(z.real), float(z.imag)]


# --------------------------------------------------------------------------------------
# Spec factories — build live objects on demand; core classes untouched
# --------------------------------------------------------------------------------------

_DOMAIN_TYPES: dict[str, type[Domain]] = {
    "rectangle": Rectangle,
    "disk": Disk,
    "annulus": Annulus,
}

_CMAP_TYPES: dict[str, type[Colormap]] = {
    "Phase": Phase,
    "Chessboard": Chessboard,
    "PolarChessboard": PolarChessboard,
    "LogRings": LogRings,
}

# Spec keys naming a complex-valued constructor kwarg (stored as [re, im]).
_COMPLEX_KEYS = frozenset({"center"})


def _kwargs_from_spec(spec: dict) -> tuple[str, dict]:
    spec = dict(spec)
    type_name = spec.pop("type", None)
    kwargs = {
        key: (_complex_from_pair(val) if key in _COMPLEX_KEYS else val) for key, val in spec.items()
    }
    return type_name, kwargs


def domain_from_spec(spec: dict) -> Domain:
    """Instantiate a ``Domain`` from a serializable spec dict (keys = constructor kwargs)."""
    type_name, kwargs = _kwargs_from_spec(spec)
    cls = _DOMAIN_TYPES.get(type_name)
    if cls is None:
        raise ValidationError(
            f"Unknown domain type {type_name!r}; supported: {sorted(_DOMAIN_TYPES)}"
        )
    return cls(**kwargs)


def cmap_from_spec(spec: dict) -> Colormap:
    """Instantiate a ``Colormap`` from a serializable spec dict (keys = constructor kwargs)."""
    type_name, kwargs = _kwargs_from_spec(spec)
    cls = _CMAP_TYPES.get(type_name)
    if cls is None:
        raise ValidationError(
            f"Unknown colormap type {type_name!r}; supported: {sorted(_CMAP_TYPES)}"
        )
    return cls(**kwargs)


def scaling_from_spec(spec: str | dict) -> dict:
    """Resolve a ``scaling_spec`` to the ``{method, params, ...}`` dict.

    Accepts a named ``SCALING_PRESETS`` key (resolved via ``get_scaling_preset``) or an
    inline dict in the ``SCALING_PRESETS`` shape.
    """
    if isinstance(spec, str):
        return get_scaling_preset(spec)
    return dict(spec)


# --------------------------------------------------------------------------------------
# Singularity answer-key records
# --------------------------------------------------------------------------------------


# Derived coordinates (roots of unity, cube roots, pi multiples) come out of libm, which differs
# by an ULP between platforms: the 10th root of unity that is -0.8090169943749475 on Windows is
# -0.8090169943749476 on Linux. The record is an interchange format that is committed, diffed and
# compared, so it is quantized to a precision no platform disagrees about. 12 significant digits is
# ~4 digits clear of double precision's noise floor and far finer than any position needs.
MANIFEST_SIGNIFICANT_DIGITS = 12


def _stable(value):
    """Quantize floats anywhere in a JSON-ready record; everything else passes through."""
    if isinstance(value, bool):  # bool is an int subclass, and must stay a bool
        return value
    if isinstance(value, float):
        return float(f"{value:.{MANIFEST_SIGNIFICANT_DIGITS}g}")
    if isinstance(value, dict):
        return {key: _stable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_stable(item) for item in value]
    return value


def singularity(
    type_: str, at: complex | tuple[float, float], order: int | None, label: str = ""
) -> dict:
    """Build one exact singularity record ``{type, at:[re,im], order, label?}``."""
    if type_ not in SINGULARITY_TYPES:
        raise ValidationError(
            f"Unknown singularity type {type_!r}; supported: {sorted(SINGULARITY_TYPES)}"
        )
    if type_ == "essential" and order is not None:
        raise ValidationError("essential singularities must have order=None")
    at_pair = _pair(at) if isinstance(at, (complex, int, float)) else [float(at[0]), float(at[1])]
    record = {"type": type_, "at": at_pair, "order": order}
    if label:
        record["label"] = label
    return record


def _features(solid: str, kind: str) -> list[complex]:
    """Projected polyhedral features, rounded for the record.

    Rounded to 10 decimals: the locations come from elementary functions and are good to ~1e-15, and
    the manifest quantizes to 12 significant digits anyway, so rounding here costs nothing and makes
    the record obviously independent of the last bit of a trig implementation.
    """
    located = polyhedral_features(solid, kind)
    return [complex(round(w.real, 10), round(w.imag, 10)) for w in located]


def roots_of_unity(n: int) -> list[list[float]]:
    """The n-th roots of unity as ``[re, im]`` pairs (for poles/zeros on the unit circle)."""
    return [[float(np.cos(2 * np.pi * k / n)), float(np.sin(2 * np.pi * k / n))] for k in range(n)]


# --------------------------------------------------------------------------------------
# FunctionPreset
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FunctionPreset:
    """A curated complex function: renderable callable + serializable description."""

    id: str
    title: str
    expression: str
    func: ComplexFunction = field(repr=False)
    domain_spec: DomainSpec = field(default_factory=lambda: DomainSpec())
    cmap_spec: CmapSpec = field(default_factory=lambda: CmapSpec(type="Phase", phase_sectors=6))
    # A plain string names a preset scaling, such as "balanced"; a dict gives method and params.
    scaling_spec: str | ScalingSpec = "balanced"
    singularities: tuple[SingularityRecord, ...] = ()
    story: str = ""
    tags: tuple[str, ...] = ()
    # Relief parameters a printable preset needs to come out as intended. Both are optional: a
    # preset that does not set them renders exactly as it did before they existed.
    #
    # pole_order is the order of the piece's features. It matters because the tip exponent is
    # `order / scale` and the scale is derived as `pointiness * pole_order`, so rendering an
    # order-2 piece as though its features were simple gives an exact cone where a cusp was
    # intended -- the blunt version of the piece. It is not inferred from the function: fitting the
    # scale to a function's log-modulus spread is a known dead end that yields tip exponents from 1
    # to 8 and blunts most shapes.
    pole_order: float | None = None
    # resolution is per-preset rather than a raised global default, because the dense pieces need
    # 400 while raising the library default would slow every unrelated render.
    resolution: int | None = None
    # Whether the ornament path should clip the sphere to `domain_spec`.
    #
    # A domain spec is really a 2D viewing window, and clipping a sphere sample with one removes
    # cells: for a preset whose relief has a genuine feature at the north pole -- which is where the
    # plane's far field lands -- that deletes the feature and the repair pass fills a flat cap over
    # it. It stays True by default because the transcendental presets rely on it to keep `exp` and
    # friends from overflowing in the far field, which is why it was forwarded in the first place.
    clip_ornament_to_domain: bool = True

    def __post_init__(self):
        for record in self.singularities:
            singularity(
                record["type"], record["at"], record.get("order"), record.get("label", "")
            )  # validates; raises on bad records
        if self.pole_order is not None and self.pole_order <= 0:
            raise ValidationError(f"pole_order must be positive; got {self.pole_order}")
        if self.resolution is not None and self.resolution < 2:
            raise ValidationError(f"resolution must be at least 2; got {self.resolution}")

    # -- live objects (instantiated on demand) --
    def domain(self) -> Domain:
        return domain_from_spec(self.domain_spec)

    def colormap(self) -> Colormap:
        return cmap_from_spec(self.cmap_spec)

    def scaling(self) -> dict:
        return scaling_from_spec(self.scaling_spec)

    # -- derived geometry (pure function of the authored answer key) --
    def answer_key_stats(self) -> dict:
        """Derived geometry of the singularity answer key.

        Returns ``count``, ``count_by_type`` (sorted by type), and ``min_separation`` — the
        smallest Euclidean distance in the z-plane between any two singularity locations, or
        ``None`` when there are fewer than two. Computed from the hand-authored records only,
        never by analyzing ``func``.
        """
        by_type: dict[str, int] = {}
        for record in self.singularities:
            by_type[record["type"]] = by_type.get(record["type"], 0) + 1
        points = [s["at"] for s in self.singularities]
        if len(points) < 2:
            min_separation: float | None = None
        else:
            min_separation = min(
                math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in itertools.combinations(points, 2)
            )
        return {
            "count": len(self.singularities),
            "count_by_type": dict(sorted(by_type.items())),
            "min_separation": min_separation,
        }

    # -- serialization (the interchange record) --
    def to_dict(self) -> dict:
        """JSON-ready record of everything EXCEPT the live ``func``.

        Floats are quantized to `MANIFEST_SIGNIFICANT_DIGITS`, so the record is identical on
        every platform. The live attributes keep full precision; only this record is quantized.
        """
        return _stable(
            {
                "id": self.id,
                "title": self.title,
                "expression": self.expression,
                "domain_spec": dict(self.domain_spec),
                "cmap_spec": dict(self.cmap_spec),
                "scaling_spec": self.scaling_spec,
                "singularities": [dict(s) for s in self.singularities],
                "answer_key_stats": self.answer_key_stats(),
                "story": self.story,
                "tags": list(self.tags),
                # Omitted entirely when unset, so the record of every existing preset is unchanged
                # and the byte-stable manifest does not gain a column of nulls.
                **({} if self.pole_order is None else {"pole_order": self.pole_order}),
                **({} if self.resolution is None else {"resolution": self.resolution}),
                **({} if self.clip_ornament_to_domain else {"clip_ornament_to_domain": False}),
            }
        )


# --------------------------------------------------------------------------------------
# Registry (cp.catalog)
# --------------------------------------------------------------------------------------


class _Catalog:
    """The function preset registry. Exposed as ``cp.catalog``."""

    def __init__(self, presets: dict[str, FunctionPreset]):
        self._presets = presets

    def get(self, preset_id: str) -> FunctionPreset:
        try:
            return self._presets[preset_id]
        except KeyError:
            raise ValidationError(
                f"Unknown preset id {preset_id!r}; {len(self._presets)} available "
                f"(see catalog.list())"
            ) from None

    def list(self) -> list[str]:
        """Sorted list of preset ids."""
        return sorted(self._presets)

    def filter(self, tag: str) -> list[FunctionPreset]:
        """All presets carrying ``tag`` (sorted by id)."""
        return [self._presets[i] for i in sorted(self._presets) if tag in self._presets[i].tags]

    def __len__(self) -> int:
        return len(self._presets)

    def __contains__(self, preset_id: str) -> bool:
        return preset_id in self._presets


# --------------------------------------------------------------------------------------
# Curated content (17 presets) — exact, hand-authored answer keys
# --------------------------------------------------------------------------------------

_RECT4 = {"type": "rectangle", "re_length": 4, "im_length": 4}
_RECT8 = {"type": "rectangle", "re_length": 8, "im_length": 4}
_ANNULUS = {"type": "annulus", "inner_radius": 0.2, "outer_radius": 3}
_PHASE = {"type": "Phase", "phase_sectors": 6, "auto_scale_r": True}


def _build_presets() -> dict[str, FunctionPreset]:
    presets: list[FunctionPreset] = []

    def add(**kw):
        presets.append(FunctionPreset(**kw))

    # --- basic maps ---
    add(
        id="identity",
        title="Identity",
        expression="z",
        func=lambda z: z,
        domain_spec=_RECT4,
        cmap_spec=_PHASE,
        scaling_spec="balanced",
        singularities=(singularity("zero", 0, 1),),
        story="The identity map. A single simple zero at the origin; phase winds once.",
        tags=("basic", "canonical", "function-guessr"),
    )
    add(
        id="square",
        title="z squared",
        expression="z**2",
        func=lambda z: z**2,
        domain_spec=_RECT4,
        cmap_spec=_PHASE,
        singularities=(singularity("zero", 0, 2),),
        story="A double zero at the origin; phase winds twice.",
        tags=("basic", "canonical", "function-guessr"),
    )
    add(
        id="reciprocal",
        title="1 / z",
        expression="1 / z",
        func=lambda z: 1 / z,
        domain_spec=_ANNULUS,
        cmap_spec=_PHASE,
        singularities=(singularity("pole", 0, 1),),
        story="A simple pole at the origin (Möbius inversion); phase winds backward.",
        tags=("basic", "canonical", "poles", "function-guessr"),
    )
    add(
        id="mobius_cayley",
        title="Cayley transform",
        expression="(z - 1) / (z + 1)",
        func=lambda z: (z - 1) / (z + 1),
        domain_spec=_RECT4,
        cmap_spec=_PHASE,
        singularities=(singularity("zero", 1, 1), singularity("pole", -1, 1)),
        story="One zero at +1, one pole at -1. Maps the right half-plane to the unit disk.",
        tags=("basic", "canonical", "mobius"),
    )
    add(
        id="cubic_real_roots",
        title="z³ - z",
        expression="z**3 - z",
        func=lambda z: z**3 - z,
        domain_spec=_RECT4,
        cmap_spec=_PHASE,
        singularities=(
            singularity("zero", -1, 1),
            singularity("zero", 0, 1),
            singularity("zero", 1, 1),
        ),
        story="Three simple zeros at -1, 0, 1.",
        tags=("basic", "canonical"),
    )
    add(
        id="rational_zeros_poles",
        title="(z² - 1)/(z² + 1)",
        expression="(z**2 - 1) / (z**2 + 1)",
        func=lambda z: (z**2 - 1) / (z**2 + 1),
        domain_spec=_RECT4,
        cmap_spec=_PHASE,
        singularities=(
            singularity("zero", -1, 1),
            singularity("zero", 1, 1),
            singularity("pole", 1j, 1, "i"),
            singularity("pole", -1j, 1, "-i"),
        ),
        story="Zeros at ±1, poles at ±i.",
        tags=("canonical", "singularity-detective"),
    )

    # --- singularities ---
    add(
        id="pole_order_2",
        title="Double pole",
        expression="1 / z**2",
        func=lambda z: 1 / z**2,
        domain_spec=_ANNULUS,
        cmap_spec=_PHASE,
        singularities=(singularity("pole", 0, 2),),
        story="An order-2 pole at the origin; phase winds backward twice.",
        tags=("poles", "singularity-detective"),
    )
    add(
        id="pole_order_3",
        title="Triple pole",
        expression="1 / z**3",
        func=lambda z: 1 / z**3,
        domain_spec=_ANNULUS,
        cmap_spec=_PHASE,
        singularities=(singularity("pole", 0, 3),),
        story="An order-3 pole at the origin.",
        tags=("poles", "singularity-detective"),
    )
    add(
        id="essential_exp_inv",
        title="Essential singularity",
        expression="exp(1 / z)",
        func=lambda z: np.exp(1 / z),
        domain_spec=_ANNULUS,
        cmap_spec=_PHASE,
        singularities=(singularity("essential", 0, None),),
        story="exp(1/z) has an essential singularity at 0 — infinitely dense structure nearby.",
        tags=("singularity-detective", "essential"),
    )

    # --- branches (principal branch in the callable) ---
    add(
        id="sqrt",
        title="Square root",
        expression="sqrt(z)",
        func=lambda z: np.sqrt(z),
        domain_spec=_RECT4,
        cmap_spec=_PHASE,
        singularities=(singularity("branch_point", 0, 2),),
        story="Principal-branch square root; an order-2 branch point at 0 (two sheets).",
        tags=("branches", "branch-cut-zoo", "ornament"),
    )
    add(
        id="log",
        title="Natural log",
        expression="log(z)",
        func=lambda z: np.log(z),
        domain_spec=_RECT4,
        cmap_spec=_PHASE,
        singularities=(singularity("branch_point", 0, None, "logarithmic"),),
        story="Principal-branch logarithm; a logarithmic (infinite-order) branch point at 0.",
        tags=("branches", "branch-cut-zoo"),
    )
    add(
        id="cbrt",
        title="Cube root",
        expression="z**(1/3)",
        func=lambda z: z ** (1 / 3),
        domain_spec=_RECT4,
        cmap_spec=_PHASE,
        singularities=(singularity("branch_point", 0, 3),),
        story="Principal-branch cube root; an order-3 branch point at 0 (three sheets).",
        tags=("branches", "branch-cut-zoo"),
    )

    # --- ornaments ---
    add(
        id="pole_flower_10",
        title="Pole Flower 10",
        expression="z / (z**10 - 1)",
        func=lambda z: z / (z**10 - 1),
        domain_spec=_ANNULUS,
        cmap_spec=_PHASE,
        scaling_spec="poles_emphasis",
        singularities=(
            singularity("zero", 0, 1),
            *(singularity("pole", _complex_from_pair(p), 1) for p in roots_of_unity(10)),
        ),
        story="A ring of ten simple poles (the 10th roots of unity) around a central simple "
        "zero. The signature printable ornament.",
        tags=("ornament", "poles", "canonical", "singularity-detective"),
    )

    # --- transcendental ---
    add(
        id="sine",
        title="Sine",
        expression="sin(z)",
        func=lambda z: np.sin(z),
        domain_spec=_RECT8,
        cmap_spec=_PHASE,
        singularities=(
            singularity("zero", -np.pi, 1, "-pi"),
            singularity("zero", 0, 1),
            singularity("zero", np.pi, 1, "pi"),
        ),
        story="Simple zeros at integer multiples of pi (−pi, 0, pi shown).",
        tags=("transcendental", "function-guessr"),
    )
    add(
        id="exp",
        title="Exponential",
        expression="exp(z)",
        func=lambda z: np.exp(z),
        domain_spec=_RECT4,
        cmap_spec=_PHASE,
        singularities=(),
        story="Entire and never zero: no finite zeros or poles (an empty answer key).",
        tags=("transcendental", "function-guessr"),
    )
    add(
        id="tangent",
        title="Tangent",
        expression="tan(z)",
        func=lambda z: np.tan(z),
        domain_spec=_RECT4,
        cmap_spec=_PHASE,
        singularities=(
            singularity("zero", 0, 1),
            singularity("pole", np.pi / 2, 1, "pi/2"),
            singularity("pole", -np.pi / 2, 1, "-pi/2"),
        ),
        story="Zero at 0; simple poles at ±pi/2 (within the shown window).",
        tags=("transcendental", "singularity-detective"),
    )

    # --- dynamics (static snapshot) ---
    add(
        id="newton_cubic",
        title="Newton map (z³ - 1)",
        expression="(2*z**3 + 1) / (3*z**2)",
        func=lambda z: (2 * z**3 + 1) / (3 * z**2),
        domain_spec=_RECT4,
        cmap_spec=_PHASE,
        singularities=(
            singularity("pole", 0, 2),
            *(
                singularity("zero", (0.5 ** (1 / 3)) * np.exp(1j * (np.pi + 2 * np.pi * k) / 3), 1)
                for k in range(3)
            ),
        ),
        story="The Newton iteration map for z³ - 1: an order-2 pole at 0 and three "
        "simple zeros at the cube roots of -1/2.",
        tags=("dynamics", "poles"),
    )

    # --- polyhedral ornaments -----------------------------------------------------------
    #
    # Ratios of Klein relative invariants, at equal binary degree so that the automorphy factors
    # cancel and |f| descends to a genuinely invariant function on the sphere. See
    # `complexplorer.core.polyhedral`, which also explains why the coefficients are derived from
    # explicit geometry rather than transcribed.
    #
    # These are printable pieces, so each carries the relief settings it needs: `pole_order` (the
    # transfer's scale is derived from it, and an order-3 piece rendered as order 1 comes out blunt)
    # and `resolution` (the dense ones need 400). None of them clips the sphere to its 2D viewing
    # window: four of the six have a real feature at the north pole, and clipping would cut it off.
    #
    # `singularities` locations come from `polyhedral_features`, which projects the solid rather than
    # solving the polynomial -- see the docstring there for why a solved key cannot survive the
    # byte-compared manifest.

    tetra = _features("tetrahedron", "vertices")
    tetra_dual = _features("tetrahedron_dual", "vertices")
    octa_v = _features("octahedron", "vertices")
    cube_v = _features("octahedron", "faces")
    octa_e = _features("octahedron", "edges")
    ico_v = _features("icosahedron", "vertices")
    dodeca_v = _features("icosahedron", "faces")
    ico_e = _features("icosahedron", "edges")

    _PHI = "(z**4 + 2j*sqrt(3)*z**2 + 1)"
    _PSI = "(z**4 - 2j*sqrt(3)*z**2 + 1)"
    _OCTA_V = "(z*(z**4 - 1))"
    _CUBE_V = "(z**8 + 14*z**4 + 1)"
    _OCTA_E = "(z**12 - 33*z**8 - 33*z**4 + 1)"
    _ICO_V = "(z*(z**10 - 11*z**5 - 1))"
    _ICO_H = "(z**20 + 228*z**15 + 494*z**10 - 228*z**5 + 1)"
    _ICO_T = "(z**30 - 522*z**25 - 10005*z**20 - 10005*z**10 + 522*z**5 + 1)"

    add(
        id="tetrahedral_dual",
        title="Tetrahedral Dual",
        expression=f"{_PHI} / {_PSI}",
        func=lambda z: tetrahedral_vertex(z) / tetrahedral_dual_vertex(z),
        domain_spec=_RECT4,
        cmap_spec=_PHASE,
        singularities=(
            *(singularity("zero", z, 1) for z in tetra),
            *(singularity("pole", p, 1) for p in tetra_dual),
        ),
        story="Four spikes on one tetrahedron over four pits on its antipode -- the two "
        "tetrahedra that together make the cube. Its symmetry is T (order 12, rotations "
        "only): alone in this family it has no mirror plane, so a cut through it gives two "
        "halves that are genuinely different rather than two copies of one part. That is the "
        "point of the piece.",
        tags=("ornament", "polyhedral", "poles"),
        pole_order=1,
        resolution=250,
        clip_ornament_to_domain=False,
    )

    add(
        id="octahedral_crown",
        title="Octahedral Crown",
        expression=f"{_OCTA_E} / {_OCTA_V}**2",
        func=lambda z: octahedral_edge(z) / octahedral_vertex(z) ** 2,
        domain_spec=_RECT4,
        cmap_spec=_PHASE,
        singularities=(
            *(singularity("zero", z, 1) for z in octa_e),
            *(singularity("pole", p, 2) for p in octa_v),
        ),
        story="Twelve simple zeros at the octahedron's edge midpoints under six double poles "
        "at its vertices: a crown of six spikes with a pit between each neighbouring pair. "
        "Full O_h symmetry (order 48, all nine mirror planes), so any coordinate plane cuts "
        "it into identical halves. The sixth pole sits at the north pole of the sphere, i.e. "
        "at infinity, so the answer key below lists five of the six.",
        tags=("ornament", "polyhedral", "poles"),
        pole_order=2,
        resolution=300,
        clip_ornament_to_domain=False,
    )

    add(
        id="cube_octahedron_dual",
        title="Cube-Octahedron Dual",
        expression=f"{_OCTA_V}**4 / {_CUBE_V}**3",
        func=lambda z: octahedral_vertex(z) ** 4 / cube_vertex(z) ** 3,
        domain_spec=_RECT4,
        cmap_spec=_PHASE,
        singularities=(
            *(singularity("zero", z, 4) for z in octa_v),
            *(singularity("pole", p, 3) for p in cube_v),
        ),
        story="The vertex form of one solid over the vertex form of its dual, at matching "
        "binary degree: six order-4 pits on the octahedron's axes, eight triple spikes on the "
        "cube's vertices. Full O_h symmetry. Its spikes point along the cube diagonals, which "
        "makes it the piece that exposed a sizing bug -- an axis-aligned bounding box "
        "understates it by exactly sqrt(3), because each spike projects onto a coordinate axis "
        "at 0.577 of its length. One of the six zeros is at infinity, so five are listed.",
        tags=("ornament", "polyhedral", "poles"),
        pole_order=3,
        resolution=300,
        clip_ornament_to_domain=False,
    )

    add(
        id="icosahedral_crown",
        title="Icosahedral Crown",
        expression=f"{_ICO_T}**2 / {_ICO_V}**5",
        func=lambda z: icosahedral_edge(z) ** 2 / icosahedral_vertex(z) ** 5,
        domain_spec=_RECT4,
        cmap_spec=_PHASE,
        singularities=(
            *(singularity("zero", z, 2) for z in ico_e),
            *(singularity("pole", p, 5) for p in ico_v),
        ),
        story="The icosahedral answer to the Octahedral Crown, and a jump from binary degree "
        "12 to 60 -- the icosahedral rotation group has order 60, so 60 is the lowest degree "
        "any invariant ratio can have. Twelve spikes of order 5 at the icosahedron's vertices "
        "over thirty double pits at its edge midpoints. Full I_h symmetry (order 120), the "
        "largest here. Its features are order 5, so the derived transfer scale is capped: this "
        "is the piece the cap exists for, because a mesh cannot deliver the dynamic range an "
        "uncapped order-5 scale asks for. One pole is at infinity; eleven are listed.",
        tags=("ornament", "polyhedral", "poles"),
        pole_order=5,
        resolution=400,
        clip_ornament_to_domain=False,
    )

    add(
        id="dodecahedron_icosahedron_dual",
        title="Dodecahedron-Icosahedron Dual",
        expression=f"{_ICO_V}**5 / {_ICO_H}**3",
        func=lambda z: icosahedral_vertex(z) ** 5 / icosahedral_hessian(z) ** 3,
        domain_spec=_RECT4,
        cmap_spec=_PHASE,
        singularities=(
            *(singularity("zero", z, 5) for z in ico_v),
            *(singularity("pole", p, 3) for p in dodeca_v),
        ),
        story="The icosahedral twin of the Cube-Octahedron Dual, built the same way: one "
        "solid's vertex form over its dual's, at matching degree. Twenty triple spikes on the "
        "dodecahedron's vertices, twelve order-5 pits on the icosahedron's. Its spikes sit "
        "exactly where the Icosahedral Crown has its pits. Full I_h symmetry. One zero is at "
        "infinity; eleven are listed.",
        tags=("ornament", "polyhedral", "poles"),
        pole_order=3,
        resolution=400,
        clip_ornament_to_domain=False,
    )

    add(
        id="icosidodecahedral_star",
        title="Icosidodecahedral Star",
        expression=f"{_ICO_H}**3 / {_ICO_T}**2",
        func=lambda z: icosahedral_hessian(z) ** 3 / icosahedral_edge(z) ** 2,
        domain_spec=_RECT4,
        cmap_spec=_PHASE,
        singularities=(
            *(singularity("zero", z, 3) for z in dodeca_v),
            *(singularity("pole", p, 2) for p in ico_e),
        ),
        story="The third degree-60 icosahedral ratio, and the one with no counterpart "
        "elsewhere in the family: thirty double spikes at the icosahedron's edge midpoints -- "
        "the vertices of an icosidodecahedron -- over twenty triple pits on the dodecahedron. "
        "The densest piece here, and the most sea-urchin-like. Full I_h symmetry, and no "
        "feature at infinity: both forms are full degree, so the answer key is complete.",
        tags=("ornament", "polyhedral", "poles"),
        pole_order=2,
        resolution=400,
        clip_ornament_to_domain=False,
    )

    return {p.id: p for p in presets}


catalog = _Catalog(_build_presets())


__all__ = [
    "FunctionPreset",
    "catalog",
    "domain_from_spec",
    "cmap_from_spec",
    "scaling_from_spec",
    "singularity",
    "roots_of_unity",
    "SINGULARITY_TYPES",
]
