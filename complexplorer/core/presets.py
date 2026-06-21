"""Function preset registry (``cp.catalog``).

A curated, metadata-rich, **serializable** description of complex functions, designed to be
the single source consumed by the gallery, CLI, STL object cards, and — under the roadmap's
games boundary — Godot game prototyping (which reimplements the math natively).

This module is deliberately **PyVista-free** (presets are data, not rendering) and imports
only the core/data layer. Distinct from ``complexplorer.api.Presets`` (plot-config presets):
this is the *function* registry, exposed as ``cp.catalog``.

A preset carries:

- ``func`` — the callable Complexplorer renders with (NOT serialized),
- ``expression`` — a string like ``"z / (z**10 - 1)"`` (Godot reimplements from it),
- ``domain_spec`` / ``cmap_spec`` / ``scaling_spec`` — plain dicts whose keys mirror the
  target constructor kwargs; every complex value is an ``[re, im]`` pair,
- ``singularities`` — hand-authored, exact answer keys (one record per location),
- ``id`` / ``title`` / ``story`` / ``tags``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..utils.validation import ValidationError
from .colormap import Chessboard, Colormap, LogRings, Phase, PolarChessboard
from .domain import Annulus, Disk, Domain, Rectangle
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
    func: Callable = field(repr=False)
    domain_spec: dict = field(default_factory=dict)
    cmap_spec: dict = field(default_factory=lambda: {"type": "Phase", "n_phi": 6})
    scaling_spec: str | dict = "balanced"
    singularities: tuple[dict, ...] = ()
    story: str = ""
    tags: tuple[str, ...] = ()

    def __post_init__(self):
        for record in self.singularities:
            singularity(
                record["type"], record["at"], record.get("order"), record.get("label", "")
            )  # validates; raises on bad records

    # -- live objects (instantiated on demand) --
    def domain(self) -> Domain:
        return domain_from_spec(self.domain_spec)

    def colormap(self) -> Colormap:
        return cmap_from_spec(self.cmap_spec)

    def scaling(self) -> dict:
        return scaling_from_spec(self.scaling_spec)

    # -- serialization (Godot interchange record) --
    def to_dict(self) -> dict:
        """JSON-ready record of everything EXCEPT the live ``func``."""
        return {
            "id": self.id,
            "title": self.title,
            "expression": self.expression,
            "domain_spec": dict(self.domain_spec),
            "cmap_spec": dict(self.cmap_spec),
            "scaling_spec": self.scaling_spec,
            "singularities": [dict(s) for s in self.singularities],
            "story": self.story,
            "tags": list(self.tags),
        }


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
# Curated content (~20) — exact, hand-authored answer keys
# --------------------------------------------------------------------------------------

_RECT4 = {"type": "rectangle", "re_length": 4, "im_length": 4}
_RECT8 = {"type": "rectangle", "re_length": 8, "im_length": 4}
_ANNULUS = {"type": "annulus", "inner_radius": 0.2, "outer_radius": 3}
_PHASE = {"type": "Phase", "n_phi": 6, "auto_scale_r": True}


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
