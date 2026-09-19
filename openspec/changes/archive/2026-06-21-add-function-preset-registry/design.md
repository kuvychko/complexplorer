# Design — function preset registry

## Context

The preset is the **serializable interchange record** between Complexplorer and its
consumers (gallery, CLI, STL cards, and — via the games boundary — Godot, which
reimplements the math natively). So the data model is designed around *serialization* and
*exact answer keys*, with the live `callable` as the one non-serializable escape hatch.

Grounding in the current code:
- `core/scaling.py::SCALING_PRESETS` is a name→dict registry — the precedent the
  `scaling_spec` shape reuses.
- `Domain` / `Colormap` have **no** serialization today. We do not add it to the core
  classes; the preset holds plain spec-dicts and small factories instantiate on demand.
- `api.Presets` already exists and means **plot-config presets** (`publication_ready`,
  `interactive`, `high_contrast`). The obvious name `cp.presets` would collide with
  `cp.Presets` by **case only** — a footgun. The function registry is therefore exposed as
  **`cp.catalog`**; `api.Presets` is left unchanged.

## The model

```
FunctionPreset (core/presets.py — PyVista-free, a frozen dataclass)
  id            "pole_flower_10"
  title         "Pole Flower 10"
  expression    "z / (z**10 - 1)"          ← Godot reimplements from this
  callable      <fn>                        ← Complexplorer renders with this (NOT serialized)
  domain_spec   {"type": "annulus", "inner_radius": 0.2, "outer_radius": 3}
  cmap_spec     {"type": "Phase", "n_phi": 6, "auto_scale_r": true}
  scaling_spec  {"method": "arctan", "params": {...}}   (SCALING_PRESETS shape)
  singularities [{"type": "zero", "at": [0, 0], "order": 1},        # simple zero at origin
                 {"type": "pole", "at": [1, 0], "order": 1},        # ONE record per location
                 … 9 more simple poles at the other 10th roots of unity ]
  story         "A ring of ten simple poles around a single central zero…"
  tags          ["poles", "ornament", "canonical", "singularity-detective"]
```

> Spec keys mirror the real constructor kwargs (`Annulus(inner_radius, outer_radius,
> center)`, `Rectangle(re_length, im_length, center)`, `Disk(radius, center)`;
> `Phase(n_phi, …)`). Every complex value in any spec (a domain `center`, a singularity
> `at`) is an `[re, im]` pair — JSON has no complex type; factories convert on the way in.
> The pole flower `z/(z**10-1)` has a **simple** zero at the origin (order 1) and **ten
> separate** simple-pole records, one per 10th root of unity — answer keys are exact.
```

### Decisions

- **D1 — callable + expression, both author-provided.** No parsing/`eval`. The registry is
  curated and trusted; the two-source drift risk is low and covered by tests (e.g. the
  callable evaluates finite away from declared singularities).
- **D2 — spec-dicts, not live objects.** Presets serialize to JSON; live `Domain`/`Colormap`
  do not. Spec keys mirror the constructor kwargs and complex values are `[re, im]` pairs.
  Factories `domain_from_spec` / `cmap_from_spec` build live objects on demand;
  `scaling_spec` uses the `SCALING_PRESETS` shape (`{"method": ..., "params": {...}}`) and
  feeds the existing scaling machinery. Core classes untouched.
- **D3 — hand-authored, exact singularities.** `{type, at:[re,im], order:int|null, label?}`,
  `type ∈ {zero, pole, essential, branch_point}`. `order` semantics: multiplicity for
  zero/pole, branching order for branch_point, `null` for essential. One record **per
  location** (the pole flower is 1 zero + 10 pole records). Ground truth, not detector
  output.
- **D4 — `to_dict()` excludes the callable** and is the JSON record (stable,
  JSON-serializable). Round-trip invariant: `domain_from_spec(p.domain_spec)` and
  `cmap_from_spec(p.cmap_spec)` build live objects **of the expected type with the
  specified parameters** (verified behaviorally, e.g. via `domain.contains` on sample
  points — the core classes have no `__eq__`).
- **D5 — base preset is static.** No parameter machinery. Families (Phase 3) are a separate
  `FunctionFamily` whose `bind(**params)` *emits* a `FunctionPreset` — so the static base is
  the snapshot shape families produce. Nothing here precludes that.

## Spec factories (scope)

```
domain_from_spec({"type": "rectangle"|"disk"|"annulus"|...})  -> Domain
cmap_from_spec({"type": "Phase"|"Chessboard"|...})            -> Colormap
```

Cover only the `Domain`/`Colormap` subclasses the ~20 curated presets use; extend on
demand (avoid fake generality). Unknown `type` → a domain-specific error listing supported
types. The `type` key matches the class name (or a lowercase alias) so the mapping is
obvious and round-trips.

## Registry surface

```python
cp.catalog.get("pole_flower_10")       # -> FunctionPreset (KeyError-like if missing)
cp.catalog.list()                       # -> list[str] of ids (or list[FunctionPreset])
cp.catalog.filter(tag="singularity-detective")   # -> list[FunctionPreset]
```

Tags are the grouping mechanism the Phase-3 "game" preset sets reuse (no new structures).
Storage: a module-level dict of `FunctionPreset` (mirrors `SCALING_PRESETS`); ~20 entries
is comfortably a module, not a directory.

## Curated content (~20, illustrative)

| Category | Presets |
|---|---|
| Basic maps | `z`, `z**2`, `1/z`, `(z-1)/(z+1)` |
| Singularities | poles of orders 1–3, essential `exp(1/z)` |
| Branches | `sqrt(z)`, `log(z)`, `z**(1/3)` |
| Ornaments | pole flower `z/(z**10-1)`, branch shell `sqrt(z)` |
| Dynamics (static reps) | a Newton-step rational, a Julia-like snapshot |

Each carries hand-authored `singularities`, a recommended domain/cmap/scaling, a short
`story`, and tags. Multivalued presets (`sqrt(z)`, `log(z)`, `z**(1/3)`) use numpy's
**principal branch** in the callable, note that in the `story`, and declare a
`branch_point` singularity — so the answer key and Godot's native reimplementation agree on
the branch convention.

## Open questions (proposal-level, low risk)

- Whether `list()` returns ids or `FunctionPreset` objects (lean: ids; objects via `get`).
- Whether `scaling_spec` stores a named SCALING_PRESETS key, an inline
  `{method, params}`, or allows both (lean: allow both; a bare string resolves via
  `get_scaling_preset`).
- Exact JSON shape of complex numbers in `at` / specs — `[re, im]` pairs (chosen, since
  JSON has no complex type and Godot wants real pairs).

## Risks

| Risk | Mitigation |
|---|---|
| Registry name confused with `api.Presets` | Named `cp.catalog` (no case collision); a test asserts `cp.catalog` and `cp.Presets` both exist and are different things |
| callable/expression drift | Test: callable is finite on a sample away from declared singularities |
| Spec factory under-covers a preset's domain/cmap | Test: every curated preset's specs build via the factories and round-trip |
| Spec keys diverge from constructors / complex encoding | Spec keys = constructor kwargs; complex values are `[re, im]`; round-trip test enforces it |
| Scope creep into rendering | Hard non-goal; no PyVista/matplotlib import in `core/presets.py` |
