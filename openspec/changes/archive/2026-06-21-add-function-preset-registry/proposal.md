# Add function preset registry

## Why

Phase 2 (v2.3) makes Complexplorer usable as a tool, not just a library — and almost
everything downstream needs the same thing first: a **named, metadata-rich description of a
complex function**. The gallery generator, the CLI, STL object cards, and the Phase-3 game
prototyping assets all consume it. Building it once is the leverage (see
`openspec/ROADMAP.md`, Phase 2).

Crucially, under the roadmap's **games boundary** (Complexplorer prototypes; Godot runs the
game with the math reimplemented natively for mobile), the preset is the **serializable
interchange record handed to Godot** — not just a gallery convenience. That reframes the
preset model: serialization and exact, hand-authored answer keys are the design center.

## What changes

Add a new `function-presets` capability — `complexplorer/core/presets.py`, **PyVista-free**
(presets are data, not rendering):

- **`FunctionPreset`** carrying both a `callable` (Complexplorer renders with it) and an
  `expression` string, e.g. `"z / (z**10 - 1)"` (Godot reimplements from it); plus
  `id`, `title`, `story`, `tags`.
- **Serializable spec-dicts, not live objects:** `domain_spec`, `cmap_spec`,
  `scaling_spec`. Spec keys mirror the constructor kwargs (e.g.
  `{"type": "annulus", "inner_radius": 0.2, "outer_radius": 3}`); `scaling_spec` reuses the
  existing `SCALING_PRESETS` shape (`{"method": ..., "params": {...}}`). Every complex value
  in a spec is encoded as an `[re, im]` pair (JSON has no complex type). Small factory
  helpers `domain_from_spec()` / `cmap_from_spec()` instantiate live `Domain`/`Colormap` on
  demand. The core `Domain`/`Colormap` classes are **not** modified.
- **`singularities`** — a list of hand-authored, exact answer keys:
  `{type, at: [re, im], order, label?}` with `type ∈ {zero, pole, essential, branch_point}`.
  Ground truth for prototyping/validation, **not** computed (the `analyze_function`
  detector stays a separate future tool).
- **`preset.to_dict()`** — a JSON-ready record of everything except the callable, so the
  preset round-trips losslessly through the spec factories.
- **Registry** exposed as **`cp.catalog`** — `get(id)`, `list()`, `filter(tag=...)`; ~20
  curated canonical presets with answer keys (e.g. `z`, `z**2`, `1/z`, `(z-1)/(z+1)`, pole
  flower `z/(z**10-1)`, `sqrt(z)`, `exp(1/z)`, …).

## Non-goals

- **No rendering, CLI, gallery, or JSON level-export** — those are `add-cli`,
  `add-gallery-generator`, `add-level-export`.
- **No parametrized families.** The base `FunctionPreset` is **static** (one callable +
  expression, fixed answer key). Parametrized families (Möbius, Julia, `z^(1/n)`,
  resonators) are a separate Phase-3 `FunctionFamily` extension whose `bind(**params)`
  emits an ordinary static preset (`cp.catalog`-shaped). This change must not *preclude*
  families (it doesn't), but does not build them.
- **No expression parsing / `eval`.** Authors provide the callable and expression directly
  (curated registry).
- **No automatic singularity detection.**

## Impact

- New: `complexplorer/core/presets.py`; new `function-presets` capability spec; public API
  surface **`cp.catalog`** (the function preset registry). Named `catalog` deliberately to
  avoid a case-only collision with the existing `api.Presets` (*plot-config* presets, which
  stay as-is).
- Touched: `complexplorer/__init__.py` (export `catalog`); reuse of
  `core/scaling.py::SCALING_PRESETS`.
- Risk: low. Pure additive data layer, PyVista-free, no core class changes. Main care
  items: keep the spec factories scoped to the subclasses the curated presets actually use
  (extended on demand).
