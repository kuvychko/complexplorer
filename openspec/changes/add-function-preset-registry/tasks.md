# Tasks — add-function-preset-registry

## 1. Spec factories (no core changes)
- [ ] 1.1 Add `domain_from_spec(spec)` mapping `type` → `Domain` subclass (rectangle, disk,
      annulus, and any other the curated presets use). Spec keys mirror constructor kwargs
      (`re_length`/`im_length`, `radius`, `inner_radius`/`outer_radius`, `center`); a
      complex `center` is an `[re, im]` pair converted to complex. Unknown type →
      `ComplexplorerError`.
- [ ] 1.2 Add `cmap_from_spec(spec)` mapping `type` → `Colormap` subclass (Phase + the
      handful the curated presets use), keys = constructor kwargs. Unknown type →
      `ComplexplorerError`.
- [ ] 1.3 Resolve `scaling_spec`: accept a named `SCALING_PRESETS` key, an inline
      `{method, params}` (the `SCALING_PRESETS` shape), or both (bare string resolves via
      `get_scaling_preset`).
- [ ] 1.4 Unit tests for each factory incl. the unknown-type error.

## 2. FunctionPreset model (core/presets.py — PyVista-free)
- [ ] 2.1 Define `FunctionPreset` (frozen dataclass): `id`, `title`, `expression`,
      `callable`, `domain_spec`, `cmap_spec`, `scaling_spec`, `singularities`, `story`,
      `tags`.
- [ ] 2.2 Define the singularity record shape `{type, at:[re,im], order, label?}` with
      `type ∈ {zero, pole, essential, branch_point}` and validation (order null only for
      essential).
- [ ] 2.3 Implement `to_dict()` — JSON-ready, excludes the `callable`.
- [ ] 2.4 Assert the module imports with no PyVista/matplotlib dependency (a test pins this).
- [ ] 2.5 Unit tests: field access, `to_dict()` JSON-serializable, specs round-trip via the
      factories, singularity validation.

## 3. Registry + curated content
- [ ] 3.1 Implement the registry: `get(id)`, `list()`, `filter(tag=...)` over a module-level
      preset dict (mirrors `SCALING_PRESETS`).
- [ ] 3.2 Author ~20 curated presets with hand-authored, EXACT answer keys (one record per
      location): basic maps (`z`, `z**2`, `1/z`, `(z-1)/(z+1)`), singularities (poles
      orders 1–3, `exp(1/z)`), branches (`sqrt(z)`, `log(z)`, `z**(1/3)` — principal branch,
      noted in `story`, `branch_point` declared), ornaments (`z/(z**10-1)` = one order-1
      zero at origin + ten order-1 poles at the 10th roots of unity), a couple of dynamics
      snapshots. Each: recommended specs, story, tags.
- [ ] 3.3 Tests: every curated preset's specs build via the factories and round-trip; the
      callable is finite on a sample away from its declared singularities; ids unique.

## 4. Public API + naming
- [ ] 4.1 Export the registry as **`cp.catalog`** (function preset registry), named to avoid
      a case-only collision with the existing `cp.Presets` (plot-config presets, unchanged).
      Update `__all__`.
- [ ] 4.2 Test asserting `cp.catalog` (registry) and `cp.Presets` (plot configs) both exist
      and are different things.

## 5. Docs & close out
- [ ] 5.1 Short docs page / section: the preset model, the spec-dict + answer-key design,
      and the serialization purpose (Godot interchange / object cards).
- [ ] 5.2 Run `pytest tests/` green; `ruff` clean; `openspec validate --specs`.
- [ ] 5.3 Update `openspec/ROADMAP.md` STATUS for this change.
