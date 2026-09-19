# Curate High-Level API for 3.0.0

## Why

The 3.0.0 release is imminent and the high-level API surface ships broken promises:
`create_animation()` and `compare_functions()` raise `NotImplementedError`, `analyze_function()`
advertises zero/pole detection that is a print-stub, and three names (`quick_plot`, `visualize`,
`explore`) alias one function. A dead plotter base-class layer (`plotting/base.py`) is exported
but used by nothing. Removing any of this *after* 3.0 would be a breaking change; curating it now
means the major version ships a surface where everything exported actually works. The baseline
`high-level-api` spec also describes a `show` entry point that has never existed in the code and
must be trued up.

## What Changes

- **BREAKING** (vs 2.x; rides the 3.0.0 release): remove `create_animation()` and
  `compare_functions()` from `complexplorer/api.py` — both are `NotImplementedError` stubs.
- **BREAKING**: remove `analyze_function()` — its headline zero/pole detection is unimplemented
  (prints a "not yet implemented" note); it adds nothing over `quick_plot`. Verified unused in
  examples, notebooks, and tests beyond its own unit test.
- **BREAKING**: remove the `visualize` and `explore` aliases; `quick_plot` becomes the single
  high-level entry point. Verified unused in examples/notebooks.
- Remove dead scaffolding: delete `complexplorer/plotting/base.py` (Base2DPlotter/Base3DPlotter/
  BasePlotter/PlotConfig — referenced only by commented-out TODOs), its star-export in
  `complexplorer/plotting/__init__.py`, its test file `tests/unit/plotting/test_base_plotting.py`,
  and the two "TODO: Implement base classes" comments in `plotting/matplotlib/plot_2d.py`.
  `complexplorer/export/base.py` is live and is kept.
- Clarify the `api.Presets` docstring to distinguish plot-config presets from the
  `cp.catalog` `FunctionPreset` registry (naming collision foot-gun).
- Update the top-level `__all__` in `complexplorer/__init__.py` and the 3.0.0 "Removed" section
  of `CHANGELOG.md`.
- True up the `high-level-api` spec: drop the "Quick exploration entry point" requirement (it
  describes a `show(range...)` function that never existed) and re-anchor the remaining
  requirements on `quick_plot`.

## Capabilities

### New Capabilities

_None._

### Modified Capabilities

- `high-level-api`: the quick-exploration (`show`) requirement is removed as never-implemented
  fiction; the mode-dispatching entry point requirement is re-anchored on `quick_plot` as the
  sole exploration entry point (aliases removed); a new requirement pins the curated surface
  (no exported callable may be an unimplemented stub).

## Impact

- `complexplorer/api.py` — deletions + docstring clarification (`Presets` stays).
- `complexplorer/__init__.py` — `__all__` and imports lose `analyze_function`, `visualize`,
  `explore`.
- `complexplorer/plotting/__init__.py` — drop `from .base import *`.
- `complexplorer/plotting/base.py` — deleted.
- `complexplorer/plotting/matplotlib/plot_2d.py` — TODO comments removed (no behavior change).
- `tests/unit/plotting/test_base_plotting.py` — deleted; any tests exercising removed API
  symbols (`tests/unit/test_api_quick_plot.py` etc.) updated.
- `CHANGELOG.md` — 3.0.0 Removed section gains these entries.
- No dependency changes. CLI, gallery, presets, plotting backends unaffected.
