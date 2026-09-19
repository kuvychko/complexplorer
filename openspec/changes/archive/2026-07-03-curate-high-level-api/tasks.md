# Tasks — curate-high-level-api

## 1. Pre-flight verification

- [x] 1.1 Repo-wide grep for `analyze_function`, `visualize`, `explore`, `create_animation`,
      `compare_functions`, `plotting.base`, `Base2DPlotter`, `Base3DPlotter`, `BasePlotter`,
      `PlotConfig` to enumerate every reference (code, tests, examples, docs) before deleting

## 2. Remove API stubs and aliases

- [x] 2.1 Delete `create_animation()`, `compare_functions()`, `analyze_function()`, and the
      `visualize = quick_plot` / `explore = quick_plot` aliases from `complexplorer/api.py`;
      prune `api.__all__` and now-unused imports (e.g. `numpy` if orphaned)
- [x] 2.2 Clarify the `Presets` class docstring: plot-config presets, distinct from the
      `cp.catalog` `FunctionPreset` registry
- [x] 2.3 Update `complexplorer/__init__.py`: drop `analyze_function`, `visualize`, `explore`
      from imports and `__all__`

## 3. Remove dead plotting base layer

- [x] 3.1 Delete `complexplorer/plotting/base.py` and remove `from .base import *` from
      `complexplorer/plotting/__init__.py`
- [x] 3.2 Delete `tests/unit/plotting/test_base_plotting.py`
- [x] 3.3 Remove the two "TODO: Implement base classes / Add base class" comments in
      `complexplorer/plotting/matplotlib/plot_2d.py`

## 4. Tests and docs

- [x] 4.1 Update any tests referencing removed symbols (check `tests/unit/test_api_quick_plot.py`,
      integration tests) so the suite is green
- [x] 4.2 Add regression coverage for the curated surface: importing removed names raises
      `ImportError`; `complexplorer.__all__` contains `quick_plot`/`Presets` and none of the
      removed names
- [x] 4.3 Update `CHANGELOG.md` 3.0.0 "Removed" section (each removed symbol + replacement)
- [x] 4.4 Sweep README and `docs/` for mentions of removed symbols

## 5. Verification

- [x] 5.1 `pytest tests/` green
- [x] 5.2 `python -c "import complexplorer as cp; print(cp.__all__)"` shows curated surface
- [x] 5.3 `openspec validate --specs` passes for the delta spec
