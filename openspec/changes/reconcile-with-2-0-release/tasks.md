## 1. Guards before refactoring

- [x] 1.1 Record a golden fixture `tests/data/phase_golden.npz` of the current `Phase(...).rgb(z)`:
  - Use a fixed grid that includes zeros, poles, NaN, and ±inf.
  - Cover the configuration matrix in design D2.
  - Add `tests/unit/core/test_phase_golden.py` asserting exact equality. It must pass on the
    unmodified code.
- [x] 1.2 Freeze the 2.0.0 public surface as `tests/data/v2_0_0_all.txt`, generated from
  `git show v2.0.0:complexplorer/__init__.py` (`__all__` plus its conditional `extend` calls).

## 2. Exceptions

- [x] 2.1 Add `ColormapError(ValidationError)` to `complexplorer/exceptions.py`.
  - Export it from `complexplorer/__init__.py` and add it to `__all__`.
  - Test that it is caught as `ValidationError`, `ComplexplorerError`, and `ValueError`, and that
    it is exported.

## 3. Colormap port

- [x] 3.1 Port `sigmoid` and `circular_interpolate` from `v2.0.0:complexplorer/core/functions.py`
  into `core/functions.py`. Do not add them to the top-level exports. Add unit tests.
- [x] 3.2 Port `v2.0.0:complexplorer/core/color_utils.py` to `complexplorer/core/color_utils.py`
  (modern typing, ruff-clean). Test that:
  - OkLCh→sRGB output is in gamut
  - `clip_to_gamut` bounds hold
  - `cubehelix` and `hsl_to_rgb` ranges hold
- [x] 3.3 Port `BasePhasePortrait` (phase-sector/modulus modulation, auto-scale, `v_base`
  validation, and the `_compute_colors` hook).
  - Rebase `Phase` onto it with `phase_sectors`.
  - Add `emphasize_unit_circle` / `unit_circle_strength` / `unit_circle_color`.
  - The golden test (1.1) must pass unchanged.
- [x] 3.4 Port the nine families: `OklabPhase`, `PerceptualPastel`, `AnalogousWedge`,
  `DivergingWarmCool`, `Isoluminant`, `CubehelixPhase`, `InkPaper`, `EarthTopographic`, and
  `FourQuadrant`.
  - Adapt mechanically only: typing, imports, ruff.
  - Confirm each goes through `Colormap.hsv()`, so the non-finite contract applies.
- [x] 3.5 Switch `PolarChessboard` to `phase_sectors`, and restore the 2.0 construction
  validation (`ColormapError`) in `Chessboard`, `PolarChessboard`, and `LogRings`.
- [x] 3.6 Add a keyword-only `n_phi` sentinel to `BasePhasePortrait`, `Phase`,
  `PolarChessboard`, and the families. It raises `ValidationError` naming `phase_sectors`.
- [x] 3.7 Export the nine families from `complexplorer/core/__init__.py` and
  `complexplorer/__init__.py` (and `__all__`), and update the `colormap.py` module docstring.
- [x] 3.8 Port the 45 tests from `v2.0.0:tests/unit/core/test_colormap.py`, adapted to the rev3
  contract.
- [x] 3.9 Add a parametrized contract test over every exported concrete colormap:
  - output shape, gamut, and HSV/RGB consistency
  - non-finite values and `outmask` get the out-of-domain color
  - determinism across repeated calls
  - `cp.plot(..., legend=True)` renders on an Agg figure

## 4. Rename call sites (`n_phi` → `phase_sectors`)

- [x] 4.1 `core/presets.py` `cmap_spec` records and `cmap_from_spec`. An `n_phi` key must raise
  `ValidationError` naming `phase_sectors`. Update the preset tests.
- [x] 4.2 The CLI `--cmap phase:N` shorthand in `cli/main.py`, and its tests.
- [x] 4.3 The remaining source call sites:
  - `api.py` (`Presets` bundles and the `quick_plot` default colormap)
  - the default colormaps in `plotting/matplotlib/plot_2d.py` and `plotting/pyvista/*`
  - `mesh/*`
  - `export/stl/ornament_generator.py`
- [x] 4.4 Update the remaining test references. Then grep and verify that no colormap `n_phi`
  remains outside the rejection tests and the PyVista `_REMOVED_KWARGS` mesh mapping, which is
  intentionally unchanged.
- [x] 4.5 Update the snippets in `README.md`, `CLAUDE.md`, `docs/*.md`, `examples/scripts/*`,
  `examples/showcase.py`, and `complexplorer/export/stl/README.md`.

## 5. Gallery

- [x] 5.1 Bump the `cp.gallery` manifest `schema_version` from 2 to 3 in `complexplorer/gallery.py`
  and update its tests.
  - Regenerate `examples/gallery/index.json` and the `card.json` files.
  - Confirm they are byte-identical across two runs.
  - Revert any PNG-only churn.
- [x] 5.2 In `examples/showcase.py`:
  - Extend `_colormap_family()` with the nine families (snippets use `phase_sectors`).
  - Add a minimal `--only colormaps` flag.
  - Regenerate the colormap section, `showcase.json`, and `docs/gallery/gallery.generated.md`.
- [x] 5.3 Extend `tests/unit/test_showcase_bundle.py`: every exported concrete colormap class
  appears in the `colormaps` section, and no entry names an unexported class.

## 6. Notebooks

- [ ] 6.1 Create `examples/notebooks/color_and_accessibility.ipynb` from 2.0 `04_colormaps_comprehensive` and
  `05_accessibility_cvd`:
  - Include the colormap tour, the CVD simulation, and recommendations.
  - Use the 3.0 surface: static PyVista backend and `phase_sectors`.
  - Commit it executed.
- [ ] 6.2 Fold composite-domain material from 2.0 `02_domains_advanced` into
  `examples/notebooks/advanced_features.ipynb`, where it isn't already covered. Re-execute.
- [ ] 6.3 Rename `n_phi` → `phase_sectors` in the four existing rev3 notebooks and re-execute them.
- [ ] 6.4 Port the 2.0 application notebooks `app_01`–`app_04` into
  `examples/notebooks/applications/`.
  - Keep each only if `pytest --nbmake` passes.
  - Record each dropped notebook in the `openspec/ROADMAP.md` 3.1+ backlog.
- [ ] 6.5 Update `examples/README.md` and `tests/unit/test_examples_structure.py` for the new
  notebooks.

## 7. Migration inventory and docs scaffold

- [x] 7.1 Add the `docs/migration-3.0.md` skeleton:
  - An inventory table: one row per 2.0.0-only name, with its 3.0 replacement or "removed — no
    replacement".
  - A note on the `n_phi` → `phase_sectors` rename, for 1.x and rev3-build users.
  - A note on the `index.json` `schema_version` bump.
- [x] 7.2 Add `tests/unit/test_v2_surface_accounted.py`: every name in
  `tests/data/v2_0_0_all.txt` is either in `complexplorer.__all__` or has a row in the inventory
  table.
- [x] 7.3 Recover `mkdocs.yml`, `docs/javascripts/mathjax.js`, and `.github/workflows/docs.yml`
  from `v2.0.0`.
  - Set the workflow trigger to `workflow_dispatch` only.
  - Add a header note that `publish-rev3-docs-site` owns them.

## 8. Verification and bookkeeping

- [ ] 8.1 Run the full gate and confirm it is green:
  - `pytest tests/`
  - `ruff check` and `ruff format --check complexplorer/ tests/`
  - `pytest --nbmake examples/notebooks/`
  - `openspec validate --specs`
  - `openspec validate reconcile-with-2-0-release`
  - gallery manifest byte-stability
- [ ] 8.2 Update `CLAUDE.md` (the colormap list and the `phase_sectors` quick reference) and flip
  C1's status in `openspec/ROADMAP.md`.
