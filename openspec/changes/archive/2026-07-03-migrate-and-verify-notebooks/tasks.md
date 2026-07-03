## 1. Tooling

- [x] 1.1 Add an `[examples]` optional-dependency group to `pyproject.toml`
  (`nbmake`, `nbconvert`, `ipykernel`); have `[dev]` include it
- [x] 1.2 Configure nbmake to be opt-in: ensure `examples/notebooks/` is NOT collected by the
  default `pytest` run (default `testpaths`/config unchanged), and document the
  `pytest --nbmake examples/notebooks/` invocation

## 2. getting_started.ipynb

- [x] 2.1 Add a setup cell: `import pyvista as pv; pv.set_jupyter_backend('static')`, plus a
  markdown note on the 3.0 backend policy (matplotlib 2D / PyVista 3D, static notebook backend)
  and how to switch to an interactive backend / the terminal scripts for high quality
- [x] 2.2 Remove the `HAS_PYVISTA` try/except and all `if HAS_PYVISTA:` guards (always available at
  3.0); drop cell 14's mpl `else` fallback and run its `_pv` body unconditionally
- [x] 2.3 Delete the dead performance-comparison cell (cell 18, times `plot_landscape_pv` vs the
  removed `cp.plot_landscape`); optionally leave a one-line "PyVista is the sole 3D backend" note
- [x] 2.4 Rework the obsolete "inline vs external quality" cell (cell 16) into a short note —
  the static backend handles inline rendering; point to the terminal scripts for interactivity
- [x] 2.5 Strip `notebook=`/`show=`/`off_screen=` from the remaining `_pv` calls so the static
  backend governs and images embed (spike: `notebook=False` opens a window / no inline image)

## 3. advanced_features.ipynb

- [x] 3.1 Add the static-backend setup cell
- [x] 3.2 Delete the dead performance-comparison cell (cell 16, times `plot_landscape_pv` vs the
  removed `cp.plot_landscape`)
- [x] 3.3 Strip `notebook=`/`show=`/`off_screen=` from all `_pv` calls (the static backend governs)
- [x] 3.4 Add a **Riemann-surface** section using `riemann_surface_pv` (power n=2, power n=3,
  log), with a short note on surface-vs-sphere; ensure any colormap material references only
  implemented colormaps (`Phase`/`Chessboard`/`PolarChessboard`/`LogRings`)

## 4. api_cookbook.ipynb

- [x] 4.1 Add the static-backend setup cell
- [x] 4.2 Strip `notebook=`/`show=`/`off_screen=` from all `_pv` calls (the static backend governs)
- [x] 4.3 Add a **preset-registry** recipe section: `cp.catalog.list()`, `cp.catalog.get(<id>)`
  (`.func`/`.domain()`/`.colormap()`), `cp.catalog.filter(tag=...)`, and a pointer to the gallery
  producer (`cp gallery` / `examples/showcase.py`)

## 5. stl_export_demo.ipynb

- [x] 5.1 Add the static-backend setup cell; strip `notebook=` from its `_pv`/`riemann_pv` calls
- [x] 5.2 Modernize any prose to the 3.0 surface (verify it still reflects the current
  `OrnamentGenerator` API)

## 6. Regenerate output and verify

- [x] 6.1 Regenerate committed output:
  `jupyter nbconvert --to notebook --execute --inplace examples/notebooks/*.ipynb`
  (fresh static images replace the stale mpl-3D output)
- [x] 6.2 Verify the DoD: `pytest --nbmake examples/notebooks/` passes (all four execute with no
  cell error)
- [x] 6.3 Confirm the M1 examples-structure guard still passes and the default `pytest tests/`
  run is green and does NOT execute notebooks
- [x] 6.4 Spot-check each committed notebook renders embedded images on a viewer (no stale
  mpl-3D plots remain)

## 7. Docs and roadmap

- [x] 7.1 Add a short "running & verifying the notebooks" note to `examples/README.md` (the
  regenerate and `pytest --nbmake` commands)
- [x] 7.2 Fix the `openspec/ROADMAP.md` M3 bullet: "perceptual-family coverage" →
  "registry + Riemann-surface coverage"
- [x] 7.3 Run `openspec validate migrate-and-verify-notebooks` and `openspec validate --specs`
  — clean
