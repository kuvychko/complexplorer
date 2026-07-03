## Why

M1 restructured `examples/` and M2 rebuilt the gallery, but the four Jupyter tutorials under
`examples/notebooks/` were deliberately left for last — they need *content* rework, not just
moves. Two of them call the removed matplotlib-3D `plot_landscape` (and `getting_started` still
branches on the deleted `HAS_PYVISTA` flag), their embedded output images are stale mpl-3D
renders, and none of them cover the 3.0 headline additions (Riemann surfaces, the preset
registry/gallery). This is M3, the final milestone of `migrate-examples-and-docs`: modernize all
four notebooks onto the 3.0 surface and make "it runs top-to-bottom" a verifiable, repeatable
guarantee.

## What Changes

- **Static PyVista backend in every notebook.** Each notebook gains a setup cell
  (`import pyvista as pv; pv.set_jupyter_backend('static')`) so its `_pv` calls render as
  embedded static images and execute headlessly under `nbconvert` (a spike confirmed 2D, 3D
  landscape, and Riemann-surface cells all execute and embed images this way). A markdown note
  points readers to `notebook=False` / the terminal scripts for interactive, high-quality views.
- **Excise the obsolete matplotlib-vs-PyVista narrative** (careful review found the
  `plot_landscape` calls are not standalone cells to swap — they are dead comparisons):
  `getting_started` cell 18 and `advanced_features` cell 16 **time `plot_landscape_pv` against
  `cp.plot_landscape`** to prove "PyVista is Nx faster" — with mpl-3D removed there is nothing to
  compare, so these cells are deleted (or reduced to a one-line "PyVista is the sole 3D backend"
  note). `getting_started` cell 14's `else:` branch is a dead mpl fallback (dropped), and cell 16
  is an obsolete "inline (notebook=True) vs external (notebook=False) quality" demo that is reworked
  into a short note (the static backend now handles inline; the terminal scripts handle interactive).
  Drop the `HAS_PYVISTA` try/except and all `if HAS_PYVISTA:` guards (always available at 3.0).
- **Normalize `_pv` render kwargs across all four notebooks.** ~15 cells pass
  `notebook=`/`show=`/`off_screen=`; a spike confirmed `notebook=False` **opens an external window**
  (hangs/pops under `nbconvert`, embeds no inline image). These kwargs are stripped so the static
  backend governs and images embed cleanly — a whole-set edit, not just the two broken notebooks.
- **New content (full scope):**
  - `advanced_features` → a **Riemann-surface** section (`riemann_surface_pv`: power n=2/3, log).
  - `api_cookbook` → a **preset-registry** recipe (`cp.catalog.get/list/filter`, pointer to
    `cp gallery` / `examples/showcase.py`).
  - `getting_started` → a **3.0 backend-policy** note (matplotlib 2D, PyVista 3D, the static
    notebook backend).
- **Keep executed output (committed).** Notebooks are re-executed and their fresh static-image
  output is committed (replacing the stale mpl-3D images), so GitHub/nbviewer render them richly.
  Regeneration command: `jupyter nbconvert --to notebook --execute --inplace examples/notebooks/*.ipynb`.
- **nbmake verification harness.** `pytest --nbmake examples/notebooks/` re-executes every
  notebook and asserts no cell errors — the execution DoD. It is **opt-in** (not collected by the
  default `pytest` run and not wired into CI; per the roadmap, notebook execution stays a local
  gate). A new `[examples]` extra in `pyproject` declares `nbmake`, `nbconvert`, `ipykernel`.
- **Roadmap correction:** the M3 bullet's "perceptual-family coverage" is replaced with
  "registry + Riemann-surface coverage" (the perceptual colormap family does not exist).

## Capabilities

### Modified Capabilities
- `examples`: extend with the notebook contract — the tutorials execute top-to-bottom on the 3.0
  surface (verified by an nbmake harness), carry committed executed output via the static PyVista
  backend, cover the 3.0 feature surface (Riemann surfaces, the registry/gallery), and reference
  no removed symbol.

## Impact

- **Edited (re-executed, output committed):** all four `examples/notebooks/*.ipynb`.
- **New:** an `[examples]` extra in `pyproject.toml`; an nbmake opt-in marker/config (e.g. in
  `pyproject`/`pytest.ini`); optionally a short `examples/notebooks/README.md` note on running and
  verifying.
- **Edited:** `openspec/ROADMAP.md` (M3 bullet wording; status flips at archive time).
- **Untouched:** all library code and APIs; the M2 gallery; the deterministic manifests. M3 is
  examples-and-docs only.
- **Repo size:** keep-output adds ~10–15 MB of embedded notebook images (a conscious, accepted
  tradeoff, consistent with M2's committed gallery).
- **Not in CI:** nbmake is a local gate (PyVista-heavy notebooks take minutes; off-screen
  screenshots crash on headless CI).
