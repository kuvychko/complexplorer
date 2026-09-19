## 1. New directory skeleton

- [x] 1.1 Create `examples/notebooks/` and `examples/scripts/` (keep the existing empty
  `examples/notebooks/` if already present)
- [x] 1.2 `git mv` the 4 tutorials into `examples/notebooks/`: `getting_started.ipynb`,
  `advanced_features.ipynb`, `api_cookbook.ipynb`, `stl_export_demo.ipynb`
- [x] 1.3 `git mv` the one clean script into `examples/scripts/`: `interactive_showcase.py`
  (it already uses `plot_landscape_pv`/`riemann_pv` with `return_plotter`; do NOT edit it)

## 2. Retire legacy generators and obsolete material

- [x] 2.1 Delete `examples/generate_gallery.py` (legacy hand-rolled generator)
- [x] 2.2 Delete `examples/gallery/generate_gallery_images.py` and
  `examples/gallery/README_scripts.md` (2nd hand-rolled generator + its readme)
- [x] 2.3 Delete `examples/modulus_scaling_showcase.py` — it is mpl-3D architecture
  (`add_subplot(projection='3d')` grid + `ax=` into `cp.plot_landscape`), NOT a mechanical
  swap; M2 rebuilds modulus-scaling coverage as a PyVista demo
- [x] 2.4 Delete `examples/archive/` and `examples/old/` (obsolete mpl-era; git history
  preserves them). Note: `archive/` was tracked (`git rm`); `old/` was untracked (`rm`)
- [x] 2.5 Confirm the hand-named PNGs under `examples/gallery/` are LEFT IN PLACE
  (25 PNGs; M2 replaces them; nothing should 404 mid-flight)

## 3. Scripts on the 3.0 surface

- [x] 3.1 Verify the moved `examples/scripts/interactive_showcase.py` imports and uses only
  3.0-surface symbols (no `cp.plot_landscape` / `cp.riemann` / `HAS_PYVISTA`); no rendering
  rewrite occurs in M1 — the only surviving script is already clean

## 4. Rewrite the examples README

- [x] 4.1 Rewrite `examples/README.md` to describe the new `notebooks/` + `scripts/` +
  `gallery/` layout, the registry-driven model (`cp.catalog` / `cp gallery`), and the 3.0
  backend policy (matplotlib 2D, PyVista 3D); reference only files that exist
- [x] 4.2 Fold the contents of `examples/README_interactive_demo.md` into a section of
  `examples/README.md`, then delete `README_interactive_demo.md`

## 5. General-docs sweep (paths + removed-symbol API references; NOT docs/gallery — see D6)

- [x] 5.1 Update `README.md` example links to the new paths
  (`examples/notebooks/*.ipynb`, `examples/scripts/interactive_showcase.py`); repoint the
  `examples/modulus_scaling_showcase.py` reference (the modulus-scaling section) to the
  modulus-scaling prose/`cp gallery` until M2 reintroduces a PyVista demo
- [x] 5.2 Update `docs/README.md`: fix notebook/script links to the new paths AND fix the
  "3D Visualization (Matplotlib)" section that lists the removed `plot_landscape()`/
  `pair_plot_landscape()`/`riemann()` as available (replace with the PyVista path + a
  "removed in 3.0" note)
- [x] 5.3 In `docs/pyvista_usage_guide.md`, replace the dead `examples/interactive_demo.py`
  references with `examples/scripts/interactive_showcase.py` (its "removed in 3.0" prose is
  already correct — keep it)
- [x] 5.4 In `docs/development/backend-policy.md`, fix the stale "2.1 (now)" framing in the
  migration timeline (the removal already happened at 3.0)
- [x] 5.5 Do NOT modify `docs/gallery/README.md` (M2 regenerates it wholesale — D6)
- [x] 5.6 Repo-wide `grep` for `examples/` in `*.md` (excluding `docs/gallery/`) confirmed no
  M1-owned link points at a non-existent or `archive/`-only file (the two remaining MISS refs
  are gallery-only — M2's)

## 6. Guard test and verification

- [x] 6.1 Add a structural guard test (`tests/unit/test_examples_structure.py`) that
  scans `examples/**/*.py` and asserts none reference `plot_landscape`, `pair_plot_landscape`,
  the 3D `riemann`, `HAS_PYVISTA`, or `HAS_STL_EXPORT`
- [x] 6.2 Extend the guard to assert the layout invariants: no `examples/archive`
  or `examples/old` dir, the legacy generator scripts are absent, and `notebooks/`+`scripts/`
  exist
- [x] 6.3 Run `pytest tests/` — full suite green (428 passed)
- [x] 6.4 Run `openspec validate --specs` (16/16) and `openspec validate restructure-examples` — clean
