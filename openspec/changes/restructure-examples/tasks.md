## 1. New directory skeleton

- [ ] 1.1 Create `examples/notebooks/` and `examples/scripts/` (keep the existing empty
  `examples/notebooks/` if already present)
- [ ] 1.2 `git mv` the 4 tutorials into `examples/notebooks/`: `getting_started.ipynb`,
  `advanced_features.ipynb`, `api_cookbook.ipynb`, `stl_export_demo.ipynb`
- [ ] 1.3 `git mv` the one clean script into `examples/scripts/`: `interactive_showcase.py`
  (it already uses `plot_landscape_pv`/`riemann_pv` with `return_plotter`; do NOT edit it)

## 2. Retire legacy generators and obsolete material

- [ ] 2.1 Delete `examples/generate_gallery.py` (legacy hand-rolled generator)
- [ ] 2.2 Delete `examples/gallery/generate_gallery_images.py` and
  `examples/gallery/README_scripts.md` (2nd hand-rolled generator + its readme)
- [ ] 2.3 Delete `examples/modulus_scaling_showcase.py` — it is mpl-3D architecture
  (`add_subplot(projection='3d')` grid + `ax=` into `cp.plot_landscape`), NOT a mechanical
  swap; M2 rebuilds modulus-scaling coverage as a PyVista demo
- [ ] 2.4 Delete `examples/archive/` and `examples/old/` (obsolete mpl-era; git history
  preserves them)
- [ ] 2.5 Confirm the 27 hand-named PNGs under `examples/gallery/` are LEFT IN PLACE
  (M2 replaces them; nothing should 404 mid-flight)

## 3. Scripts on the 3.0 surface

- [ ] 3.1 Verify the moved `examples/scripts/interactive_showcase.py` imports and uses only
  3.0-surface symbols (no `cp.plot_landscape` / `cp.riemann` / `HAS_PYVISTA`); no rendering
  rewrite occurs in M1 — the only surviving script is already clean

## 4. Rewrite the examples README

- [ ] 4.1 Rewrite `examples/README.md` to describe the new `notebooks/` + `scripts/` +
  `gallery/` layout, the registry-driven model (`cp.catalog` / `cp gallery`), and the 3.0
  backend policy (matplotlib 2D, PyVista 3D); reference only files that exist
- [ ] 4.2 Fold the contents of `examples/README_interactive_demo.md` into a section of
  `examples/README.md`, then delete `README_interactive_demo.md`

## 5. Documentation link sweep (notebook/script paths only)

- [ ] 5.1 Update `README.md` example links to the new paths
  (`examples/notebooks/*.ipynb`, `examples/scripts/interactive_showcase.py`); repoint the
  `examples/modulus_scaling_showcase.py` reference (the modulus-scaling section) to the
  modulus-scaling prose/`cp gallery` until M2 reintroduces a PyVista demo
- [ ] 5.2 Update `docs/README.md` notebook/script links to the new paths
- [ ] 5.3 In `docs/gallery/README.md`, fix the notebook links and remove the already-dead
  `plots_example.ipynb` / `domains_cmaps_example.ipynb` references (leave gallery **image**
  links for M2)
- [ ] 5.4 In `docs/pyvista_usage_guide.md`, replace the dead `examples/interactive_demo.py`
  references with `examples/scripts/interactive_showcase.py`
- [ ] 5.5 Repo-wide `grep` for `examples/` in `*.md` to confirm no notebook/script link points
  at a non-existent or `archive/`-only file

## 6. Guard test and verification

- [ ] 6.1 Add a structural guard test (e.g. `tests/unit/test_examples_structure.py`) that
  scans `examples/**/*.py` and asserts none reference `plot_landscape`, `pair_plot_landscape`,
  the 3D `riemann`, `HAS_PYVISTA`, or `HAS_STL_EXPORT`
- [ ] 6.2 (Optional) Extend the guard to assert the layout invariants: no `examples/archive`
  or `examples/old` dir, and the legacy generator scripts are absent
- [ ] 6.3 Run `pytest tests/` — full suite green
- [ ] 6.4 Run `openspec validate --specs` and `openspec validate restructure-examples` — clean
