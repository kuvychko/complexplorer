## Why

The 3.0 library work (`require-pyvista-3d-backend` + `add-riemann-surfaces`) left `examples/`
stale and self-contradictory. Two of four notebooks and one script call removed mpl-3D
functions (`cp.plot_landscape`); `examples/` carries **two** hand-rolled gallery generators
that predate — and now duplicate — the canonical registry-driven `cp.gallery`; and several
`README`/`docs` links already point at files that were moved into `archive/`
(`interactive_demo.py`, `plots_example.ipynb`). This is M1 of the phased
`migrate-examples-and-docs` rework (gallery-first M1→M2→M3): establish a clean, registry-
driven structure so M2 (gallery regeneration) and M3 (notebook migration) land on solid
ground.

## What Changes

- **New `examples/` layout**: `notebooks/` (the 4 tutorials), `scripts/` (the curated
  runnable demos), and `gallery/` (regenerated output, populated by M2). A rewritten
  `examples/README.md` maps the new structure.
- **Retire the two legacy hand-rolled gallery generators** — `examples/generate_gallery.py`
  and `examples/gallery/generate_gallery_images.py` (plus `gallery/README_scripts.md`). They
  predate the registry and are superseded by the library `cp.gallery` / `cp gallery`. (The
  library `complexplorer/gallery.py` is a *different* thing and is untouched.)
- **Cull `examples/archive/` and `examples/old/`** — obsolete mpl-era reference material;
  git history preserves it.
- **Delete the broken `modulus_scaling_showcase.py`** — it is architecturally matplotlib-3D
  (a `fig.add_subplot(projection='3d')` grid with `ax=` passed into `cp.plot_landscape`,
  `ax.view_init`, `plt.savefig`) and **cannot** be ported by a symbol swap; `plot_landscape_pv`
  is a PyVista renderer with no `ax=`. Its modulus-scaling coverage is rebuilt as a PyVista
  demo in **M2** (the change that owns PyVista screenshot rendering). `interactive_showcase.py`
  is already on the 3.0 surface (`plot_landscape_pv` / `riemann_pv` with `return_plotter`) and
  is kept as-is.
- **Sweep documentation links** to resolve against the new layout and remove the already-dead
  references: `README.md`, `docs/README.md`, `docs/gallery/README.md` (notebook/script links
  only — gallery *image* links are M2's job), `docs/pyvista_usage_guide.md`.
- **Out of scope (deferred):** regenerating gallery images + rewiring gallery image links and
  rebuilding the PyVista modulus-scaling demo (M2); notebook *content* rework + execution
  verification (M3). M1 moves, deletes, and sweeps links; it performs **no rendering rewrite**.

## Capabilities

### New Capabilities
- `examples`: the contract for the `examples/` directory and example/docs references —
  registry-driven (no parallel hand-rolled generator), a defined directory layout, and the
  invariant that no example or doc references a symbol or file removed at 3.0. M2 and M3
  extend this capability (gallery regeneration; notebook execution).

### Modified Capabilities
<!-- None. M1 changes no library runtime behavior; the library cp.gallery, the catalog, and
     all rendering APIs are untouched. -->

## Impact

- **Files moved:** `examples/*.ipynb` → `examples/notebooks/`; `interactive_showcase.py` →
  `examples/scripts/`.
- **Files deleted:** `examples/generate_gallery.py`, `examples/gallery/generate_gallery_images.py`,
  `examples/gallery/README_scripts.md`, `examples/modulus_scaling_showcase.py` (broken mpl-3D;
  rebuilt in M2), `examples/archive/`, `examples/old/`, `examples/README_interactive_demo.md`.
- **Files edited:** `examples/README.md` (rewrite), `README.md` + `docs/README.md` +
  `docs/gallery/README.md` + `docs/pyvista_usage_guide.md` (link sweep, incl. repointing the
  `modulus_scaling_showcase.py` reference to the modulus-scaling docs until M2 reintroduces it).
- **No library code, API, dependency, or test-of-library-behavior changes.** A small
  structural guard test MAY be added under `tests/` to enforce the "no removed-symbol
  references in examples" invariant.
- **Untracked artifacts** (`site/`, the empty `examples/notebooks/`, the root
  `complexplorer_phased_implementation_plan.md`) are not part of this change.
