## Context

`examples/` accreted across the 1.x→2.x→3.0 evolution and is now internally inconsistent.
Current tree:

```
examples/
├── getting_started.ipynb        ✗ 2× cp.plot_landscape + 5× HAS_PYVISTA  (content → M3)
├── advanced_features.ipynb      ✗ 1× cp.plot_landscape                   (content → M3)
├── api_cookbook.ipynb           ✓ clean
├── stl_export_demo.ipynb        ✓ clean
├── interactive_showcase.py      ✓ clean (plot_landscape_pv/riemann_pv)  ── keep, move
├── modulus_scaling_showcase.py  ✗ mpl-3D ARCHITECTURE (subplot grid + ax=)  ── DELETE (M2 rebuilds)
├── generate_gallery.py          ⚠ legacy hand-rolled generator (pre-registry)  ── DELETE
├── gallery/
│   ├── generate_gallery_images.py  ⚠ 2nd hand-rolled generator             ── DELETE
│   ├── README_scripts.md           ── DELETE
│   └── <27 hand-named PNGs>         ── leave for now (M2 regenerates/replaces)
├── archive/ (11 mpl-era files)     ── DELETE (git history preserves)
├── old/ (1 file)                   ── DELETE
├── README.md                       ── REWRITE to new layout
└── README_interactive_demo.md      ── fold into README, then DELETE
```

The library already owns the canonical generator: `complexplorer/gallery.py`
(`cp.generate_gallery` + the `gallery` CLI), preset-driven from `cp.catalog` (17 presets),
deterministic, PyVista-free. The two `examples/` generators are obsolete duplicates with
their own parallel function lists — exactly the drift the registry exists to prevent.

Constraint discovered during exploration: README/docs **hard-link** example paths, and some
links are *already dead* (`docs/gallery/README.md` → `plots_example.ipynb`;
`docs/pyvista_usage_guide.md` → `interactive_demo.py`; both already in `archive/`). So moving
files is necessarily coupled to a documentation link sweep.

## Goals / Non-Goals

**Goals:**
- A clean, registry-driven `examples/` layout (`notebooks/`, `scripts/`, `gallery/`) that M2
  and M3 build on without further restructuring.
- Zero references to removed-at-3.0 symbols anywhere under `examples/`.
- Every notebook/script link in the repo's docs resolves to a file that exists.
- Retire both legacy generators and the obsolete `archive/`+`old/` clutter.

**Non-Goals:**
- Regenerating gallery images or rewiring gallery **image** links — that is M2 (the new
  `examples/showcase.py` produces the high-res renders and the manifest those links derive
  from). M1 leaves the 27 PNGs in place so nothing 404s mid-flight.
- Rewriting notebook **content** or verifying execution — that is M3. M1 *moves* the notebooks
  and may make the most trivial import-level swaps only if a notebook would fail to import on
  load; the substantive rework (and `nbconvert` execution) is M3's DoD.
- Any change to library runtime behavior, the catalog, or the library gallery.

## Decisions

**D1 — Move notebooks and scripts into subdirectories (not keep flat).**
`examples/notebooks/` + `examples/scripts/`. Rationale: the user asked for a "full rework /
clean structure"; subdirs separate the three concerns (tutorials, runnable demos, generated
output) and give M2's `gallery/` a clean home. Cost: a doc link sweep — accepted, because the
links need touching anyway (some are already dead). *Alternative considered:* keep files flat
at `examples/` root (minimal link churn) — rejected; it preserves the clutter the rework
exists to remove and still requires fixing the dead links.

**D2 — Delete `archive/` and `old/` rather than relocate them.**
They are obsolete mpl-era material; git history is the archive. Keeping them in-tree invites
confusion (they reference removed APIs) and future false-positive matches in the
"no removed-symbol" guard. *Alternative:* move to a top-level `docs/legacy/` — rejected as
preserving dead weight with no consumer.

**D3 — Delete `modulus_scaling_showcase.py` in M1; rebuild it as PyVista in M2. Defer
notebooks to M3.**
The careful-review pass found this script is **not** mechanically portable: it is built on the
matplotlib 3D API end-to-end — a `fig.add_subplot(2, 3, i+1, projection='3d')` grid, `ax=ax`
passed into `cp.plot_landscape`, `ax.view_init`, `ax.dist`, `plt.savefig`. `plot_landscape_pv`
is a PyVista renderer with no `ax=` parameter; reproducing the 2×3 multi-mode composition means
a `pv.Plotter(shape=(2,3))` screenshot rewrite. That is exactly the PyVista-screenshot rendering
machinery **M2** builds — so the demo is deleted here (like `archive/`) and reintroduced in M2,
rather than building that machinery twice. `interactive_showcase.py` needs nothing: it already
uses `plot_landscape_pv`/`riemann_pv` with `return_plotter` (its `ax=` calls are 2D `cp.plot`),
so M1 only *moves* it. The seam is therefore clean: **M1 deletes broken mpl-3D example content;
M2 rebuilds it as PyVista; M3 reworks the notebooks.** M1 contains **no rendering rewrite at
all**. *Alternatives considered:* (a) mechanical swap — impossible, the architectures differ;
(b) rewrite the PyVista demo inside M1 — rejected, it pulls M2's screenshot machinery forward
and bloats a change meant to be pure structure.

**D4 — A structural guard test enforces the invariant.**
Add a lightweight `tests/` check that scans `examples/**/*.py` for the forbidden symbols
(`plot_landscape`/`pair_plot_landscape`/3D `riemann`/`HAS_PYVISTA`/`HAS_STL_EXPORT`) and
asserts none appear. This makes the "no removed-symbol references" requirement a real,
regression-proof test rather than a one-time cleanup. Notebooks are excluded from the guard
(they are exercised by M3's execution pass, not a static grep). *Alternative:* manual review
only — rejected; it would silently rot.

**D5 — Fold `README_interactive_demo.md` into `examples/README.md`.**
One README is the entry point. The interactive-demo notes become a section there, and the
standalone file is deleted, removing a second source of truth.

**D6 — `docs/gallery/README.md` is 100% M2's; M1 does not touch it.** (Decided during apply.)
The careful-review pass found the gallery doc carries a *third* category of staleness beyond
the planned notebook-link fix and M2's image-link rewire: removed-symbol **code examples**
(`cp.riemann`, `cp.pair_plot_landscape`, `cp.plot_landscape`). Since M2 regenerates this file
wholesale from the preset registry (code, images, and prose), splitting its ownership — M1
fixes some lines, M2 rewrites them — produces throwaway work and a messy half-edited file.
Resolution: **one file, one owner.** M1 owns symbol-cleanliness + path links for the *general*
docs only (`README.md`, `docs/README.md`, `docs/pyvista_usage_guide.md`, `backend-policy.md`);
`docs/gallery/README.md` is left untouched and its staleness rides through the brief M1→M2
window as M2's explicit responsibility. The M1 spec was narrowed accordingly. *Alternative:*
M1 makes the gallery doc symbol-clean now — rejected as throwaway against M2's wholesale
rewrite. *Note:* `pyvista_usage_guide.md` and `backend-policy.md` legitimately *document* the
mpl-3D removal (with migration guidance); those mentions are correct and are kept, not "fixed."

## Risks / Trade-offs

- **[Moving notebooks breaks external/bookmarked links]** (e.g. blog posts, PyPI cached
  README) → Mitigation: M1 is pre-PyPI-3.0; the GitHub README updates atomically with the
  move, and there is no published 3.0 yet to have propagated old paths.
- **[Doc link sweep misses a reference]** → Mitigation: D4's guard plus a repo-wide grep for
  `examples/` in `*.md` as a task step; the explore already enumerated the reference sites.
- **[Deleting `archive/` loses a useful snippet someone wanted]** → Mitigation: git history
  retains every deleted file; nothing is truly lost, only removed from the working tree.
- **[Gallery image links temporarily reference soon-to-be-replaced PNGs]** → Accepted: the 27
  PNGs stay valid through M1; M2 replaces them and rewires the image links in one coherent step.
- **[Modulus-scaling demo coverage gap between M1 and M2]** → Accepted: M1 deletes the broken
  `modulus_scaling_showcase.py` and M2 reintroduces a PyVista equivalent. M1→M2 run back-to-back
  inside the same pre-3.0-release window, so no *published* release ships without the demo; M1's
  link sweep repoints the README reference to the modulus-scaling prose until M2 lands.

## Migration Plan

1. Create `examples/notebooks/` and `examples/scripts/`; `git mv` the 4 notebooks into
   `notebooks/` and the one clean script (`interactive_showcase.py`) into `scripts/`.
2. Delete `generate_gallery.py`, `gallery/generate_gallery_images.py`,
   `gallery/README_scripts.md`, `modulus_scaling_showcase.py` (broken mpl-3D), `archive/`,
   `old/`, `README_interactive_demo.md`.
3. Rewrite `examples/README.md` to the new layout (absorbing the interactive-demo notes).
4. Sweep the **general** docs (`README.md`, `docs/README.md`, `docs/pyvista_usage_guide.md`,
   `docs/development/backend-policy.md`): fix notebook/script links to the new paths; drop the
   already-dead links; repoint the `modulus_scaling_showcase.py` reference to the
   modulus-scaling prose (M2 reintroduces it); and ensure no removed mpl-3D function is listed
   as a currently available API (correct "removed in 3.0 — use `*_pv`" notes are kept).
   `docs/gallery/README.md` is **not touched** here — it is M2's (see D6).
5. Add the structural guard test; run the suite.

Rollback: revert the commit; all moves are `git mv` and all deletions are recoverable from
history.

## Open Questions

- None blocking. (Whether `interactive_showcase.py` is eventually superseded by M2's
  `showcase.py` is an M2 question, not M1's — M1 keeps it and fixes nothing in it since it is
  already clean.)
