## 1. Locked render settings

- [x] 1.1 Add a `RENDER_PROFILES` table to `examples/showcase.py`: one entry per family
  (`portrait`, `landscape`, `sphere`, `relief`, `surface`, `mesh`) carrying ground, lighting,
  specular, antialiasing, orientation widget, camera zoom, window size and mesh resolution.
  - Seed it with the R0 decision (Gallery grey) and the R0b resolutions.
  - Replace `WINDOW_3D`, `RESOLUTION_3D` and `SURFACE_RESOLUTION`.
- [x] 1.2 Route every render through the profile: take the plotter with `return_plotter=True`,
  apply ground/lighting/specular/antialiasing/zoom, then screenshot. 2D portraits get the
  phase-wheel legend and a tight bounding box.
- [x] 1.3 Record the profile name and mesh resolution on every `showcase.json` render record.

## 2. Producer plumbing

- [x] 2.1 Add `--out DIR` (redirect every write, including the generated page) and extend
  `--only` to `{presets, colormaps, tour, hero, thumbs, all}`. Sections that are not re-rendered
  keep their existing manifest records.
- [x] 2.2 Bump `showcase.json` to schema 2: add the `tour` and `curated` sections and fold the
  existing `banner` entry into `curated`.
- [x] 2.3 Generate a ~400 px PNG thumbnail for every render, under `thumb/`.

## 3. Library defect fix

- [x] 3.1 Save portraits from `complexplorer/gallery.py` with a tight bounding box, and add a test
  that the saved figure keeps its axis labels (no clipped `Im(z)`), then regenerate the committed
  portraits.

## 4. Tour recipes — review round R1

- [x] 4.1 Add the tour recipe record (id, title, section, inputs, profile, caption, alt text,
  snippet) and the dispatch that renders one.
- [x] 4.2 Legend portrait and the matching analytic landscape (the 2D → 3D pair).
- [x] 4.3 Engineering figure: `transfer_portrait` with poles/zeros/stability boundary, plus
  `pole_zero_plot`, `bode_plot` and `nyquist_plot` for one nontrivial stable transfer function.
- [x] 4.4 Composition proof: the same `TransferFunction` through `cp.ee` and through
  `plot_landscape_pv`.
- [x] 4.5 Composite-domain portrait whose geometry visibly depends on a union, intersection or
  difference.
- [x] 4.6 Riemann sphere beside Riemann surface, framed so the sheet structure is legible.
- [x] 4.7 Physical-output triptych: relief render, STL mesh render (clay shading), and the photo
  slot.
- [x] 4.8 Orbit loop GIF of one surface or relief, within the ~3 MB budget.
- [x] 4.9 Render R1 into staging, publish it to the review page, and iterate until every tour asset
  is approved.

## 5. Curated physical-output assets

- [x] 5.1 Add the `_curated/` section: ingest the owner's photographs (crop and size them to match
  the render panels), record kind and origin in `showcase.json`, and keep them out of every
  regeneration path. Absent photographs must not break the build.

## 6. Hero montage — review round R2

- [x] 6.1 Generate the 2 × 3 montage from already-rendered assets, with consistent crops, labels
  and dimensions, and the mesh-render fallback for the physical-output panel.
- [x] 6.2 Preview the montage at GitHub README and PyPI widths in light and dark, publish R2, and
  iterate to approval.

## 7. The gallery page

- [x] 7.1 Rewrite the generated page: thumbnail grid / table of contents, idea-ordered sections,
  per-figure interpretation, alt text naming the render type, snippets inside `<details>`, and a
  header naming the producer and its command.
- [x] 7.2 Update `docs/gallery/README.md` to frame the new page.

## 8. Full regeneration — review round R3 — and tests

- [x] 8.1 Regenerate every asset under the locked profiles, review the diff, publish R3, and
  iterate to approval.
- [x] 8.2 Extend `tests/unit/test_showcase_bundle.py`:
  - the tour covers the required capability set and every file exists
  - every tour entry has a recipe, caption and alt text
  - every render has a thumbnail
  - curated and generated assets are disjoint, and no curated entry claims a recipe
  - the hero montage is recorded with its source assets
  - every snippet parses and references only exported names
  - the orbit loop stays within its size budget
- [x] 8.3 Round R4: checked on the pushed branch — the hero, section figures and collapsed
  snippets render on GitHub and the table-of-contents anchors resolve. The README's own rendering
  is checked by `prepare-3-0-release-notes`, which rewrites it.

## 9. Verification and bookkeeping

- [x] 9.1 Run the gate: `pytest`, `ruff check`/`format`, `openspec validate --specs`,
  `openspec validate curate-rev3-visual-tour`, a clean `python examples/showcase.py` with the diff
  reviewed, and `index.json` byte-stability.
- [x] 9.2 Record the R0b outcome and the final profile table in `openspec/REV3_CLOSEOUT.md`, and
  flip C2's status in `openspec/ROADMAP.md`.
