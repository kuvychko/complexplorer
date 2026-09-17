## Context

`examples/showcase.py` renders the curated registry (`cp.catalog`) into the committed gallery:
every preset gets a 2D portrait from the deterministic library generator, and tags decide which
PyVista screenshots follow (`canonical` → landscape + sphere, `branches` → surface, `ornament` →
relief). That structure works and stays.

What it cannot express is everything that is not a catalog preset: the phase-wheel legend, an
engineering figure, a composite domain, a sphere/surface comparison, a photograph. Those are the
rev3 capabilities the closeout says are claimed but never shown.

Render settings were previously three module constants (`WINDOW_3D`, `RESOLUTION_3D`,
`SURFACE_RESOLUTION`) passed per call, with background, lighting and camera left at each
renderer's hard-coded defaults (white ground, PyVista light kit, orientation widget on).

**Review round R0 (decided 2026-09-15)** settled the house style: profile B, "Gallery grey", for
all five families, recorded in `openspec/REV3_CLOSEOUT.md`. **Round R0b** sets the mesh resolution
per family; the owner's R0 note was that the sphere is blocky and that PyVista's visual quality is
the main reason the gallery exists.

Defects found during R0, both owned here:

- committed portraits clip the `Im(z)` label and carry a wide top margin — `cp.gallery` saves
  without a tight bounding box;
- brightness bands stair-step where they cross sphere mesh rows, which is a resolution problem
  (R0b), not a style one.

## Goals / Non-Goals

**Goals:**
- One locked profile and resolution per render family, applied everywhere.
- Every rev3 headline capability has a committed visual with a runnable recipe.
- The gallery page can be scanned in about 30 seconds.
- Every asset is either regenerable from one documented command or explicitly recorded as curated.
- Each review round renders into staging, never into the committed tree.

**Non-Goals:**
- The README rewrite, hero wiring and claim softening (`prepare-3-0-release-notes`).
- The docs-site build and nav (`publish-rev3-docs-site`).
- Linting the `examples/` tree and adding CI coverage for it (`gate-release-artifacts-and-ci`).
- New catalog presets. The tour composes what the library already exposes.

## Decisions

### D1. Profiles and resolutions are data, not call-site arguments

`RENDER_PROFILES` maps a family (`portrait`, `landscape`, `sphere`, `relief`, `surface`, `mesh`)
to ground, lighting, specular, antialiasing, orientation widget, camera zoom, window size and mesh
resolution. Both the registry renders and the tour recipes read it, so a future change re-styles
the gallery in one place and the committed images cannot drift apart.

The PyVista entry points expose camera, window and orientation but not ground or lighting, so the
producer takes the plotter with `return_plotter=True` and applies the profile before the
screenshot. This keeps the library's defaults untouched: styling is a gallery concern, not a
library one.

- *Alternative:* add background/lighting parameters to the renderers. Rejected here — it widens
  the public API for a presentation need, during a feature freeze.

### D2. Tour assets are declarative recipes

Each tour asset is a record: id, title, section, the callable or preset it renders, domain,
colormap, scaling, family profile, caption, alt text, and the snippet that reproduces it. A
dispatch table renders from the record, and the same record populates `showcase.json`, the gallery
page and the review page. The closeout's "record the function, domain, colormap, scaling,
resolution and camera for each curated asset" is then satisfied by construction rather than by a
separate note.

### D3. `showcase.json` gains `tour` and `curated`; its schema version goes 1 to 2

The presentation manifest is versioned independently of the library's `index.json` (now at 3).
`banner` folds into `curated` as one entry, so there is a single rule for "committed but not
regenerated".

### D4. Photographs are curated assets with provenance

Owner-supplied photographs live under `_curated/` and are recorded with their source and a note
that they are not regenerated. A photograph can never be mistaken for a render: the manifest marks
its kind, and the tests assert that no curated entry claims a recipe.

### D5. The hero montage is generated, not hand-assembled

A matplotlib composition lays the six chosen panels out 2 × 3 at a fixed size with consistent
crops and labels, reading its inputs from the already-rendered assets. It therefore reproduces
from the same command, and a panel swap is a one-line change. The photo panel degrades to the STL
mesh render when no photograph is available yet, so the montage never blocks on it.

### D6. Thumbnails are PNG

The docs gallery links full-resolution images from ~400 px thumbnails. They are PNG rather than
WebP because the same assets are referenced from README/PyPI contexts where WebP support is not
guaranteed. The size cost is acceptable at thumbnail dimensions.

### D7. The page is ordered by idea

Sections: phase portraits → mapping and topology → Riemann surfaces → engineering → colormaps →
physical output. Each entry carries its interpretation; the snippet sits in a `<details>` block so
code does not push the images apart. The page header names the producer and the command.

### D8. The portrait bounding-box fix lands in the library

`cp.gallery` saves with a tight bounding box. This changes portrait pixels but not the manifest,
which the `gallery` capability already declares best-effort, and it fixes a real defect: the
`Im(z)` label is clipped in every committed portrait today.

### D9. Staging flags, and promotion by regeneration

`--out DIR` redirects every write; `--only SECTION` restricts the work (`presets`, `colormaps`,
`tour`, `hero`, `thumbs`). Review rounds render into the scratch area and publish to the review
page; approved assets reach the repository only by running the producer against the committed
tree, so "every committed asset reproduces from a documented command" stays true.

### D10. The orbit loop is a docs asset with a size budget

PyVista's `open_gif` plus an orbital path produces a short rotating loop of one surface or relief.
It is capped at roughly 3 MB and referenced from the docs gallery. The README stays static unless
`prepare-3-0-release-notes` decides otherwise.

## Risks / Trade-offs

- **Re-rendering every asset under the new profile churns ~45 PNGs in one commit.** → Expected,
  and reviewed as a batch in round R3; the manifests stay byte-stable and the diff is inspected
  before commit.
- **The photographs may arrive late.** → D5's fallback keeps the hero and the tour renderable;
  only the physical-output panels wait.
- **Snippets can rot.** → A test parses every snippet and checks that each referenced name exists
  in the public API; execution stays out of CI because off-screen VTK is unreliable there.
- **A locked zoom can clip a silhouette that differs from the calibration function.** → Each tour
  recipe names its family profile and may override the zoom, as the landscape already does.
- **GIF weight.** → Budgeted in D10 and asserted by a test.

## Migration Plan

No runtime behavior changes beyond the portrait bounding box. Assets are replaced in place; the
deterministic `index.json` contract is untouched. Rollback is a revert of the asset commit.

## Open Questions

- **Mesh resolution per family is pending round R0b.** Recommended, pending the owner's picks:
  sphere 1000, relief 800, landscape 600, surface 312 — measured at roughly a second per render
  and within 40 KB of each other, so the choice is about quality, not cost. The profile table
  lands with these values and is corrected if R0b decides otherwise.
