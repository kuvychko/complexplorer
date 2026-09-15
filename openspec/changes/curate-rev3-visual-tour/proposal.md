## Why

Complexplorer is a visualization library, so its images are the primary evidence that it works
and the fastest explanation of why it exists. Today they under-sell the release:

- The gallery presents ~45 full-size images in registry order, with no navigation and captions
  that repeat the preset title. A visitor cannot see the range in 30 seconds.
- Several headline 3.0 capabilities are **described but never shown**: the phase-wheel legend,
  engineering mode (`cp.ee`), composite domains, the Riemann sphere/surface distinction, and the
  STL-to-physical-object story.
- Nothing demonstrates interactivity; a static screenshot cannot show rotation or depth.
- The committed portraits have a defect: the `Im(z)` axis label is clipped at the left edge.
- Render settings were ad-hoc per call site, so images did not share a look.

Round R0 of the visual review settled the house style. This change produces the tour itself.

## What Changes

**Locked render settings**
- A `RENDER_PROFILES` table becomes the single source for camera, ground, lighting, antialiasing,
  window size, and orientation widget, replacing the ad-hoc constants.
- The R0 decision is the baseline for every family: the "Gallery grey" profile — a `#fbfcfd` →
  `#e6e9ee` vertical gradient, three-point lighting at specular `0.3`, SSAA, no orientation
  widget, camera zoom `1.22` (`0.94` for landscapes, so the domain corners stay in frame), at
  1560 × 1560 px. 2D portraits keep their axes and ticks, add the phase-wheel legend, and save
  with a tight bounding box.
- Mesh resolution per family is set by review round R0b and recorded with the profile.

**A curated tour layer**
- A reserved `_tour/` section holds the assets the registry cannot express, each one a
  declarative recipe: function, domain, colormap, scaling, profile, resolution, caption, alt
  text, and a runnable snippet.
- The initial set: a portrait with `legend=True`; the same function as a landscape (the 2D → 3D
  pair); an engineering figure (transfer portrait, pole-zero, Bode, Nyquist); the composition
  proof that one `TransferFunction` drives both `cp.ee` and a general renderer; a composite-domain
  portrait; Riemann sphere beside Riemann surface; the STL triptych (relief, mesh, printed
  object); and a short orbit loop for the docs.

**Physical output**
- Owner-supplied photographs of printed ornaments live in a `_curated/` section: recorded with
  provenance, exempt from regeneration, and never silently replaced by a render.

**Hero montage**
- A generated 2 × 3 montage covering domain coloring, landscape, sphere/relief, multivalued
  surface, engineering mode, and physical output, checked at GitHub README and PyPI widths in
  both light and dark. (Wiring it into the README belongs to `prepare-3-0-release-notes`.)

**A navigable gallery page**
- A thumbnail grid and table of contents at the top; sections ordered by idea rather than by
  registry id; one- or two-sentence interpretations; alt text that names the render type;
  snippets collapsed so they do not drown the images; and a header stating which file to edit
  and which command regenerates the page.

**Producer and review workflow**
- `--out DIR` and `--only SECTION` let a round render into staging without touching the
  committed gallery, which is what feeds the private review page.
- Assets reach the repository only by re-running the documented command, never by copying a
  staged file.

**Defect fix**
- `cp.gallery` saves portraits with a tight bounding box, so axis labels are no longer clipped.

## Capabilities

### New Capabilities

_None._

### Modified Capabilities

- `examples`:
  - Renders use one locked profile and resolution per family.
  - A curated tour section demonstrates the capabilities the registry cannot express, each asset
    carrying a recipe, caption, and alt text.
  - Owner-supplied physical-output assets are recorded and exempt from regeneration.
  - A generated hero montage summarises the release.
  - The gallery page is navigable: thumbnails, idea-ordered sections, interpretations, and
    regeneration instructions.
  - The producer supports staging output so a review round never touches committed assets.
- `gallery`: portrait images are written without clipping their axis labels.

## Impact

- **Code:** `examples/showcase.py` (profiles, tour recipes, hero montage, thumbnails, staging
  flags, page generation); `complexplorer/gallery.py` (tight bounding box).
- **Data:** `examples/gallery/showcase.json` gains `tour` and `curated` sections and a schema
  bump; new `_tour/`, `_curated/`, and thumbnail assets; existing renders are re-rendered under
  the locked profile.
- **Docs:** `docs/gallery/gallery.generated.md` is restructured; `docs/gallery/README.md` frames it.
- **Tests:** `tests/unit/test_showcase_bundle.py` covers the tour, curated, hero, and thumbnail
  invariants and checks that every snippet parses and references only public API.
- **Dependencies:** none added. The orbit loop uses PyVista's own GIF writer.
- **Not here:** the README rewrite and claim softening (`prepare-3-0-release-notes`), and the
  docs-site build (`publish-rev3-docs-site`).
