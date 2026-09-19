## Why

M1 left `examples/gallery/` holding 25 hand-named PNGs (e.g. `Riemann_relief_map_20250726.png`)
produced by the now-retired hand-rolled generators, and `docs/gallery/README.md` still
hand-links them with hand-written code snippets that reference removed mpl-3D functions. The
library already has the canonical, registry-driven `cp.gallery` (deterministic 2D portraits +
`index.json`), but it is PyVista-free by design, so the *visual* gallery the README/PyPI page
shows — 3D landscapes, Riemann spheres, the new Riemann **surfaces** — has no producer. This is
M2 of the `migrate-examples-and-docs` arc: build the presentation layer that turns the preset
registry into the high-res visual gallery, and regenerate everything from one command.

## What Changes

- **New `examples/showcase.py`** — the single high-res producer (Option B from the roadmap). It
  wraps the deterministic `cp.gallery` for 2D portraits + `index.json`, then adds the PyVista
  screenshots the manifest deliberately omits, and emits the docs gallery page. Run locally
  (never in CI — off-screen VTK screenshots crash only on headless CI; locally on the user's
  machine they render fine).
- **Tag-driven render set.** The render type follows from each preset's tags (which already
  encode mathematical character), not a hand-list:
  - all presets → `portrait.png` (2D, from `cp.gallery`)
  - `canonical` → `+ landscape.png` (`plot_landscape_pv`) `+ sphere.png` (`riemann_pv`)
  - `branches` (sqrt/cbrt/log) → `+ surface.png` (`riemann_surface_pv`: power n=2, power n=3, log)
  - `ornament` → `+ ornament.png` (a relief render)
- **Per-id image naming** `<id>/<render-type>.png`, extending `cp.gallery`'s `<id>/portrait.png`
  — the showcase bundle is a strict superset of the deterministic bundle.
- **Colormap gallery section.** Because every preset uses `Phase`, a registry-only gallery would
  hide the colormap variety. The showcase additionally renders one designated reference function
  (a canonical preset) across the library's actual colormaps — `Phase` and its enhancement
  variants, `Chessboard`, `PolarChessboard` (linear/log), `LogRings` — into a reserved
  `examples/gallery/_colormaps/<name>.png`, with its own `colormaps` section in `showcase.json`
  and the docs gallery. (Scope note: the "perceptual family" advertised in CLAUDE.md/README —
  `OklabPhase`, `CubehelixPhase`, etc. — is **not implemented**; only the four colormaps above
  exist. Reconciling that documentation is out of M2's scope — see Impact.)
- **Split manifests.** `index.json` (byte-deterministic math/Godot interchange) is **untouched**;
  the showcase writes a separate `showcase.json` (presentation manifest: catalog data + every
  render path). The non-deterministic screenshot layer never contaminates the deterministic one.
- **Generated docs gallery (middle path).** `showcase.py` generates
  `docs/gallery/gallery.generated.md` (image + derived code snippet + title/story per preset,
  from `showcase.json`); a thin, hand-written `docs/gallery/README.md` frames and includes/links
  it. Code snippets are *derived* from `expression`/`domain_spec`/`cmap_spec`, so they cannot
  drift and never reference a removed symbol.
- **Retire the hand-named PNGs and rewire the image links** M1 deferred: delete 24 of the 25
  legacy PNGs and repoint `README.md`'s gallery thumbnails (3 refs) to id-based renders; rewrite
  `docs/gallery/README.md` wholesale (its removed-symbol code examples + dead links die here).
  **Exception (per request):** keep the hand-crafted hero composite
  `examples/gallery/Riemann_relief_map_20250726.png` — it is a curated banner asset (the README
  header), not a registry render, and is documented as such (recorded in `showcase.json` under a
  `banner` key, exempt from the "id-addressed + regenerable" invariant).
- **STL is NOT committed.** `.stl` binaries bloat the repo; the gallery shows a relief PNG and
  links the generation code. (A dedicated 3D-printing site is the future home for downloadable
  STLs.)
- **Rebuild the modulus-scaling demo** M1 deleted, as a PyVista demo under `examples/scripts/`
  (the multi-mode composition `showcase.py`'s screenshot machinery now makes cheap).

## Capabilities

### Modified Capabilities
- `examples`: extend with the showcase producer contract — a registry-driven high-res gallery
  whose images are committed-but-regenerable from one command, a presentation manifest separate
  from the deterministic `index.json`, and a generated docs gallery page derived from the
  registry (no hand-written snippets, no removed-symbol references).

## Impact

- **New:** `examples/showcase.py`; `examples/gallery/showcase.json`; the regenerated
  `examples/gallery/<id>/*.png`; `docs/gallery/gallery.generated.md`; a rebuilt PyVista
  `examples/scripts/modulus_scaling_showcase.py`.
- **Edited:** `README.md` (gallery image links → id-based); `docs/gallery/README.md` (thin hand
  frame including the generated page); the `examples` capability spec.
- **Deleted:** 24 of the 25 hand-named `examples/gallery/*.png` (the curated hero composite
  `Riemann_relief_map_20250726.png` is kept as the README banner).
- **Untouched:** `complexplorer/gallery.py` and its `index.json` contract; the catalog; all
  library rendering APIs. The library stays PyVista-free where it already was; all PyVista use
  lives in `examples/`.
- **Not in CI.** `showcase.py` is a local regeneration tool (off-screen screenshots crash on
  headless CI). A guard test MAY assert the committed `showcase.json` is consistent with the
  catalog, but it does not render.
- **Flagged, not fixed here:** CLAUDE.md and `README.md` advertise a perceptual colormap family
  (`OklabPhase`, `PerceptualPastel`, `AnalogousWedge`, `DivergingWarmCool`, `Isoluminant`,
  `CubehelixPhase`, `InkPaper`, `EarthTopographic`, `FourQuadrant`) that does not exist in the
  code (only `Phase`/`Chessboard`/`PolarChessboard`/`LogRings` are implemented). M2 renders only
  the real colormaps; whether to *implement* the perceptual family or *correct the docs* is a
  separate decision (a future change or a doc fix), tracked outside this proposal.
