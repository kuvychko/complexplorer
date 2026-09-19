## ADDED Requirements

### Requirement: A registry-driven showcase renders the high-res visual gallery

The repository SHALL provide a single showcase producer (`examples/showcase.py`) that renders
the curated preset registry (`cp.catalog`) into the high-resolution visual gallery, including
the PyVista 3D renders the library `cp.gallery` deliberately omits. The set of renders per
preset SHALL be determined by the preset's tags (which encode mathematical character), not a
hand-maintained per-preset list:

- every preset receives a 2D `portrait.png` (produced via the library `cp.gallery`);
- `canonical` presets additionally receive a 3D `landscape.png` and a Riemann-sphere `sphere.png`;
- `branches` presets additionally receive a Riemann-`surface.png` (via `riemann_surface_pv`);
- `ornament` presets additionally receive a relief `ornament.png`.

The producer is a local regeneration tool and is NOT run in CI (off-screen VTK screenshots
crash only on headless CI).

#### Scenario: The render set follows the tag policy

- **WHEN** the showcase renders a preset
- **THEN** a `portrait.png` is always produced, and a `landscape.png` + `sphere.png`,
  `surface.png`, or `ornament.png` is produced exactly when the preset carries the
  corresponding tag (`canonical`, `branches`, `ornament`)

#### Scenario: Multivalued presets map to Riemann-surface families

- **WHEN** a `branches` preset is rendered (`sqrt`, `cbrt`, `log`)
- **THEN** its `surface.png` is produced by `riemann_surface_pv` with the matching family
  (`power` n=2, `power` n=3, and `log` respectively)

### Requirement: The showcase bundle extends the deterministic bundle with split manifests

Showcase images SHALL be written under a per-id directory as `<id>/<render-type>.png`, extending
the library bundle's `<id>/portrait.png` convention so the showcase bundle is a strict superset
of the deterministic bundle. The deterministic `index.json` written by `cp.gallery` SHALL remain
the byte-stable math/interchange contract and SHALL NOT be modified by the showcase. The showcase
SHALL write a separate presentation manifest (`showcase.json`) carrying the catalog data plus
every render's relative path, so the non-deterministic screenshot layer never contaminates the
deterministic manifest.

#### Scenario: index.json is preserved; showcase.json is the presentation manifest

- **WHEN** the showcase runs
- **THEN** `index.json` is exactly what `cp.gallery` produces for the same selection (unchanged),
  and a distinct `showcase.json` lists every preset's render paths (`portrait`, and any
  `landscape`/`sphere`/`surface`/`ornament`) as relative paths under the bundle root

#### Scenario: showcase.json is consistent with the catalog and the tag policy

- **WHEN** `showcase.json` is read
- **THEN** it references exactly the presets in the catalog, and each preset's listed render
  paths match the tag policy (no `surface` for a non-`branches` preset, etc.)

### Requirement: The visual gallery is committed but regenerable from one command

The registry-driven gallery renders SHALL be committed to the repository (so GitHub and PyPI
render them) and SHALL be reproducible by running the showcase producer once
(`python examples/showcase.py`). Every committed *render* SHALL be id-addressed
(`<id>/<render-type>.png`) and correspond to a preset id and render type the producer emits. A
small number of explicitly-curated banner/hero assets MAY be hand-maintained and exempt from the
regenerable invariant, provided each is recorded in `showcase.json` (e.g. under a `banner` key)
and documented as curated; no OTHER hand-named legacy image SHALL remain. STL meshes SHALL NOT be
committed — the gallery shows a relief render and links the generation code instead.

#### Scenario: Every render is reproducible and id-addressed; only documented heroes are exempt

- **WHEN** the committed `examples/gallery/` tree is inspected
- **THEN** every render lives at `<id>/<render-type>.png` for a catalog preset id; the only
  non-id-addressed image is the curated hero composite recorded under `showcase.json`'s `banner`
  key; and no other legacy hand-named PNG (e.g. `Phase_portrait_2d.png`) remains

#### Scenario: No STL binaries are committed

- **WHEN** the gallery tree is inspected
- **THEN** it contains no `.stl` files; ornament presets are represented by a relief PNG plus a
  link to the generation code

### Requirement: The gallery showcases the colormap family

Because the curated presets all use `Phase`, the showcase SHALL additionally render a colormap
gallery: one designated reference function (a catalog preset) rendered under each of the library's
implemented colormaps — `Phase` and its enhancement parameter variants, `Chessboard`,
`PolarChessboard` (linear and log spacing), and `LogRings`. These renders SHALL be written under a
reserved `_colormaps/` directory (`_colormaps/<name>.png`), recorded in `showcase.json` under a
`colormaps` section (with the reference preset id), and presented as a colormap section in the
generated docs gallery. The colormap gallery SHALL cover only colormaps that exist in the public
API; it SHALL NOT reference unimplemented colormaps.

#### Scenario: A colormap gallery is produced for the reference function

- **WHEN** the showcase runs
- **THEN** the designated reference function is rendered under each implemented colormap, each
  written to `_colormaps/<name>.png` and listed under `showcase.json`'s `colormaps` section with
  the reference preset id

#### Scenario: Colormap-section snippets reconstruct the colormap explicitly

- **WHEN** a colormap-section entry's snippet is generated
- **THEN** it uses the registry reference function (`cp.catalog.get(<id>)`) and constructs the
  colormap explicitly (e.g. `cp.Chessboard(spacing=0.25)`), runs as shown, and references only
  3.0-surface APIs

### Requirement: The docs gallery is generated from the registry

The documentation gallery SHALL be generated from the registry rather than hand-authored: the
showcase producer SHALL emit a generated gallery page (`docs/gallery/gallery.generated.md`) whose
per-preset entries carry a registry-driven code snippet (built against `cp.catalog.get(<id>)` and
matched to the render type, with the `expression` shown as a comment), a description from
`title`/`story`, and images drawn from `showcase.json`. A thin hand-written
`docs/gallery/README.md` SHALL frame and include or link the generated page. Generated snippets
SHALL be runnable as shown and SHALL NOT reference any symbol removed at 3.0.

#### Scenario: Snippets are registry-driven and runnable

- **WHEN** the generated gallery page is produced
- **THEN** each entry's snippet reconstructs the visualization via the registry
  (`cp.catalog.get(<id>)` plus `.func`/`.domain()`/`.colormap()`, or the matching
  `riemann_surface_pv` family call), runs as written, and references only 3.0-surface APIs (no
  `plot_landscape`, `pair_plot_landscape`, or 3D `riemann`)

#### Scenario: The gallery doc no longer hand-links legacy images

- **WHEN** `docs/gallery/README.md` and the generated page are read
- **THEN** they reference id-based renders from `showcase.json` (no link to a hand-named legacy
  PNG or a non-existent `examples/*.ipynb`)
