# Examples

## Purpose

The examples capability defines how the repository's `examples/` tree is organized and kept
honest: it derives its catalog content from the curated function preset registry rather than a
parallel hand-rolled generator, follows a documented directory layout (`notebooks/`, `scripts/`,
`gallery/`), renders under one locked profile per family, and references no symbol removed at 3.0
nor any missing file. It also owns the curated visual tour: the assets that demonstrate
capabilities the registry cannot express, the hero montage, and the navigable gallery page. It is the contract that
keeps the user-facing demos and general documentation consistent with the live API.

## Requirements

### Requirement: Examples derive from the curated registry

The `examples/` tree SHALL treat the function preset registry (`cp.catalog`) and the library
gallery generator (`cp.gallery` / the `gallery` CLI subcommand) as the single source of truth
for catalog content. The repository SHALL NOT contain a second, hand-rolled gallery-generation
script that maintains its own parallel list of functions or its own image-naming scheme.

#### Scenario: No parallel hand-rolled generator exists

- **WHEN** the `examples/` tree is inspected
- **THEN** it contains no standalone gallery-generation script that enumerates its own
  function catalog (the retired `examples/generate_gallery.py` and
  `examples/gallery/generate_gallery_images.py` are absent), and any gallery content is
  produced from `cp.catalog` via the library generator

#### Scenario: The library gallery generator is untouched

- **WHEN** the library is imported after this change
- **THEN** `complexplorer.generate_gallery` and the `gallery` CLI subcommand behave exactly as
  before (this change retires only the duplicate *example* scripts, not the library capability)

### Requirement: Examples follow a defined directory layout

The `examples/` directory SHALL be organized into a documented layout: `notebooks/` for the
Jupyter tutorials, `scripts/` for the curated runnable Python demos, and `gallery/` for
generated gallery output. `examples/README.md` SHALL describe this layout and point readers at
the correct entry points. Obsolete reference material (the former `archive/` and `old/`
subdirectories) SHALL NOT be carried in the tree.

#### Scenario: Tutorials and scripts live in their dedicated subdirectories

- **WHEN** the `examples/` tree is inspected
- **THEN** the tutorial notebooks reside under `examples/notebooks/`, the runnable demo
  scripts reside under `examples/scripts/`, and no `examples/archive/` or `examples/old/`
  directory remains

#### Scenario: The examples README maps the layout

- **WHEN** `examples/README.md` is read
- **THEN** it describes the `notebooks/`, `scripts/`, and `gallery/` directories and references
  only files that exist in the tree

### Requirement: Examples and docs reference no removed-at-3.0 symbol or missing file

Example scripts SHALL NOT reference any symbol removed at 3.0 — the matplotlib 3D functions
(`plot_landscape`, `pair_plot_landscape`, the 3D `riemann`) and the deleted capability flags
(`HAS_PYVISTA`, `HAS_STL_EXPORT`). The general in-repository documentation SHALL NOT present a
removed symbol as a currently available API (legitimately *documenting* that a symbol was
removed — with migration guidance — is allowed), and SHALL NOT link to files that do not exist.

The gallery documentation (`docs/gallery/README.md`) is **out of scope** for this requirement:
it is regenerated wholesale from the preset registry by the follow-up gallery change (M2),
which owns its code examples, image links, and prose. M1 does not touch it.

#### Scenario: No example script imports or calls a removed symbol

- **WHEN** the Python files under `examples/` are scanned
- **THEN** none of them call `cp.plot_landscape`, `cp.pair_plot_landscape`, the 3D `cp.riemann`,
  or reference `HAS_PYVISTA` / `HAS_STL_EXPORT`; 3D demos use the PyVista `*_pv` functions

#### Scenario: General docs do not present removed symbols as available

- **WHEN** the general docs M1 owns (the top-level `README.md`, `docs/README.md`,
  `docs/pyvista_usage_guide.md`, `docs/development/backend-policy.md`) are read
- **THEN** none of them list a removed matplotlib-3D function as a currently available API; any
  mention is framed as "removed in 3.0 — use the `*_pv` equivalent"

#### Scenario: Documentation links resolve to existing files

- **WHEN** the notebook/script links in the general docs M1 owns (`README.md`, `docs/README.md`,
  `docs/pyvista_usage_guide.md`) are followed
- **THEN** every referenced notebook and script path resolves to a file that exists under the
  new layout (no link to a former `archive/`-only file such as `interactive_demo.py`)

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

### Requirement: Renders use one locked profile per family

The showcase producer SHALL define one render profile per family (2D portrait, analytic landscape,
Riemann sphere, relief, Riemann surface, STL mesh) and SHALL apply it to every render of that
family, covering ground, lighting, antialiasing, orientation widget, camera framing, window size
and mesh resolution. Profiles SHALL be declared in one place rather than passed ad-hoc at each
call site, so the committed gallery shares a single look.

#### Scenario: Every render of a family uses that family's profile

- **WHEN** the producer renders any asset, whether registry-driven or curated
- **THEN** it takes the ground, lighting, antialiasing, widget, framing, window size and mesh resolution from that family's profile entry

#### Scenario: The profile is recorded with the asset

- **WHEN** a render is recorded in `showcase.json`
- **THEN** the record names the profile and the mesh resolution used, so the settings behind any committed image are recoverable

### Requirement: A curated tour demonstrates the capabilities the registry cannot express

Beyond the tag-driven preset renders, the producer SHALL render a curated tour section covering
the capabilities that are not catalog presets. The tour SHALL include at least:

- a 2D portrait with the phase-wheel legend enabled
- the same function as an analytic landscape, so the 2D to 3D relationship is visible
- an engineering-mode figure (transfer portrait with poles, zeros and the stability boundary, plus
  the Bode and Nyquist views)
- a composition proof: one `TransferFunction` rendered by `cp.ee` and by a general renderer
- a portrait whose geometry depends on a composite domain (union, intersection or difference)
- a Riemann sphere and a Riemann surface presented side by side
- the physical-output story: a relief render, its STL mesh, and a printed object

Each tour asset SHALL be recorded with the inputs that produced it (function, domain, colormap,
scaling, profile and resolution), a one- or two-sentence interpretation, alt text naming the render
type, and a runnable snippet.

#### Scenario: The tour covers the rev3 capability set

- **WHEN** the tour section of `showcase.json` is read
- **THEN** it contains an entry for each capability listed above, and every referenced image file exists

#### Scenario: Each tour entry carries its recipe and prose

- **WHEN** a tour entry is inspected
- **THEN** it records the inputs that produced the image, a caption, alt text that distinguishes the render type, and a snippet

#### Scenario: Snippets reference only the public API

- **WHEN** the snippets in the tour and colormap sections are parsed
- **THEN** each one parses as Python, and every `complexplorer` name it references is exported by the package (or by `cp.ee`)

### Requirement: Curated assets are recorded, never regenerated

Assets that cannot be produced by running the producer — photographs of printed objects, and any
hand-made image — SHALL be recorded in `showcase.json` as curated, with their kind and origin, and
SHALL be exempt from the regenerable invariant. A curated entry SHALL NOT claim a render recipe,
and the producer SHALL NOT overwrite one.

#### Scenario: A photograph is recorded as curated

- **WHEN** a photograph of a printed ornament is added to the gallery
- **THEN** it is listed in the curated section with its kind and origin, and no code path regenerates it

#### Scenario: Curated and generated assets are distinguishable

- **WHEN** the manifest is read
- **THEN** every image is either a render with a recipe or a curated asset with an origin, and none is both

### Requirement: A hero montage summarises the release

The producer SHALL generate a hero montage laying out six panels — domain coloring, analytic
landscape, Riemann sphere or relief, multivalued surface, engineering mode and physical output —
with consistent crops, labels and dimensions, at a size that stays legible at typical README width.
The montage SHALL be generated from already-rendered assets, so it reproduces from the same
command. When no photograph is available, the physical-output panel SHALL fall back to the STL mesh
render rather than failing.

#### Scenario: The montage is generated and recorded

- **WHEN** the producer runs
- **THEN** the montage image is written, recorded in `showcase.json`, and lists the assets it composed

#### Scenario: The montage degrades without a photograph

- **WHEN** no curated photograph is present
- **THEN** the montage still renders, using the mesh render for the physical-output panel

### Requirement: The gallery page is navigable

The generated gallery page SHALL open with a thumbnail grid or table of contents linking to its
sections, SHALL group entries by idea (phase portraits; mapping and topology; Riemann surfaces;
engineering; colormaps; physical output) rather than by registry order, and SHALL present each
figure with an interpretation and alt text that names the render type. Thumbnails SHALL link to the
full-resolution image. Code snippets SHALL be present but visually secondary. The page SHALL state
which file to edit and which command regenerates it.

#### Scenario: The page opens with navigation

- **WHEN** the generated gallery page is read
- **THEN** it begins with a thumbnail grid or table of contents linking to each section, followed by the idea-ordered sections

#### Scenario: Figures carry interpretation and useful alt text

- **WHEN** any figure on the page is inspected
- **THEN** it has a caption explaining what structure is visible, and alt text that names the render type rather than repeating the title alone

#### Scenario: The page says how it was made

- **WHEN** a reader wants to change the page
- **THEN** the page names the producer file and the command that regenerates it

### Requirement: The producer can render into a staging directory

The producer SHALL accept an output directory and a section selector, so a render can be produced
for review without touching the committed gallery. Committed assets SHALL be updated only by
running the producer against the committed tree.

#### Scenario: Staging output leaves the committed gallery untouched

- **WHEN** the producer is run with an output directory outside the repository gallery
- **THEN** every image, manifest and page it writes goes to that directory, and the committed gallery is unchanged

#### Scenario: A single section can be regenerated

- **WHEN** the producer is run with a section selector
- **THEN** only that section is re-rendered, and the other sections' committed records are preserved

### Requirement: The visual gallery is committed but regenerable from one command

The registry-driven gallery renders SHALL be committed to the repository (so GitHub and PyPI render
them) and SHALL be reproducible by running the showcase producer once
(`python examples/showcase.py`). Every committed *render* SHALL be id-addressed
(`<id>/<render-type>.png`) or written under a reserved section the producer emits (`_colormaps/`,
`_tour/`, thumbnails, the hero montage). Curated assets (photographs and the hero banner) MAY be
hand-maintained and exempt from the regenerable invariant, provided each is recorded in
`showcase.json` as curated; no OTHER hand-named legacy image SHALL remain. STL meshes SHALL NOT be
committed — the gallery shows a relief render and links the generation code instead.

#### Scenario: Every render is reproducible; only recorded curated assets are exempt

- **WHEN** the committed `examples/gallery/` tree is inspected
- **THEN** every image is either a render the producer emits (id-addressed, or under a reserved
  section directory) or an entry in the manifest's curated section, and no other hand-named legacy
  image remains

#### Scenario: No STL binaries are committed

- **WHEN** the gallery tree is inspected
- **THEN** it contains no `.stl` files; ornament presets are represented by a relief PNG plus a
  link to the generation code


### Requirement: The gallery showcases the colormap family

Because the curated presets all use `Phase`, the showcase SHALL additionally render a colormap
gallery: one designated reference function (a catalog preset) rendered under each of the
library's public colormaps. The set comprises:

- `Phase` and its enhancement parameter variants
- `Chessboard`
- `PolarChessboard` (linear and log spacing)
- `LogRings`
- each perceptual phase-portrait colormap (`OklabPhase`, `PerceptualPastel`, `AnalogousWedge`,
  `DivergingWarmCool`, `Isoluminant`, `CubehelixPhase`, `InkPaper`, `EarthTopographic`,
  `FourQuadrant`)

These renders SHALL be written under a reserved `_colormaps/` directory (`_colormaps/<name>.png`),
recorded in `showcase.json` under a `colormaps` section (with the reference preset id), and
presented as a colormap section in the generated docs gallery. The colormap gallery SHALL cover
exactly the colormaps that exist in the public API. It SHALL NOT reference unimplemented
colormaps, and it SHALL NOT omit a public one.

#### Scenario: A colormap gallery is produced for the reference function

- **WHEN** the showcase runs
- **THEN** the designated reference function is rendered under each public colormap, each
  written to `_colormaps/<name>.png` and listed under `showcase.json`'s `colormaps` section with
  the reference preset id

#### Scenario: Every public colormap appears in the colormap gallery

- **WHEN** the `colormaps` section of `showcase.json` is compared with the colormap classes exported by `complexplorer`
- **THEN** every exported concrete colormap class is rendered at least once, and no entry names a class that is not exported

#### Scenario: Colormap-section snippets reconstruct the colormap explicitly

- **WHEN** a colormap-section entry's snippet is generated
- **THEN** it uses the registry reference function (`cp.catalog.get(<id>)`) and constructs the
  colormap explicitly (e.g. `cp.Chessboard(spacing=0.25)`, `cp.Phase(phase_sectors=6)`), runs as
  shown, and references only 3.0-surface APIs


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

### Requirement: Notebooks execute top-to-bottom on the 3.0 surface

Every tutorial notebook under `examples/notebooks/` SHALL execute top-to-bottom without error on
the 3.0 API surface. No notebook SHALL call a symbol removed at 3.0 (`plot_landscape`,
`pair_plot_landscape`, the 3D `riemann`, `HAS_PYVISTA`, `HAS_STL_EXPORT`); 3D cells SHALL use the
PyVista `*_pv` functions. Execution SHALL be reproducible via a documented command.

#### Scenario: Each notebook runs without a cell error

- **WHEN** `examples/notebooks/*.ipynb` are executed (via the verification harness)
- **THEN** every notebook runs to completion with no cell raising an error

#### Scenario: No notebook references a removed symbol

- **WHEN** the notebook sources are scanned
- **THEN** none call `cp.plot_landscape`, `cp.pair_plot_landscape`, the 3D `cp.riemann`, or
  reference `HAS_PYVISTA` / `HAS_STL_EXPORT`

### Requirement: Notebooks render via the static PyVista backend with committed output

Each notebook SHALL select the static PyVista backend (`pv.set_jupyter_backend('static')`) in a
setup cell so its `*_pv` calls render as embedded static images and execute headlessly. Notebooks
SHALL be committed with their executed output (the static images), so they render on GitHub /
nbviewer; the committed output SHALL be reproducible by re-executing the notebook. A note SHALL
point readers to the interactive / high-quality path (`notebook=False` or the terminal scripts).

#### Scenario: A setup cell selects the static backend

- **WHEN** a notebook is opened
- **THEN** an early cell sets `pv.set_jupyter_backend('static')` and a note explains how to switch
  to an interactive backend for higher-quality exploration

#### Scenario: Committed notebooks carry executed image output

- **WHEN** a committed notebook is viewed without running it
- **THEN** its 2D and 3D cells show embedded image output (no stale matplotlib-3D renders remain)

### Requirement: Notebooks cover the 3.0 feature surface

The tutorials SHALL cover the headline 3.0 additions:

- a Riemann-**surface** example (`riemann_surface_pv`)
- the preset registry / gallery workflow (`cp.catalog`, and a pointer to the gallery producer)

Colormap material SHALL cover the library's public colormaps, including the perceptual family,
and SHALL reference only colormaps that exist in the public API. At least one tutorial SHALL
address color-vision deficiency (CVD): which colormaps remain interpretable under common CVD
types, and how to choose one.

#### Scenario: Riemann surfaces and the registry are demonstrated

- **WHEN** the tutorial set is read
- **THEN** at least one notebook demonstrates `riemann_surface_pv` and at least one demonstrates
  the preset registry (`cp.catalog`) with a pointer to the gallery producer

#### Scenario: Color-vision guidance is present

- **WHEN** the tutorial set is read
- **THEN** at least one notebook compares colormaps under simulated color-vision deficiency and recommends colormaps that remain interpretable

#### Scenario: Colormap material matches the public API

- **WHEN** the notebook sources are scanned for colormap class names
- **THEN** every referenced colormap is exported by `complexplorer`, and phase-sector counts are passed as `phase_sectors` (never `n_phi`)


### Requirement: A notebook execution harness verifies the tutorials

The project SHALL provide a documented, repeatable way to verify notebook execution using
`nbmake` (`pytest --nbmake examples/notebooks/`). The notebook tooling (`nbmake`, `nbconvert`,
`ipykernel`) SHALL be declared as installable dependencies (an `[examples]` extra). The harness
SHALL be opt-in — it SHALL NOT be collected by the default `pytest` run and SHALL NOT be required
in CI.

#### Scenario: The harness verifies all notebooks on demand

- **WHEN** `pytest --nbmake examples/notebooks/` is run in an environment with the `[examples]`
  extra installed
- **THEN** every notebook is executed and the run passes only if all notebooks complete without a
  cell error

#### Scenario: The default test run does not execute notebooks

- **WHEN** the default `pytest` suite is collected
- **THEN** it does not execute the notebooks (the nbmake harness is opt-in, keeping the default
  suite fast and CI free of PyVista-heavy notebook execution)
