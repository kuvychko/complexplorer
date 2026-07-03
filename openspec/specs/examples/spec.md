# Examples

## Purpose

The examples capability defines how the repository's `examples/` tree is organized and kept
honest: it derives its catalog content from the curated function preset registry rather than a
parallel hand-rolled generator, follows a documented directory layout (`notebooks/`, `scripts/`,
`gallery/`), and references no symbol removed at 3.0 nor any missing file. It is the contract that
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

The tutorials SHALL cover the headline 3.0 additions: a Riemann-**surface** example
(`riemann_surface_pv`) and the preset registry / gallery workflow (`cp.catalog`, and a pointer to
the gallery producer). Colormap material SHALL reference only implemented colormaps
(`Phase`, `Chessboard`, `PolarChessboard`, `LogRings`) — not the non-existent perceptual family.

#### Scenario: Riemann surfaces and the registry are demonstrated

- **WHEN** the tutorial set is read
- **THEN** at least one notebook demonstrates `riemann_surface_pv` and at least one demonstrates
  the preset registry (`cp.catalog`) with a pointer to the gallery producer

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
