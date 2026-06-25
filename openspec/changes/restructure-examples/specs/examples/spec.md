## ADDED Requirements

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

Example scripts and in-repository documentation SHALL NOT reference any symbol removed at 3.0
— the matplotlib 3D functions (`plot_landscape`, `pair_plot_landscape`, the 3D `riemann`) and
the deleted capability flags (`HAS_PYVISTA`, `HAS_STL_EXPORT`) — and SHALL NOT link to files
that do not exist in the repository.

#### Scenario: No example script imports or calls a removed symbol

- **WHEN** the Python files under `examples/` are scanned
- **THEN** none of them call `cp.plot_landscape`, `cp.pair_plot_landscape`, the 3D `cp.riemann`,
  or reference `HAS_PYVISTA` / `HAS_STL_EXPORT`; 3D demos use the PyVista `*_pv` functions

#### Scenario: Documentation links resolve to existing files

- **WHEN** the in-repo docs that point into `examples/` (the top-level `README.md`,
  `docs/README.md`, `docs/gallery/README.md`, `docs/pyvista_usage_guide.md`) are followed
- **THEN** every referenced notebook and script path resolves to a file that exists under the
  new layout (no link to a former `archive/`-only file such as `interactive_demo.py` or
  `plots_example.ipynb`)
