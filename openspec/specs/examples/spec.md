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
