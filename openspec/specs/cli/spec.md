# Command-Line Interface

## Purpose

The cli capability provides a `complexplorer` console entry point exposing `render`, `stl`,
`list`, and `gallery` subcommands. It resolves a function from a preset or an expression, reuses
the spec factories for domain/colormap shorthands, drives the STL export pipeline and the gallery
generator, and reports library errors cleanly. PyVista is a required dependency, so every
subcommand is always available.
## Requirements
### Requirement: CLI entry point with render, stl, list, and gallery commands

The package SHALL provide a `complexplorer` console entry point (via `[project.scripts]`)
exposing `render`, `stl`, `list`, and `gallery` subcommands. `main(argv)` SHALL return a
process exit code and SHALL report any `ComplexplorerError` (not only `ValidationError`) as a
clean, non-zero exit with a message on stderr rather than an uncaught traceback.

#### Scenario: Entry point is installed

- **WHEN** the package is installed
- **THEN** a `complexplorer` console command is available and `complexplorer --help` lists
  the `render`, `stl`, `list`, and `gallery` subcommands

#### Scenario: Library errors are reported cleanly

- **WHEN** a subcommand raises any `ComplexplorerError` (e.g. a bad expression or an unknown preset id)
- **THEN** `main` prints an `error: ...` message to stderr and returns a non-zero exit code, without a Python traceback

### Requirement: Function argument resolves a preset or an expression

The `render` and `stl` commands SHALL accept a function argument that is either
`preset:<id>` (resolved through the function preset registry, using the preset's recommended
domain/colormap/scaling as defaults) or an expression string (evaluated via the expression
evaluator). Both commands SHALL apply the preset's recommended specs; in particular `stl`
SHALL use the preset's recommended domain (and colormap where the exporter accepts one) rather
than discarding them.

#### Scenario: Render a registry preset

- **WHEN** `render preset:pole_flower_10 --output out.png` is run
- **THEN** the preset's function and recommended specs are used and an image is written to
  `out.png`

#### Scenario: Render a raw expression

- **WHEN** `render "z**2 - 1" --domain rect:4:4 --output out.png` is run
- **THEN** the expression is evaluated, the domain shorthand is built into a `Domain`, and an
  image is written

#### Scenario: STL from a preset uses the preset's domain

- **WHEN** `stl preset:pole_flower_10 --output flower.stl` is run
- **THEN** the preset's recommended domain is passed to the ornament generator (not silently discarded)

### Requirement: Domain and colormap shorthands reuse the spec factories

CLI shorthands (e.g. `--domain annulus:0.2:3`, `--cmap phase:6`) SHALL be parsed into the
registry's spec dicts and built through the existing `domain_from_spec` / `cmap_from_spec`
factories — not a parallel construction path.

#### Scenario: Domain shorthand builds the right domain

- **WHEN** `--domain annulus:0.2:3` is supplied
- **THEN** it is parsed to `{"type": "annulus", "inner_radius": 0.2, "outer_radius": 3}` and
  built via `domain_from_spec`

### Requirement: STL export from the CLI

The `stl` command SHALL export a 3D-printable mesh for the resolved function via the surface
kernel / STL pipeline, honoring size and resolution options.

#### Scenario: Export an STL

- **WHEN** `stl preset:pole_flower_10 --size-mm 80 --output flower.stl` is run
- **THEN** a non-empty STL file is written at the requested size

### Requirement: Interactive display works in every render mode

`render --show` SHALL open an interactive window regardless of `--mode`. For `--mode 2d` this
SHALL display the matplotlib figure; for `--mode 3d|riemann` it SHALL open the PyVista window.
`--show` SHALL NOT be silently ignored in any mode.

#### Scenario: 2D show opens a window

- **WHEN** `render "z**2" --mode 2d --show` is run (no `--output`)
- **THEN** the matplotlib figure is displayed rather than the command drawing off-screen and exiting with no visible output

### Requirement: Console output survives a legacy console encoding

The CLI SHALL produce output on every console it can be run from, including a Windows console using
a legacy code page (cp437 in a stock `cmd.exe`, cp1252 for a redirected stream under many locales).
Preset titles and descriptions are mathematical text and SHALL keep their notation; where the
console cannot represent a character, the CLI SHALL degrade that character and continue, and SHALL
NOT fail with `UnicodeEncodeError`.

#### Scenario: Listing presets on a legacy code page

- **WHEN** `complexplorer list` is run on a console whose encoding cannot represent a preset title
  such as `z³ - z`
- **THEN** the command prints the full listing and exits 0, with only the unrepresentable
  characters degraded

#### Scenario: Notation is not stripped from the catalog

- **WHEN** preset titles and descriptions are authored
- **THEN** they may use mathematical notation, because the degradation happens at the console
  rather than in the catalog

