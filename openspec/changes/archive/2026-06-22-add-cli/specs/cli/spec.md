# Command-Line Interface

## ADDED Requirements

### Requirement: CLI entry point with render, stl, and list commands

The package SHALL provide a `complexplorer` console entry point (via `[project.scripts]`)
exposing `render`, `stl`, and `list` subcommands. `main(argv)` SHALL return a process exit
code.

#### Scenario: Entry point is installed

- **WHEN** the package is installed
- **THEN** a `complexplorer` console command is available and `complexplorer --help` lists
  the `render`, `stl`, and `list` subcommands

### Requirement: Function argument resolves a preset or an expression

The `render` and `stl` commands SHALL accept a function argument that is either
`preset:<id>` (resolved through the function preset registry, using the preset's recommended
domain/colormap/scaling as defaults) or an expression string (evaluated via the expression
evaluator).

#### Scenario: Render a registry preset

- **WHEN** `render preset:pole_flower_10 --output out.png` is run
- **THEN** the preset's function and recommended specs are used and an image is written to
  `out.png`

#### Scenario: Render a raw expression

- **WHEN** `render "z**2 - 1" --domain rect:4:4 --output out.png` is run
- **THEN** the expression is evaluated, the domain shorthand is built into a `Domain`, and an
  image is written

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

### Requirement: 2D commands work without the 3D backend

`render --mode 2d` and `list` SHALL function without PyVista installed. Commands that need
the 3D backend (`render --mode 3d|riemann`, `stl`) SHALL exit with a clear message when
PyVista is absent.

#### Scenario: list without PyVista

- **WHEN** `list` is run in an environment without PyVista
- **THEN** it prints the available presets and exits successfully

#### Scenario: stl without PyVista fails clearly

- **WHEN** `stl …` is run without PyVista
- **THEN** the command exits non-zero with a message that PyVista is required for 3D/STL
