## MODIFIED Requirements

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

## REMOVED Requirements

### Requirement: 2D commands work without the 3D backend

**Reason**: PyVista became a required core dependency at 3.0, so "without PyVista installed"
is no longer a supported configuration. Graceful degradation when the 3D backend is absent —
and the availability checks that implemented it — are dead behavior that contradicts the
dependency contract.

**Migration**: Installing `complexplorer` always installs PyVista; all subcommands
(`render` in every mode, `stl`, `list`, `gallery`) are always available. No user action is
required.

## ADDED Requirements

### Requirement: Interactive display works in every render mode

`render --show` SHALL open an interactive window regardless of `--mode`. For `--mode 2d` this
SHALL display the matplotlib figure; for `--mode 3d|riemann` it SHALL open the PyVista window.
`--show` SHALL NOT be silently ignored in any mode.

#### Scenario: 2D show opens a window

- **WHEN** `render "z**2" --mode 2d --show` is run (no `--output`)
- **THEN** the matplotlib figure is displayed rather than the command drawing off-screen and exiting with no visible output
