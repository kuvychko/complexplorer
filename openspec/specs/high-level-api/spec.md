# High-Level API

## Purpose

The high-level-api capability is the friendly front door to the library: a single `quick_plot`
entry point that dispatches across 2D, 3D, and Riemann modes, and named presets that bundle good
colormap and resolution settings for common goals. It composes the lower-level domain, colormap,
and plotting capabilities so casual users need not assemble them by hand, and it exports only
implemented functionality — no stubs, no aliases.
## Requirements
### Requirement: Mode- and backend-dispatching plot

The library SHALL provide a `quick_plot` function — the single high-level exploration entry
point — that selects among 2D, 3D landscape, and Riemann renderings, defaulting the domain
(`Rectangle(4, 4)`) and colormap (enhanced `Phase`) when not supplied. 2D uses matplotlib;
**3D and Riemann use PyVista, which is a required dependency** — there is no matplotlib 3D
backend. The `backend` selector SHALL NOT be forwarded to the underlying renderer, and
requesting `backend="matplotlib"` for a 3D or Riemann mode SHALL raise a clear error stating
the matplotlib 3D backend was removed. A caller-supplied `domain` SHALL be forwarded to the
selected renderer in every mode, including Riemann. No alias names for `quick_plot` are exported.

#### Scenario: Quick plot with only a function

- **WHEN** `quick_plot` is called with only a function
- **THEN** a `Rectangle(4, 4)` domain and an enhanced `Phase` colormap are applied and a 2D plot is produced

#### Scenario: Mode selects the renderer

- **WHEN** `quick_plot` is called with mode `2d`, `3d`, or `riemann`
- **THEN** the corresponding renderer is invoked (matplotlib for 2D, PyVista for 3D/Riemann) and its native result (matplotlib axes or PyVista plotter) is returned

#### Scenario: 3D and Riemann always use PyVista

- **WHEN** a 3D or Riemann plot is requested without an explicit backend
- **THEN** the PyVista renderer is used (and renderer-specific options such as a modulus scaling mode are accepted)

#### Scenario: Requesting the removed matplotlib 3D backend errors

- **WHEN** `quick_plot` is called with mode `3d` or `riemann` and `backend="matplotlib"`
- **THEN** an error is raised explaining the matplotlib 3D backend was removed in 3.0

#### Scenario: Unknown mode is rejected

- **WHEN** `quick_plot` is called with a mode other than `2d`, `3d`, or `riemann`
- **THEN** an error is raised naming the unknown mode

#### Scenario: A supplied domain is forwarded in Riemann mode

- **WHEN** `quick_plot` is called with mode `riemann` and an explicit `domain`
- **THEN** that domain is passed through to `riemann_pv` (used to mask the sphere) rather than being discarded in favor of the default

### Requirement: Named configuration presets

The library SHALL provide named presets that return bundled colormap and resolution settings for
common rendering goals, usable as keyword arguments to the plotting entry points. They SHALL be
exposed as `PlotPresets`, a name that states what they configure and cannot be confused with the
function registry `catalog`. Each SHALL declare its return type.

#### Scenario: Presets return ready-to-use settings

- **WHEN** the publication, interactive, or high-contrast preset is requested from `PlotPresets`
- **THEN** a settings bundle is returned with a configured `Phase` colormap and a resolution tuned for that goal (higher resolution and more phase sectors for publication, balanced for interactive, many sectors and tighter modulus scaling for high contrast)

#### Scenario: The older name is gone

- **WHEN** `complexplorer.Presets` is accessed
- **THEN** it is not part of the public surface, because the published 2.x releases never exposed
  it and 3.0 is where the name is settled

### Requirement: Curated exported surface

Every callable exported by the high-level API SHALL be fully implemented (this covers
`complexplorer.api` and its top-level re-exports): the module SHALL NOT export stubs that raise
`NotImplementedError` or advertise unimplemented behavior. `quick_plot` SHALL be the only
quick-exploration entry point; the former `visualize`/`explore` aliases and the
`analyze_function`, `create_animation`, and `compare_functions` stubs are not part of the
surface.

#### Scenario: Removed stubs are not importable

- **WHEN** a user attempts to import `create_animation`, `compare_functions`, or `analyze_function` from `complexplorer.api`, or `visualize`, `explore`, or `analyze_function` from `complexplorer`
- **THEN** an `ImportError` is raised because the names no longer exist

#### Scenario: Top-level surface lists only working entry points

- **WHEN** `complexplorer.__all__` is inspected
- **THEN** it contains `quick_plot` and `PlotPresets`, and contains none of `visualize`, `explore`, or `analyze_function`

### Requirement: The public typing contract describes what the library actually calls

The package ships `py.typed`, so its annotations are a promise to downstream type checkers. A
user-supplied function SHALL be typed by a `ComplexFunction` protocol describing what the
renderers actually pass: an array of complex values in, an array of complex values out, with
scalar input accepted. `Callable[[complex], complex]` SHALL NOT be used for such parameters,
because it describes a scalar contract the library does not use.

Every public callable SHALL declare an explicit return type.

#### Scenario: A vectorized function satisfies the annotation

- **WHEN** a user passes a function that maps an array of complex values to an array of complex
  values to a public entry point
- **THEN** a type checker accepts it

#### Scenario: A downstream program type-checks

- **WHEN** a small program that imports complexplorer and uses its public entry points is checked
  with pyright
- **THEN** the check passes, and it is run in CI so it keeps passing

