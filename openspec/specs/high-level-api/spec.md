# High-Level API

## Purpose

The high-level-api capability is the friendly front door to the library: a quick `show` for
one-line exploration, a more configurable `plot` that dispatches across 2D, 3D, and Riemann modes
and backends, and named presets that bundle good colormap and resolution settings for common goals.
It composes the lower-level domain, colormap, and plotting capabilities so casual users need not
assemble them by hand.

## Requirements

### Requirement: Quick exploration entry point

The library SHALL provide a `show` function that visualizes a function from a simple range
specification with minimal boilerplate, defaulting to a sensible domain, colormap, and 2D mode.

#### Scenario: Show with default ranges

- **WHEN** `show` is called with only a function
- **THEN** a square domain is built from the default range, an enhanced `Phase` colormap is applied, and a 2D plot is produced

#### Scenario: Range tuples may carry resolution

- **WHEN** a range is given as `(min, max)` or `(min, max, resolution)`, and the vertical range is omitted
- **THEN** the resolution defaults when absent and the horizontal range is reused for both axes

### Requirement: Mode- and backend-dispatching plot

The library SHALL provide a `plot` function that selects among 2D, 3D landscape, and Riemann
renderings, defaulting the domain and colormap when not supplied. 2D uses matplotlib; **3D and
Riemann use PyVista, which is a required dependency** — there is no matplotlib 3D backend. The
`backend` selector SHALL NOT be forwarded to the underlying renderer, and requesting
`backend="matplotlib"` for a 3D or Riemann mode SHALL raise a clear error stating the
matplotlib 3D backend was removed.

#### Scenario: Mode selects the renderer

- **WHEN** `plot` is called with mode `2d`, `3d`, or `riemann`
- **THEN** the corresponding renderer is invoked (matplotlib for 2D, PyVista for 3D/Riemann) and its native result (matplotlib axes or PyVista plotter) is returned

#### Scenario: 3D and Riemann always use PyVista

- **WHEN** a 3D or Riemann plot is requested without an explicit backend
- **THEN** the PyVista renderer is used (and renderer-specific options such as a modulus scaling mode are accepted)

#### Scenario: Requesting the removed matplotlib 3D backend errors

- **WHEN** `plot` is called with mode `3d` or `riemann` and `backend="matplotlib"`
- **THEN** an error is raised explaining the matplotlib 3D backend was removed in 3.0

#### Scenario: Unknown mode is rejected

- **WHEN** `plot` is called with a mode other than `2d`, `3d`, or `riemann`
- **THEN** an error is raised naming the unknown mode

### Requirement: Named configuration presets

The library SHALL provide named presets that return bundled colormap and resolution settings for
common rendering goals, usable as keyword arguments to the plotting entry points.

#### Scenario: Presets return ready-to-use settings

- **WHEN** the publication, interactive, or high-contrast preset is requested
- **THEN** a settings bundle is returned with a configured `Phase` colormap and a resolution tuned for that goal (higher resolution and more phase sectors for publication, balanced for interactive, many sectors and tighter modulus scaling for high contrast)
