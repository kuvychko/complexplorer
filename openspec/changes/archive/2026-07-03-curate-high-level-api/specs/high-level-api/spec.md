# High-Level API — Delta for curate-high-level-api

## MODIFIED Requirements

### Requirement: Mode- and backend-dispatching plot

The library SHALL provide a `quick_plot` function — the single high-level exploration entry
point — that selects among 2D, 3D landscape, and Riemann renderings, defaulting the domain
(`Rectangle(4, 4)`) and colormap (enhanced `Phase`) when not supplied. 2D uses matplotlib;
**3D and Riemann use PyVista, which is a required dependency** — there is no matplotlib 3D
backend. The `backend` selector SHALL NOT be forwarded to the underlying renderer, and
requesting `backend="matplotlib"` for a 3D or Riemann mode SHALL raise a clear error stating
the matplotlib 3D backend was removed. No alias names for `quick_plot` are exported.

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

## REMOVED Requirements

### Requirement: Quick exploration entry point

**Reason**: This requirement described a `show` function taking simple range specifications
(`(min, max)` / `(min, max, resolution)` tuples). No such function has ever existed in the
implementation — the requirement was a baseline-capture error. The quick-exploration role is
filled by `quick_plot`, now covered by the modified mode-dispatching requirement above.

**Migration**: Use `quick_plot(func)` — it defaults to a `Rectangle(4, 4)` domain, an enhanced
`Phase` colormap, and 2D mode. To control the viewing region, pass a `Domain`
(e.g. `Rectangle(re_length, im_length)`) instead of range tuples.

## ADDED Requirements

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
- **THEN** it contains `quick_plot` and `Presets`, and contains none of `visualize`, `explore`, or `analyze_function`
