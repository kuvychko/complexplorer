## MODIFIED Requirements

### Requirement: Curated exported surface

Every callable exported by the high-level API SHALL be fully implemented (this covers
`complexplorer.api` and its top-level re-exports): the module SHALL NOT export stubs that raise
`NotImplementedError` or advertise unimplemented behavior. `quick_plot` SHALL be the only
quick-exploration entry point; the former `visualize`/`explore` aliases and the
`analyze_function`, `create_animation`, and `compare_functions` stubs are not part of the
surface.

The exported surface SHALL be visualization API. Environment plumbing — configuring a matplotlib
backend, or turning interactive mode on — SHALL NOT be part of it: those are matplotlib's own
concern, and exporting them invites users to depend on this library for something it merely wraps.

#### Scenario: Removed stubs are not importable

- **WHEN** a user attempts to import `create_animation`, `compare_functions`, or `analyze_function` from `complexplorer.api`, or `visualize`, `explore`, or `analyze_function` from `complexplorer`
- **THEN** an `ImportError` is raised because the names no longer exist

#### Scenario: Top-level surface lists only working entry points

- **WHEN** `complexplorer.__all__` is inspected
- **THEN** it contains `quick_plot` and `PlotPresets`, and contains none of `visualize`, `explore`, or `analyze_function`

#### Scenario: Backend helpers are not part of the public surface

- **WHEN** `complexplorer.__all__` is inspected
- **THEN** it does not contain `setup_matplotlib_backend` or `ensure_interactive_plots`, and the
  migration guide names what to use instead
