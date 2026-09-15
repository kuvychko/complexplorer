# 3D Plotting (PyVista)

## Purpose

The plotting-3d-pyvista capability provides high-performance, interactive 3D landscapes using
PyVista, the required 3D backend, mirroring the (now-removed) matplotlib 3D landscape functions
but with per-vertex coloring, smooth shading, and live camera interaction. As of 3.0 PyVista is a
required core dependency, so these functions are always available.

## Requirements

### Requirement: Interactive landscape surface

The library SHALL render `f(z)` over a domain as an interactive PyVista surface with per-vertex
color, height from `|f(z)|`, and configurable camera, window, and edge display.

#### Scenario: Interactive landscape

- **WHEN** `plot_landscape_pv` is given a domain and a function with interaction enabled
- **THEN** an interactive window is shown with a per-vertex-colored height surface

#### Scenario: Per-vertex scalars are attached

- **WHEN** the surface mesh is built
- **THEN** RGB color, magnitude, and phase are stored as mesh data so the surface colors without interpolation artifacts

#### Scenario: Returning the plotter for composition

- **WHEN** the caller requests the plotter object
- **THEN** the PyVista plotter is returned instead of being shown, so it can be embedded or exported

#### Scenario: Off-screen rendering and export

- **WHEN** interaction is disabled or a filename is provided
- **THEN** the scene is rendered off-screen and/or exported rather than requiring a display

### Requirement: Paired interactive landscape

The library SHALL render domain and codomain landscapes in two linked PyVista viewports that share
camera movement when interactive. A supplied `title` SHALL be applied as a figure-level title over
the paired scene and SHALL NOT replace the codomain panel's label.

#### Scenario: Linked dual viewports

- **WHEN** `pair_plot_landscape_pv` is called interactively
- **THEN** the identity-over-domain and `f(z)` surfaces appear in two viewports whose cameras are linked

#### Scenario: Title is a figure title, not a panel label

- **WHEN** `pair_plot_landscape_pv` is called with a `title`
- **THEN** the title is shown for the overall figure and the codomain panel keeps its own label (e.g. `Codomain f(z)`)

### Requirement: Keyword arguments are validated, not silently forwarded

The PyVista landscape entry points (`plot_landscape_pv`, `pair_plot_landscape_pv`) SHALL accept
only their documented parameters and SHALL NOT forward arbitrary keyword arguments into
`pyvista.Plotter`. An unrecognized keyword argument SHALL raise a `ValidationError`; for keyword
arguments removed in the 3.0 API migration, the error message SHALL name the current replacement
(e.g. `n_theta`/`n_phi` → `resolution`, `show` → `interactive`).

#### Scenario: Unknown keyword argument is rejected

- **WHEN** a landscape function is called with a keyword argument that is not part of its documented signature (e.g. `n_theta=200`)
- **THEN** a `ValidationError` is raised naming the offending argument (and its replacement when it is a known-removed 2.x name), rather than a raw `TypeError` from `pyvista.Plotter` or a silent no-op
