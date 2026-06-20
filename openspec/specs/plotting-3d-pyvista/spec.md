# 3D Plotting (PyVista)

## Purpose

The plotting-3d-pyvista capability provides high-performance, interactive 3D landscapes using
PyVista as an optional backend, mirroring the matplotlib 3D landscape functions but with
per-vertex coloring, smooth shading, and live camera interaction. It degrades cleanly when PyVista
is not installed.

## Requirements

### Requirement: Optional-dependency gating

PyVista-backed plotting SHALL be available only when PyVista is installed and SHALL fail with a
clear error otherwise, without breaking import of the rest of the library.

#### Scenario: Missing PyVista raises a clear error

- **WHEN** a PyVista plotting function is called and PyVista is not installed
- **THEN** an `ImportError` explaining that PyVista is required (with install guidance) is raised

#### Scenario: Library imports without PyVista

- **WHEN** the package is imported without PyVista present
- **THEN** the core and matplotlib functionality remains usable and a capability flag reports PyVista as unavailable

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
camera movement when interactive.

#### Scenario: Linked dual viewports

- **WHEN** `pair_plot_landscape_pv` is called interactively
- **THEN** the identity-over-domain and `f(z)` surfaces appear in two viewports whose cameras are linked
