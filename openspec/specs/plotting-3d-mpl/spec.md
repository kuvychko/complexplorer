# 3D Plotting (Matplotlib)

## Purpose

The plotting-3d-mpl capability renders complex functions as 3D surfaces using matplotlib: analytic
landscapes where height encodes magnitude and color encodes phase, a side-by-side domain/codomain
landscape pair, and a Riemann sphere surface. It is the dependency-free 3D backend; the
plotting-3d-pyvista capability mirrors it for higher performance.

## Requirements

### Requirement: Analytic landscape surface

The library SHALL render `f(z)` over a domain as a 3D surface whose height is derived from
`|f(z)|` and whose color comes from a colormap, with the height transform selectable.

#### Scenario: Landscape from domain and function

- **WHEN** `plot_landscape` is given a domain and a function
- **THEN** a 3D surface is drawn with real and imaginary on the horizontal axes and a height derived from `|f(z)|`, colored by the colormap

#### Scenario: Height transform options

- **WHEN** a modulus scaling mode, a height cap, or a logarithmic height is requested
- **THEN** the height is transformed accordingly before rendering (cap applied before log)

#### Scenario: Default height is raw magnitude

- **WHEN** no modulus scaling mode is specified
- **THEN** the height equals `|f(z)|` directly

#### Scenario: Missing inputs are rejected

- **WHEN** neither a domain nor a `z` array, or neither a function nor an `f` array, is provided
- **THEN** a `ValidationError` is raised

### Requirement: Paired landscape

The library SHALL render domain and codomain landscapes side by side with shared colormap and
height settings, returning a single figure.

#### Scenario: Pair landscape shows identity and function

- **WHEN** `pair_plot_landscape` is called
- **THEN** the left surface is the identity map over the domain and the right surface is `f(z)`, with the same scaling applied to both

### Requirement: Riemann sphere surface

The library SHALL render `f(z)` on a sphere, coloring the surface by the function and optionally
distorting the radius by a modulus scaling mode.

#### Scenario: Phase-only sphere by default

- **WHEN** `riemann` is called with the default constant modulus mode
- **THEN** an undistorted unit sphere is rendered, colored by `f(z)`

#### Scenario: Modulus relief distorts the radius

- **WHEN** a non-constant modulus mode is selected
- **THEN** the sphere radius is scaled per-point by the mode applied to `|f(z)|`

#### Scenario: Projection pole is selectable

- **WHEN** the projection-from-north option is toggled
- **THEN** the assignment of the origin and infinity to the sphere poles flips accordingly

### Requirement: Matplotlib 3D paths are deprecated toward PyVista-only

The matplotlib 3D entry points SHALL be deprecated in favor of their PyVista equivalents
and signal their removal at 3.0. This applies to the 3D surface functions only; the 2D
stereographic charts (`riemann_chart`, `riemann_hemispheres`) are unaffected.

#### Scenario: Deprecation warning on matplotlib 3D landscape

- **WHEN** `plot_landscape` or `pair_plot_landscape` is called
- **THEN** a `DeprecationWarning` is emitted naming the PyVista replacement
  (`plot_landscape_pv` / `pair_plot_landscape_pv`) and stating removal at 3.0, and the
  plot is still produced

#### Scenario: Deprecation warning on matplotlib 3D Riemann sphere

- **WHEN** the 3D `riemann()` surface function is called
- **THEN** a `DeprecationWarning` is emitted naming `riemann_pv` as the replacement and
  stating removal at 3.0, and the plot is still produced

#### Scenario: 2D stereographic charts are not deprecated

- **WHEN** `riemann_chart` or `riemann_hemispheres` is called
- **THEN** no backend-deprecation warning is emitted, because these are matplotlib 2D
  features that remain supported
