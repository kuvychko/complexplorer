# 2D Plotting (Matplotlib)

## Purpose

The plotting-2d capability renders complex functions as flat domain-coloring images using
matplotlib: the standard domain plot, a side-by-side domain/codomain pair, and flat stereographic
"chart" views of the Riemann sphere hemispheres. It ties together a domain (for sampling and
masking), a function, and a colormap, and produces matplotlib artists without forcing display.

## Requirements

### Requirement: Domain coloring plot

The library SHALL render `f(z)` over a domain as a 2D image whose colors come from a colormap and
whose exterior points are masked, accepting either a domain plus function or pre-computed mesh
arrays.

#### Scenario: Plot from domain and function

- **WHEN** `plot` is given a domain and a function
- **THEN** the function is sampled over the domain mesh at the requested resolution and drawn as an image with real on the horizontal axis and imaginary on the vertical axis (increasing upward)
- **AND** points outside the domain are rendered with the colormap's out-of-domain color

#### Scenario: Plot from pre-computed arrays

- **WHEN** `plot` is given mesh arrays `z` and values `f` directly
- **THEN** the arrays are rendered as-is without domain masking

#### Scenario: Default colormap is supplied

- **WHEN** no colormap is provided
- **THEN** an enhanced auto-scaled `Phase` portrait is used by default

#### Scenario: Missing inputs are rejected

- **WHEN** neither a domain nor a `z` array is provided, or neither a function nor an `f` array is provided
- **THEN** a `ValidationError` is raised

#### Scenario: Display is left to the caller

- **WHEN** a plot is produced
- **THEN** the figure is not shown automatically; it is only written to disk if a filename is provided

### Requirement: Paired domain and codomain plot

The library SHALL render the input domain and the mapped codomain side by side using a shared
colormap and resolution.

#### Scenario: Pair plot shows identity and function

- **WHEN** `pair_plot` is given a domain and a function
- **THEN** the left panel shows the identity map over the domain and the right panel shows `f(z)`, returning a single figure

### Requirement: Riemann hemisphere chart

The library SHALL render a flat stereographic view of a Riemann sphere hemisphere, projecting the
unit disk (plus a margin) and visually distinguishing the interior, exterior, and the unit circle.

#### Scenario: Single hemisphere chart

- **WHEN** `riemann_chart` is called for a hemisphere
- **THEN** the function is sampled over a disk of radius `1 + margin`, points outside the unit circle are desaturated, and the unit circle is highlighted

#### Scenario: Margin bounds are enforced

- **WHEN** a margin outside `[0.0, 0.5]` is requested
- **THEN** a `ValidationError` is raised

#### Scenario: Both hemispheres at once

- **WHEN** `riemann_hemispheres` is called
- **THEN** the south and north hemisphere charts are drawn side by side in one figure
