## MODIFIED Requirements

### Requirement: Domain coloring plot

The library SHALL render `f(z)` over a domain as a 2D image whose colors come from a colormap and
whose exterior points are masked, accepting either a domain plus function or pre-computed mesh
arrays. `plot` SHALL always return the matplotlib `Axes` it drew on, and SHALL honor a supplied
`filename` regardless of whether the caller also passed an `ax`.

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
- **THEN** the figure is not shown automatically

#### Scenario: Filename is honored with a caller-supplied axes

- **WHEN** `plot` is called with both an `ax` and a `filename`
- **THEN** the figure containing that axes is written to `filename` (saving is not skipped just because an axes was supplied)

#### Scenario: The drawn axes is always returned

- **WHEN** `plot` returns
- **THEN** it returns the matplotlib `Axes` used, whether or not the caller supplied `ax`

### Requirement: Riemann hemisphere chart

The library SHALL render a flat stereographic view of a Riemann sphere hemisphere, projecting the
unit disk (plus a margin) and visually distinguishing the interior, exterior, and the unit circle.
When an optional `domain` is supplied, samples outside that domain SHALL be masked with the
colormap's out-of-domain color.

#### Scenario: Single hemisphere chart

- **WHEN** `riemann_chart` is called for a hemisphere
- **THEN** the function is sampled over a disk of radius `1 + margin`, points outside the unit circle are desaturated, and the unit circle is highlighted

#### Scenario: Margin bounds are enforced

- **WHEN** a margin outside `[0.0, 0.5]` is requested
- **THEN** a `ValidationError` is raised

#### Scenario: Both hemispheres at once

- **WHEN** `riemann_hemispheres` is called
- **THEN** the south and north hemisphere charts are drawn side by side in one figure

#### Scenario: Optional domain masks the chart

- **WHEN** `riemann_chart` is given a `domain`
- **THEN** samples for which `domain.contains` is false are rendered with the out-of-domain color
- **AND** when no `domain` is given, no domain masking is applied
