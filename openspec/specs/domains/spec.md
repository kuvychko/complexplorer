# Domains

## Purpose

The domains capability defines regions of the complex plane over which functions are
sampled and visualized. It provides concrete region types (rectangle, disk, annulus),
a composition algebra (union, intersection, difference) for building irregular regions,
and the meshing/masking services that plotting and export capabilities rely on to
evaluate `f(z)` only where it is meaningful.

## Requirements

### Requirement: Region membership testing

Every domain SHALL classify any array of complex points as inside or outside the region,
with boundary points treated as inside.

#### Scenario: Disk includes its boundary

- **WHEN** a point satisfies `|z - center| == radius` for a `Disk(radius, center)`
- **THEN** the point is classified as inside the domain

#### Scenario: Annulus includes both boundaries

- **WHEN** a point satisfies `|z - center| == inner_radius` or `|z - center| == outer_radius`
- **THEN** the point is classified as inside the annulus

#### Scenario: Rectangle includes its edges

- **WHEN** a point lies exactly on a rectangle edge (real or imaginary extent equals the half-dimension)
- **THEN** the point is classified as inside the rectangle

### Requirement: Construction validation

A domain SHALL reject geometrically degenerate parameters at construction time by raising
`ValidationError`.

#### Scenario: Non-positive size is rejected

- **WHEN** a `Rectangle` is created with a non-positive `re_length` or `im_length`, or a `Disk`/`Annulus` with a non-positive radius
- **THEN** a `ValidationError` is raised

#### Scenario: Inverted annulus radii are rejected

- **WHEN** an `Annulus` is created with `outer_radius <= inner_radius`
- **THEN** a `ValidationError` is raised

#### Scenario: Degenerate viewing window is rejected

- **WHEN** a domain's real or imaginary viewing bounds are equal
- **THEN** a `ValidationError` is raised

### Requirement: Square viewing window

A domain SHALL support an optional square viewing window so that visualizations are not distorted;
when enabled, the shorter axis is expanded symmetrically to match the longer axis without changing
membership.

#### Scenario: Shorter axis expands symmetrically

- **WHEN** a domain with `square=True` has unequal real and imaginary extents
- **THEN** the viewing window expands the shorter axis equally on both sides to become square
- **AND** the membership test still uses the original, unexpanded geometry

### Requirement: Mesh generation

A domain SHALL produce a 2-D grid of complex sample points covering its viewing window at a
caller-specified resolution.

#### Scenario: Resolution controls sampling density

- **WHEN** `mesh(n)` is called with resolution `n` (default 500)
- **THEN** a 2-D complex grid is returned whose spacing is the longer axis length divided by `n`
- **AND** the grid extends to the window edges

#### Scenario: Resolution is bounds-checked

- **WHEN** a resolution outside the supported range (2 to 10000) is requested
- **THEN** a validation error is raised

### Requirement: Interior masking services

A domain SHALL expose masks identifying which mesh points lie inside versus outside the region,
and a masked mesh in which exterior points are marked invalid.

#### Scenario: Interior and exterior masks are complementary

- **WHEN** `inmask(n)` and `outmask(n)` are requested for the same resolution
- **THEN** `outmask` is the boolean complement of `inmask`

#### Scenario: Masked mesh marks exterior as NaN

- **WHEN** `domain(n)` is requested
- **THEN** the returned complex mesh contains the sample value at interior points and `NaN` at exterior points

### Requirement: Domain composition algebra

A domain SHALL support combination with another domain via union, intersection, and difference,
producing a composite domain whose membership follows set semantics.

#### Scenario: Union admits points in either region

- **WHEN** two domains are combined with `union` (or the `|` operator)
- **THEN** a point is inside the composite if it is inside either source domain

#### Scenario: Intersection admits points in both regions

- **WHEN** two domains are combined with `intersection` (or the `&` operator)
- **THEN** a point is inside the composite only if it is inside both source domains

#### Scenario: Difference subtracts the second region

- **WHEN** two domains are combined with `difference` (or the `-` operator)
- **THEN** a point is inside the composite only if it is inside the first domain and outside the second

#### Scenario: Invalid operation name is rejected

- **WHEN** a composite domain is constructed with an operation other than union, intersection, or difference
- **THEN** a `ValidationError` is raised

### Requirement: Tight bounds for composite domains

A composite domain SHALL be able to compute a tight viewing window around the points that
actually belong to it, so that irregular regions are framed without excessive empty space.

#### Scenario: Tight bounds frame the populated region

- **WHEN** tight bounds are requested for a composite domain that contains points
- **THEN** the bounds enclose the in-domain points with a small margin and a minimum extent
- **AND** the result is reused on subsequent requests rather than recomputed

#### Scenario: Empty composite yields a safe fallback window

- **WHEN** tight bounds are requested for a composite domain whose regions do not overlap (no in-domain points sampled)
- **THEN** a small non-degenerate fallback window is returned rather than an invalid one

#### Scenario: Composite meshing uses tight bounds by default

- **WHEN** a composite domain is meshed without overriding the bounds option
- **THEN** the mesh covers the tight bounds rather than the full union window
