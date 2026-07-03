# Riemann Surfaces — Delta for add-algebraic-curves

## MODIFIED Requirements

### Requirement: Multivalued-family Riemann surface rendering

The library SHALL provide a PyVista renderer that draws the Riemann surface of a multivalued
function for the supported families — power roots `z^(1/n)`, the logarithm, and the algebraic
family `w² = P(z)` for a polynomial `P` — embedding the surface in 3D and coloring it by the
phase of the function value. The renderer SHALL be distinct from the Riemann *sphere* renderer
(which visualizes a single-valued function on the compactified plane).

#### Scenario: Power-root surface has n sheets

- **WHEN** the renderer is called for `family="power"` with degree `n`
- **THEN** it produces an `n`-sheeted surface of `z^(1/n)` (obtained by sampling `w` and mapping `z = w^n`), embedded with height equal to the real part of the value (so the self-intersection lies along the negative real axis, the principal-branch cut), and colored by the value's phase

#### Scenario: Logarithm surface is a helicoid

- **WHEN** the renderer is called for `family="log"` with a number of turns
- **THEN** it produces a helicoidal surface spanning the requested turns, with height equal to the imaginary part of the logarithm

#### Scenario: Algebraic curve is a two-sheeted cover

- **WHEN** the renderer is called for `family="algebraic"` with polynomial coefficients `p` (highest degree first, degree ≥ 1)
- **THEN** it produces the two-sheeted surface of `w² = P(z)` over a disk of radius `r_max`: the sheets are the graphs of `±Re(√P(z))`, each colored by the phase of its value, intersecting along the curves where `P(z) ≤ 0` so branch points (the roots of `P`) and cuts are emergent from the geometry

#### Scenario: Standard plotter options are honored

- **WHEN** the renderer is called with the shared options (colormap, interactivity, return-plotter, output filename)
- **THEN** it honors them consistently with the other PyVista renderers (returning a plotter or writing a file as requested)

## ADDED Requirements

### Requirement: Algebraic-family inputs and branch-point metadata

The algebraic family SHALL validate its polynomial input and SHALL expose the branch points it
implies: `p` must contain at least two coefficients with a nonzero leading coefficient, and the
returned `SurfaceMesh` metadata SHALL record the roots of `P` as branch points.

#### Scenario: Branch points recorded in metadata

- **WHEN** the surface builder is invoked with `family="algebraic"` and a valid `p`
- **THEN** the returned `SurfaceMesh.metadata` contains the roots of `P` (the finite branch points of the curve) and a topology tag identifying the algebraic family and degree

#### Scenario: Invalid polynomial is rejected

- **WHEN** `family="algebraic"` is requested with `p` missing, shorter than two coefficients, or with a zero leading coefficient
- **THEN** a `ValidationError` is raised naming the problem
