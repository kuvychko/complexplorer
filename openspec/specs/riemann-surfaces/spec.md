# Riemann Surfaces

## Purpose

The riemann-surfaces capability covers rendering the Riemann surface of a multivalued complex
function for the supported families (power roots and the logarithm). It is distinct from the
riemann-sphere capability: instead of visualizing a single-valued function on the compactified
plane, it embeds the multi-sheeted surface of a multivalued function in 3D, colored by phase.
The feature is visualization-only and reuses the shared surface kernel mesh and colormap path.

## Requirements

### Requirement: Multivalued-family Riemann surface rendering

The library SHALL provide a PyVista renderer that draws the Riemann surface of a multivalued
function for the supported families — power roots `z^(1/n)` and the logarithm — embedding the
surface in 3D and coloring it by the phase of the function value. The renderer SHALL be
distinct from the Riemann *sphere* renderer (which visualizes a single-valued function on the
compactified plane).

#### Scenario: Power-root surface has n sheets

- **WHEN** the renderer is called for `family="power"` with degree `n`
- **THEN** it produces an `n`-sheeted surface of `z^(1/n)` (obtained by sampling `w` and mapping `z = w^n`), embedded with height equal to the real part of the value (so the self-intersection lies along the negative real axis, the principal-branch cut), and colored by the value's phase

#### Scenario: Logarithm surface is a helicoid

- **WHEN** the renderer is called for `family="log"` with a number of turns
- **THEN** it produces a helicoidal surface spanning the requested turns, with height equal to the imaginary part of the logarithm

#### Scenario: Standard plotter options are honored

- **WHEN** the renderer is called with the shared options (colormap, interactivity, return-plotter, output filename)
- **THEN** it honors them consistently with the other PyVista renderers (returning a plotter or writing a file as requested)

### Requirement: Riemann surface meshes reuse the surface kernel

The Riemann surface builder SHALL produce a `SurfaceMesh` (the shared kernel mesh) decorated
through the standard color path, so the surface participates in the common mesh pipeline
without special-casing.

#### Scenario: Builder returns a decorated surface mesh

- **WHEN** the surface builder is invoked for a supported family
- **THEN** it returns a `SurfaceMesh` whose geometry embeds the parameter grid and whose colors come from the shared colormap path (finite RGB within `[0, 1]`)

### Requirement: Honest embedding; visualization-only scope

The Riemann surface SHALL use the faithful ("honest") embedding in which sheets and branch
cuts are emergent from the geometry (the power surface self-intersects along the cut). The
feature is visualization-only.

#### Scenario: Self-intersection is preserved, not separated

- **WHEN** a power-root surface with `n ≥ 2` is built
- **THEN** the surface is a single continuous mesh that passes through itself along the branch cut (the sheets are not artificially separated)

#### Scenario: STL export is not offered for Riemann surfaces

- **WHEN** a Riemann surface is produced
- **THEN** it is a visualization object and no STL export is provided for it (the self-intersecting geometry is non-manifold)
