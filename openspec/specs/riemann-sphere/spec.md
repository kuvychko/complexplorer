# Riemann Sphere

## Purpose

The riemann-sphere capability is the cross-cutting concept of visualizing a complex function on the
one-point-compactified plane (the sphere), including the point at infinity. It defines the shared
contract that the matplotlib `riemann` and PyVista `riemann_pv` renderers honor: stereographic
mapping between plane and sphere, phase coloring of the surface, and optional modulus-driven radial
relief ("Riemann relief maps"). The concrete rendering lives in the matplotlib and PyVista 3D
plotting capabilities; this spec captures the behavior they must agree on.

## Requirements

### Requirement: Sphere parameterization via stereographic projection

A Riemann sphere rendering SHALL map each sphere surface point to a complex value by stereographic
projection so that the origin and the point at infinity occupy opposite poles and the unit circle
maps to the equator.

#### Scenario: Poles correspond to zero and infinity

- **WHEN** a sphere surface is built and projected to the plane
- **THEN** one pole corresponds to `z = 0` and the opposite pole to `z = ∞`, with `|z| = 1` mapping to the equator

#### Scenario: Function is evaluated on the projected plane

- **WHEN** the sphere is colored
- **THEN** `f(z)` is evaluated at the projected complex value of each surface point and passed through the colormap, with non-finite values handled without error

### Requirement: Phase-only sphere by default

A Riemann sphere rendering SHALL default to an undistorted unit sphere that shows phase (and
colormap structure) only.

#### Scenario: Default constant modulus

- **WHEN** a Riemann rendering is requested without choosing a modulus mode
- **THEN** the sphere keeps unit radius and conveys only the colormap encoding of `f(z)`

### Requirement: Modulus relief distortion

A Riemann sphere rendering SHALL support distorting the sphere radius per point by a modulus
scaling mode applied to `|f(z)|`, producing a relief map of the function's magnitude.

#### Scenario: Relief from a modulus mode

- **WHEN** a non-constant modulus mode is selected
- **THEN** each surface point's radius is scaled by that mode applied to `|f(z)|`, raising peaks where the magnitude is large

#### Scenario: Scaling parameters default sensibly

- **WHEN** a modulus mode is selected without explicit parameters
- **THEN** visualization-appropriate default parameters for that mode are used

### Requirement: Backend parity with documented differences

The matplotlib and PyVista Riemann renderers SHALL present the same conceptual
visualization, while each MAY differ in mesh construction, default resolution,
interactivity, and grid overlays. They SHALL NOT differ in projection orientation: both use
the canonical convention (`z = 0` at the south pole).

#### Scenario: Both backends render the same concept

- **WHEN** the same function and modulus mode are rendered by the matplotlib and PyVista Riemann renderers
- **THEN** both show a phase-colored sphere with equivalent modulus relief and the same orientation, differing only in performance, interactivity, and backend-specific options (such as the PyVista latitude/longitude grid overlay)

### Requirement: Canonical stereographic projection convention

All sphere renderers and the STL ornament SHALL use one canonical stereographic projection
convention via a single shared projection implementation — the documented core convention
in which `z = 0` maps to the south pole and `z = ∞` to the north pole — so that the same
function yields the same orientation across matplotlib `riemann`, PyVista `riemann_pv`, and
the exported ornament, and the printed object matches the rendered sphere.

#### Scenario: All sphere outputs share one orientation

- **WHEN** the same function is rendered by `riemann`, by `riemann_pv`, and exported as an
  STL ornament
- **THEN** all three place `z = 0` at the same pole (south) and are not mirror images of
  one another

#### Scenario: Single projection implementation

- **WHEN** a sphere point is mapped to a complex value for any sphere renderer or for STL
  export
- **THEN** the mapping uses one canonical stereographic projection function (no divergent
  duplicate implementations)
