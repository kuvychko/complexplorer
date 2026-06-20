# Core Functions

## Purpose

The core-functions capability provides the foundational mathematical primitives shared by
colormaps, plotting, and export: phase extraction, periodic sawtooth waves for contouring,
stereographic projection between the complex plane and the Riemann sphere, and smooth
interpolation helpers. These are pure, vectorized functions with no rendering side effects.

## Requirements

### Requirement: Phase extraction

The library SHALL compute the phase (argument) of complex values normalized to the half-open
range `[0, 2π)` so that phase maps consistently onto a cyclic color wheel.

#### Scenario: Positive real axis maps to zero

- **WHEN** the phase of `1 + 0j` is computed
- **THEN** the result is `0`

#### Scenario: Negative arguments wrap into the positive range

- **WHEN** the phase of a value with negative `np.angle` (e.g. `0 - 1j`) is computed
- **THEN** the result is shifted by `2π` into `[0, 2π)` (e.g. `3π/2`)

#### Scenario: Shape and type are preserved

- **WHEN** phase is computed for a scalar versus an array
- **THEN** a scalar returns a float and an array returns an array of the same shape

### Requirement: Sawtooth contouring waves

The library SHALL provide periodic sawtooth functions, in both linear and logarithmic forms,
that map an input to a ramp in `[0, 1)` for generating evenly spaced contour bands.

#### Scenario: Linear sawtooth ramps and wraps

- **WHEN** `sawtooth(x, period)` is evaluated
- **THEN** the result is `mod(x / period, 1.0)`, ramping from 0 up to but not including 1 and wrapping at each period

#### Scenario: Logarithmic sawtooth handles the origin

- **WHEN** `sawtooth_log(x)` is evaluated at `x == 0`
- **THEN** the result is `0` rather than an invalid value from the logarithm of zero

### Requirement: Stereographic projection to the Riemann sphere

The library SHALL map complex values onto the surface of the unit Riemann sphere and back,
sending the point at infinity to a pole, with a selectable projection pole.

#### Scenario: Origin maps to a pole

- **WHEN** `stereographic_projection(0 + 0j)` is computed with the default (south-pole) convention
- **THEN** the result is the point `(0, 0, -1)` on the unit sphere

#### Scenario: Unit circle maps to the equator

- **WHEN** a value with `|z| == 1` (e.g. `1 + 0j`) is projected
- **THEN** the result lies on the sphere's equator

#### Scenario: Infinity maps to the opposite pole

- **WHEN** a value of very large magnitude is projected
- **THEN** the result approaches the pole opposite the origin's image

#### Scenario: Inverse projection sends the pole to infinity

- **WHEN** the projection pole point is passed to `inverse_stereographic`
- **THEN** the result is complex infinity

#### Scenario: Projection direction is round-trippable

- **WHEN** a point is projected and then inverse-projected using the same `project_from_north` setting
- **THEN** the original complex value is recovered (away from the pole)

### Requirement: Smooth interpolation helpers

The library SHALL provide a logistic `sigmoid` mapping and a shortest-arc `circular_interpolate`
for blending scalar values and angles without discontinuities.

#### Scenario: Sigmoid is centered and bounded

- **WHEN** `sigmoid(x, center, scale)` is evaluated at `x == center`
- **THEN** the result is `0.5`, and all results lie strictly within the open interval `(0, 1)`

#### Scenario: Angle interpolation takes the shortest arc

- **WHEN** two angles straddling the `2π` wraparound are interpolated at `t = 0.5`
- **THEN** the result follows the shorter arc across the wrap, returned in `[0, 2π)`, rather than passing through the far side
