# Modulus Scaling

## Purpose

The modulus-scaling capability maps function magnitude `|f(z)|` to a radius or height used by 3D
landscapes, Riemann sphere relief, and STL ornaments. Because `|f(z)|` can range from zero to
infinity (poles), this capability provides a menu of transfer functions — bounded, unbounded,
data-adaptive, and custom — plus named presets, so users can emphasize different features (poles,
fine near-zero detail, overall balance) without rewriting math.

## Requirements

### Requirement: Modulus scaling modes

The library SHALL provide a set of named scaling modes that transform an array of moduli into
radii, each preserving input shape and documenting its output range.

#### Scenario: Constant mode ignores magnitude

- **WHEN** `constant` scaling is applied
- **THEN** every radius equals the configured constant regardless of modulus

#### Scenario: Linear mode grows without bound

- **WHEN** `linear` scaling is applied
- **THEN** the radius is `1 + scale * |f|`, increasing without an upper limit

#### Scenario: Arctan mode saturates smoothly

- **WHEN** `arctan` scaling is applied
- **THEN** radii map `[0, ∞)` smoothly into `[r_min, r_max]`, approaching `r_max` as magnitude grows

#### Scenario: Logarithmic mode compresses exponential growth

- **WHEN** `logarithmic` scaling is applied
- **THEN** the modulus is log-transformed (guarded against `log(0)`) and mapped via a sigmoid into `[r_min, r_max]`

#### Scenario: Linear-clamp mode caps at a threshold

- **WHEN** `linear_clamp` scaling is applied
- **THEN** the radius grows linearly up to `m_max` and is held at `r_max` beyond it

#### Scenario: Power mode normalizes by the maximum

- **WHEN** `power` scaling is applied
- **THEN** moduli are normalized by their maximum, raised to the configured exponent, and mapped into `[r_min, r_max]`

#### Scenario: Sigmoid mode gives a tunable S-curve

- **WHEN** `sigmoid` scaling is applied
- **THEN** radii follow an S-curve centered at `center` with the configured steepness, bounded in `[r_min, r_max]`

#### Scenario: Adaptive mode is robust to outliers

- **WHEN** `adaptive` scaling is applied
- **THEN** the low and high percentiles of the finite moduli map to `r_min` and `r_max`, ignoring infinities and NaNs
- **AND** when the percentile band is degenerate, a mid-range radius is returned for all points

#### Scenario: Hybrid mode blends linear and logarithmic regions

- **WHEN** `hybrid` scaling is applied
- **THEN** magnitudes below the transition scale linearly and those above scale logarithmically, joined continuously at the transition

#### Scenario: Custom mode applies a user function

- **WHEN** `custom` scaling is applied with a user-supplied function
- **THEN** the function's output is clipped to `[0, 1]` and mapped into `[r_min, r_max]`

### Requirement: Named scaling presets

The library SHALL provide named presets that resolve to a scaling mode and parameter set tuned for
a common visualization goal.

#### Scenario: A known preset resolves to a configuration

- **WHEN** a preset such as `balanced`, `detail_near_zero`, `auto`, `high_contrast`, or `poles_emphasis` is requested
- **THEN** a configuration naming the scaling method and its parameters is returned

#### Scenario: An unknown preset is rejected

- **WHEN** a preset name that does not exist is requested
- **THEN** an error is raised listing the available presets

### Requirement: Visualization and print parameter defaults

The library SHALL supply default parameters for each scaling mode, and MAY differ between
on-screen visualization and STL export so that printed ornaments use print-appropriate radius
ranges.

#### Scenario: Defaults are provided per mode and target

- **WHEN** default parameters are requested for a scaling mode for a given target (visualization or STL)
- **THEN** a parameter set appropriate to that mode and target is returned

### Requirement: Custom scaling honours the radius bounds

A custom scaling callable maps moduli into `[0, 1]`, which the capability then maps onto
`[r_min, r_max]`. Dispatching a custom mode SHALL route through the same custom scaling method that
documents this contract, so the callable's output is clipped to `[0, 1]` and mapped onto the
requested bounds like every other mode.

Dispatch SHALL NOT return the callable's raw output while silently discarding `r_min` and `r_max`
supplied alongside it. A caller who follows the documented contract and returns values in `[0, 1]`
would otherwise obtain radii in `[0, 1]` rather than in the requested range, collapsing the relief
toward the origin.

#### Scenario: Bounds supplied with a custom callable are applied

- **WHEN** a custom scaling mode is applied with a callable and explicit radius bounds
- **THEN** the callable's output is clipped to `[0, 1]` and mapped onto those bounds

#### Scenario: A missing callable is still rejected

- **WHEN** a custom scaling mode is applied without a scaling callable
- **THEN** a validation error is raised naming the missing parameter

### Requirement: Normalization constant for a rational function

The capability SHALL expose a helper computing the constant that places the area-weighted geometric
mean of `|f|` over the Riemann sphere at 1, for a rational function given by its finite zeros,
finite poles and leading gain. The constant is the reciprocal of the gain magnitude, times the
product over finite poles of `sqrt(1 + |p|^2)`, divided by the product over finite zeros of
`sqrt(1 + |z|^2)`.

The helper SHALL be documented as valid only when the divisor is **complete** — every finite zero
and pole of a rational function — and SHALL NOT be presented as applicable to a transcendental
function or to a partial list of singularities, for which it returns a confidently wrong number.

#### Scenario: The constant matches quadrature

- **WHEN** the constant is computed for a rational function from its complete divisor
- **THEN** it agrees with an area-weighted numerical estimate of the same quantity

#### Scenario: The constant is self-dual

- **WHEN** the constant is computed for a function and for its reciprocal, exchanging zeros and
  poles
- **THEN** the two results are reciprocal

#### Scenario: A reciprocal-symmetric divisor normalizes to unity

- **WHEN** the constant is computed for a function whose zero and pole divisors are
  reciprocal-symmetric
- **THEN** it is 1

### Requirement: Sampled normalization is area-weighted and seam-free

An estimate of the normalization constant taken from a latitude/longitude sphere grid SHALL weight
each sample by `sin(theta)` and SHALL exclude the duplicated seam meridian that a longitude grid
spanning `0` to `2*pi` inclusive produces by including both endpoints.

Both corrections are required for correctness, not precision. Without area weighting the grid
crowds the poles and the estimate is wrong by an order of magnitude for a function with a
high-order feature at infinity. With the seam counted twice, a function whose zeros or poles lie on
the positive real axis — where the seam lies, and where a real-coefficient function places its
features — is biased by around 1%, two orders of magnitude worse than the same grid with the seam
removed.

Samples at which `log|f|` is not finite SHALL be excluded from the weighted average.

#### Scenario: Polar crowding is corrected

- **WHEN** the constant is estimated for a function with a high-order zero or pole at infinity
- **THEN** the area-weighted estimate agrees with the closed form, where an unweighted average over
  the same grid does not

#### Scenario: A feature on the seam is not double-counted

- **WHEN** the constant is estimated for a function whose zeros and poles lie on the positive real
  axis
- **THEN** the estimate agrees with the closed form to the same accuracy as for a function whose
  features lie off that axis

#### Scenario: Singular samples are ignored

- **WHEN** the grid contains samples at which `|f|` is zero or infinite
- **THEN** those samples are excluded and the remaining weighted average is returned
