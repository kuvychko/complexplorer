## MODIFIED Requirements

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

## ADDED Requirements

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
