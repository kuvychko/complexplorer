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
