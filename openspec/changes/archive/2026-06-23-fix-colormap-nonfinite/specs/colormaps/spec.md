# Colormaps

## MODIFIED Requirements

### Requirement: Out-of-domain coloring

A colormap SHALL render points marked as outside the domain — and points whose input value
is **non-finite** (an in-domain pole or essential singularity, where `z` is infinite or NaN)
— with a distinct, neutral out-of-domain color rather than a computed one, so masked or
singular regions read as "no data." As a result, `rgb()` SHALL be **finite, within `[0, 1]`,
and deterministic for any input**, including non-finite values.

#### Scenario: Masked points use the out-of-domain color

- **WHEN** `rgb(z, outmask=mask)` is called with a boolean mask
- **THEN** points where the mask is true are colored with the colormap's out-of-domain color (a light gray by default) instead of the value-derived color

#### Scenario: Non-finite values use the out-of-domain color

- **WHEN** `rgb(z)` is called and some entries of `z` are non-finite (infinite or NaN, e.g.
  a pole on a grid node)
- **THEN** those entries are colored with the out-of-domain color, and the resulting RGB is
  finite and within `[0, 1]`

#### Scenario: Non-finite coloring is deterministic

- **WHEN** `rgb(z)` is evaluated more than once for the same `z` containing non-finite
  entries
- **THEN** the result is identical each time (no run-varying values from a NaN-to-int cast)
