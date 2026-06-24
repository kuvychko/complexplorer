# Colormaps

## Purpose

The colormaps capability turns complex values into colors. It defines the colormap contract
(HSV and RGB conversion with out-of-domain handling), the enhanced phase-portrait base shared by
the perceptual colormaps, and a family of concrete colormaps ranging from classic phase portraits
and grayscale grid/ring patterns to perceptually-uniform schemes built on OKLAB/OkLCh. Colormaps
are the visual encoding layer consumed by every 2D and 3D plotting capability.

## Requirements

### Requirement: Colormap conversion contract

Every colormap SHALL convert an array of complex values into color, exposing both HSV and RGB
forms, with each color channel in the range `[0, 1]` and the output shape matching the input with
a trailing length-3 color axis.

#### Scenario: RGB output is gamut-valid

- **WHEN** a colormap's `rgb(z)` is called for any complex input
- **THEN** the result has shape `(*z.shape, 3)` with all channel values in `[0, 1]`

#### Scenario: HSV and RGB are consistent

- **WHEN** `hsv(z)` and `rgb(z)` are both evaluated
- **THEN** `rgb` equals the HSV-to-RGB conversion of `hsv`

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

### Requirement: Enhanced phase portrait modulation

Phase-portrait colormaps SHALL encode phase as hue and MAY overlay phase-sector and modulus
contour bands that raise brightness toward a configured floor, so both argument and magnitude
structure are visible at once.

#### Scenario: Phase maps to hue

- **WHEN** a basic phase portrait renders a value
- **THEN** the hue is the value's phase divided by `2π`

#### Scenario: Phase sectors create angular bands

- **WHEN** a phase portrait is configured with `phase_sectors = N`
- **THEN** brightness is modulated by a sawtooth across `N` angular sectors

#### Scenario: Modulus contours create rings

- **WHEN** a linear or logarithmic modulus step is configured
- **THEN** brightness is modulated by a sawtooth across concentric magnitude rings

#### Scenario: Brightness floor is respected

- **WHEN** modulations combine into the final brightness with a base value `v_base`
- **THEN** the result lies in `[v_base, 1]`, never going fully dark

### Requirement: Auto-scaled square cells

A phase portrait SHALL support optionally auto-computing its modulus ring spacing from the number
of phase sectors so that phase and modulus bands form approximately square cells.

#### Scenario: Auto-scaling derives the ring step

- **WHEN** `auto_scale_r` is enabled with `phase_sectors = N` and a `scale_radius`
- **THEN** the linear ring step is set to `(2π / N) * scale_radius`

#### Scenario: Auto-scaling requires phase sectors

- **WHEN** `auto_scale_r` is enabled but no `phase_sectors` is given, or an explicit ring step is also given
- **THEN** a `ValidationError` is raised

### Requirement: Colormap parameter validation

A colormap SHALL reject out-of-range configuration parameters at construction time.

#### Scenario: Invalid brightness floor is rejected

- **WHEN** a phase portrait is constructed with `v_base` outside `[0, 1)`
- **THEN** a `ValidationError` is raised

#### Scenario: Non-positive pattern spacing is rejected

- **WHEN** a `Chessboard`, `PolarChessboard`, or `LogRings` is constructed with a non-positive spacing, sector count, or ring base
- **THEN** a `ColormapError` is raised

### Requirement: Grayscale structural colormaps

The library SHALL provide non-hued black/white colormaps that reveal grid and ring structure
independent of phase: a Cartesian `Chessboard`, a polar `PolarChessboard`, and logarithmic
`LogRings`.

#### Scenario: Cartesian chessboard alternates by cell parity

- **WHEN** `Chessboard` colors a point
- **THEN** the cell is white or black according to the parity of its row and column indices, with no hue or saturation

#### Scenario: Polar chessboard alternates by sector and ring parity

- **WHEN** `PolarChessboard` colors a point
- **THEN** the cell is white or black according to the combined parity of its angular sector and (linear or logarithmic) radial ring

#### Scenario: Log rings alternate by logarithmic radius

- **WHEN** `LogRings` colors a point
- **THEN** the ring is white or black according to the parity of `floor(log|z| / spacing)`, with the origin colored white

### Requirement: Perceptual colormap family

The library SHALL provide perceptually-motivated phase-portrait colormaps built on OKLAB/OkLCh or
cubehelix, each with a distinct visual intent while honoring the shared phase-portrait contract.

#### Scenario: Perceptual colormaps are selectable and self-validating

- **WHEN** any of `OklabPhase`, `PerceptualPastel`, `AnalogousWedge`, `DivergingWarmCool`, `Isoluminant`, `CubehelixPhase`, `InkPaper`, `EarthTopographic`, or `FourQuadrant` is constructed with valid parameters and used to color values
- **THEN** it produces gamut-valid RGB encoding phase (and, where configured, modulus) per its design intent
- **AND** constructing it with a lightness or chroma parameter outside its allowed range raises a `ValidationError`

#### Scenario: Smooth versus enhanced rendering

- **WHEN** a perceptual colormap that supports a smooth mode is used without enhancement
- **THEN** it produces continuous (cplot-like) color; **WHEN** enhancement is enabled it adds sawtooth contour structure
