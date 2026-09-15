# Colormaps

## Purpose

The colormaps capability turns complex values into colors. It defines the colormap contract
(HSV and RGB conversion with out-of-domain handling), the enhanced phase-portrait base shared by
the perceptual colormaps (`BasePhasePortrait`), and 13 concrete colormaps ranging from classic
phase portraits and grayscale grid/ring patterns to perceptually-uniform schemes built on
OKLAB/OkLCh. Colormaps
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

### Requirement: Phase-sector count is named phase_sectors

Every colormap that divides the phase circle into sectors SHALL take the sector count as
`phase_sectors`. This covers `Phase`, `PolarChessboard`, and each perceptual phase-portrait
colormap. The former name `n_phi` SHALL NOT be accepted: passing `n_phi` to a colormap
constructor, or using an `n_phi` key in a colormap spec, SHALL raise a `ValidationError` whose
message names `phase_sectors` as the replacement.

#### Scenario: phase_sectors configures sectors

- **WHEN** `Phase(phase_sectors=6)` or `PolarChessboard(phase_sectors=6)` is constructed
- **THEN** the colormap uses six phase sectors, and `Phase(phase_sectors=6, auto_scale_r=True)` derives its ring step from `phase_sectors`

#### Scenario: The former keyword is rejected with guidance

- **WHEN** `Phase(n_phi=6)` or `PolarChessboard(n_phi=6)` is called
- **THEN** a `ValidationError` is raised whose message names `phase_sectors` as the replacement

#### Scenario: A colormap spec using the former key is rejected

- **WHEN** a colormap is built from a spec dict (e.g. via `cmap_from_spec` or a preset) that carries an `n_phi` key
- **THEN** a `ValidationError` naming `phase_sectors` is raised rather than the key being silently ignored

### Requirement: Shared enhanced phase-portrait base

Hue-based phase-portrait colormaps (`Phase` and every perceptual phase-portrait colormap) SHALL
derive from a common `BasePhasePortrait`. The base owns the shared enhancements: phase-sector
modulation, modulus-ring modulation, auto-scaled square cells, and brightness-floor validation.
Every enhanced phase colormap therefore accepts the same enhancement parameters with the same
meaning. Rebasing `Phase` onto this base SHALL NOT change its output.

#### Scenario: Phase colormaps share the base

- **WHEN** `Phase` or any perceptual phase-portrait colormap is instantiated
- **THEN** it is an instance of `BasePhasePortrait`

#### Scenario: Rebasing preserves Phase output

- **WHEN** `Phase` is evaluated with any combination of `phase_sectors`, `r_linear_step`, `r_log_base`, `v_base`, `auto_scale_r`, and `scale_radius`, with unit-circle emphasis disabled
- **THEN** `rgb(z)` is identical to the reference output recorded before the rebase, for the same inputs

### Requirement: Optional unit-circle emphasis

`Phase` SHALL support an opt-in emphasis of the unit circle `|z| = 1`. With
`emphasize_unit_circle=True`, a narrow band around `|z| = 1` is either brightened or, when a
`unit_circle_color` is given, blended toward that color, weighted by `unit_circle_strength` in
`[0, 1]`. Emphasis SHALL be off by default.

#### Scenario: Emphasis is off by default

- **WHEN** `Phase` is constructed without `emphasize_unit_circle`
- **THEN** its output is unchanged by the emphasis feature

#### Scenario: Emphasis changes only the band near the unit circle

- **WHEN** `Phase(emphasize_unit_circle=True)` colors points on and far from `|z| = 1`
- **THEN** points near `|z| = 1` differ from the unemphasized rendering while points far from it are unchanged

#### Scenario: Out-of-range strength is rejected

- **WHEN** `unit_circle_strength` is outside `[0, 1]`
- **THEN** a `ValidationError` is raised

### Requirement: Colormap parameter validation

A colormap SHALL reject out-of-range configuration parameters at construction time. An invalid
pattern parameter SHALL raise `ColormapError`. Because `ColormapError` is a `ValidationError`
(and a `ValueError`), existing validation handlers keep catching it.

#### Scenario: Invalid brightness floor is rejected

- **WHEN** a phase portrait is constructed with `v_base` outside `[0, 1)`
- **THEN** a `ValidationError` is raised

#### Scenario: Non-positive pattern spacing is rejected

- **WHEN** a `Chessboard`, `PolarChessboard`, or `LogRings` is constructed with a non-positive spacing, sector count, or ring base
- **THEN** a `ColormapError` is raised, and `except ValidationError` also catches it


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
cubehelix. Each has a distinct visual intent and honors the shared phase-portrait contract. Each
SHALL be exported from the top-level `complexplorer` package, SHALL honor the out-of-domain and
non-finite coloring contract, and SHALL render the phase-wheel legend through its own RGB
pipeline, like any other colormap.

#### Scenario: Perceptual colormaps are selectable and self-validating

- **WHEN** any of `OklabPhase`, `PerceptualPastel`, `AnalogousWedge`, `DivergingWarmCool`, `Isoluminant`, `CubehelixPhase`, `InkPaper`, `EarthTopographic`, or `FourQuadrant` is constructed with valid parameters and used to color values
- **THEN** it produces gamut-valid RGB encoding phase (and, where configured, modulus) per its design intent
- **AND** constructing it with a lightness or chroma parameter outside its allowed range raises a `ValidationError`

#### Scenario: Smooth versus enhanced rendering

- **WHEN** a perceptual colormap that supports a smooth mode is used without enhancement
- **THEN** it produces continuous (cplot-like) color; **WHEN** enhancement is enabled it adds sawtooth contour structure

#### Scenario: Perceptual colormaps honor the non-finite contract

- **WHEN** any perceptual colormap's `rgb(z)` is called with `z` containing infinite or NaN entries, or with an `outmask`
- **THEN** those entries receive the out-of-domain color and the result is finite, within `[0, 1]`, and identical across repeated calls

#### Scenario: Perceptual colormaps are top-level exports

- **WHEN** `complexplorer.__all__` is inspected
- **THEN** it lists every perceptual colormap named above, and each is importable as `complexplorer.<Name>`
