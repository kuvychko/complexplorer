## MODIFIED Requirements

### Requirement: Colormap parameter validation

A colormap SHALL reject out-of-range configuration parameters at construction time. An invalid
pattern parameter SHALL raise `ColormapError`. Because `ColormapError` is a `ValidationError`
(and a `ValueError`), existing validation handlers keep catching it.

This SHALL include `phase_sectors` on every phase portrait, not only on the pattern colormaps: it
SHALL be a positive integer, and any other value SHALL raise at construction with a message naming
the value received and what is accepted. No configuration reachable through the public API SHALL
surface as a bare Python exception such as `ZeroDivisionError`.

#### Scenario: Invalid brightness floor is rejected

- **WHEN** a phase portrait is constructed with `v_base` outside `[0, 1)`
- **THEN** a `ValidationError` is raised

#### Scenario: Non-positive pattern spacing is rejected

- **WHEN** a `Chessboard`, `PolarChessboard`, or `LogRings` is constructed with a non-positive spacing, sector count, or ring base
- **THEN** a `ColormapError` is raised, and `except ValidationError` also catches it

#### Scenario: An invalid phase-sector count is rejected at construction

- **WHEN** any phase portrait, including the perceptual families, is constructed with a
  `phase_sectors` that is zero, negative, or not an integer
- **THEN** a `ValidationError` is raised naming the value and the accepted values, and no
  `ZeroDivisionError` occurs
