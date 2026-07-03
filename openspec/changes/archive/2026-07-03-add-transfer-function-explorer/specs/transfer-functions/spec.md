# Transfer Functions — Delta for add-transfer-function-explorer

## ADDED Requirements

### Requirement: Rational transfer-function object

The library SHALL provide `complexplorer.ee.TransferFunction`, a rational function
`H = N/D` built from polynomial coefficient sequences (highest degree first) for
continuous-time (`system="s"`) or discrete-time (`system="z"`) systems. The object SHALL
expose `poles`, `zeros`, `is_stable` (all poles strictly left of the imaginary axis for `s`,
strictly inside the unit circle for `z`), and SHALL be callable on complex scalars and arrays
so it composes with every existing renderer as an ordinary complex function. Invalid inputs
(empty or non-1-D coefficients, identically zero denominator, unknown system) SHALL raise a
`ValidationError`.

#### Scenario: Poles and zeros match polynomial roots

- **WHEN** a `TransferFunction` is built from numerator and denominator coefficients
- **THEN** `zeros` equals the roots of the numerator and `poles` equals the roots of the denominator

#### Scenario: Callable evaluation

- **WHEN** the object is called on a complex array `s`
- **THEN** it returns `polyval(num, s) / polyval(den, s)` elementwise, so `cp.plot(domain, tf)` and the PyVista renderers accept it directly

#### Scenario: Stability verdict per system

- **WHEN** `is_stable` is read
- **THEN** it is true iff every pole has negative real part (`system="s"`) or modulus strictly less than one (`system="z"`)

#### Scenario: Invalid construction is rejected

- **WHEN** the numerator or denominator is empty or not one-dimensional, the denominator is identically zero, or `system` is not `"s"` or `"z"`
- **THEN** a `ValidationError` is raised

### Requirement: Frequency response along the stability contour

The transfer-function object SHALL evaluate its frequency response along the system's
canonical contour — `s = jω` for continuous systems, `z = e^{jω}` for discrete systems —
returning the frequency grid and complex response, with an automatic log-spaced frequency
range derived from the pole/zero magnitudes when none is given.

#### Scenario: Response values lie on the contour evaluation

- **WHEN** `frequency_response(omega)` is called with an explicit frequency array
- **THEN** the returned response equals the callable evaluated at `jω` (or `e^{jω}` for `system="z"`) for each frequency

#### Scenario: Automatic frequency range

- **WHEN** `frequency_response()` is called without frequencies
- **THEN** a log-spaced grid is chosen spanning at least a decade below the smallest and above the largest nonzero pole/zero magnitude

### Requirement: Canonical companion views

The library SHALL provide matplotlib views for a transfer function: a pole-zero plot (poles
`×`, zeros `○`, stability boundary drawn), a Bode plot (magnitude in dB and phase in degrees
over log frequency), a Nyquist plot (the `H(jω)` locus with the critical point `−1` marked),
and an annotated phase portrait (`transfer_portrait`) that renders `H` through the standard
2D domain-coloring path over a domain enclosing the poles and zeros, overlaying the pole/zero
markers and the stability boundary.

#### Scenario: Pole-zero plot shows structure and boundary

- **WHEN** `pole_zero_plot` is called for a transfer function
- **THEN** poles are drawn as `×`, zeros as `○`, and the stability boundary (imaginary axis for `s`, unit circle for `z`) is drawn, on an equal-aspect axes

#### Scenario: Bode panels

- **WHEN** `bode_plot` is called
- **THEN** a figure with magnitude (20·log₁₀|H|, dB) and phase (degrees) panels over a shared log-frequency axis is returned

#### Scenario: Nyquist locus

- **WHEN** `nyquist_plot` is called
- **THEN** the locus of `H` along the frequency contour is drawn for positive and negative frequencies and the point `−1 + 0j` is marked

#### Scenario: Annotated phase portrait composes the standard plot path

- **WHEN** `transfer_portrait` is called
- **THEN** the portrait is rendered via the standard 2D `plot()` (honoring its options, e.g. `legend=True`) over an automatically sized domain enclosing all poles and zeros, with pole/zero markers and the stability boundary overlaid
