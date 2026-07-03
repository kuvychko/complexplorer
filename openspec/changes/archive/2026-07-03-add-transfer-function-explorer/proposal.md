# Add Transfer-Function Explorer (cp.ee)

## Why

Engineering users (EE, controls, signal processing) reason about rational transfer functions
`H(s) = N(s)/D(s)` — poles, zeros, stability, frequency response — and complexplorer's whole
premise ("make complex-valued structure visible") applies directly: a phase portrait of `H`
shows poles, zeros, resonances, and notches as geometry. This was Phase 5 of the vision plan,
deferred to 3.1+ on 2026-07-03 and pulled back into 3.0.0 by owner decision the same day. The
design center that keeps it a *core library improvement* rather than an EE demo:
`TransferFunction` is a plain complex **callable**, so every existing renderer (2D portraits,
PyVista landscapes, Riemann sphere, STL ornaments) works on it unchanged — the new code is one
small math class plus the three canonical EE companion views.

## What Changes

- New subpackage `complexplorer/ee/` with `transfer_function.py`:
  - `TransferFunction(num, den, system="s")` — rational function from polynomial coefficients
    (`numpy.polyval` order). Continuous-time (`"s"`, stability boundary = imaginary axis) and
    discrete-time (`"z"`, stability boundary = unit circle) systems.
  - Pure math, PyVista-free: `poles`, `zeros` (via `numpy.roots`), `is_stable`,
    `frequency_response(omega=None)` (evaluation along the system's frequency contour with an
    auto log-spaced range derived from pole/zero magnitudes), and `__call__(s)` so the object
    is a complex callable usable with `cp.plot`, `cp.plot_landscape_pv`, `cp.riemann_pv`,
    `cp.quick_plot`, and STL export directly.
  - Validation (`ValidationError`): each coefficient array non-empty and 1-D, denominator not
    identically zero, `system` in `{"s", "z"}`.
- Three matplotlib companion views in the same module:
  - `pole_zero_plot(tf)` — poles ×, zeros ○, stability boundary drawn (imaginary axis or unit
    circle), equal aspect.
  - `bode_plot(tf)` — magnitude (dB) and phase (degrees) panels over log frequency.
  - `nyquist_plot(tf)` — the `H(jω)` locus for `ω ∈ [−ω_max, ω_max]` with the critical point
    `−1 + 0j` marked.
  - `transfer_portrait(tf)` — a phase portrait of `H` over an auto-sized domain enclosing the
    poles/zeros, with pole/zero markers and the stability boundary overlaid (composes the
    existing `plot()`; forwards `legend=` etc.).
- Namespacing: everything lives under `cp.ee` (`from complexplorer import ee` /
  `cp.ee.TransferFunction`). Only the `ee` module itself is added to the top-level package —
  the core namespace stays math-generic, per the roadmap's audience-mode boundary.
- Out of scope (remains 3.1+): filter design families (Butterworth/Chebyshev/…), resonators,
  QCM, impedance/RF/Smith-chart features, time-domain responses.

## Capabilities

### New Capabilities

- `transfer-functions`: the rational transfer-function object (poles/zeros/stability/
  frequency response, callable evaluation) and its canonical views (pole-zero, Bode, Nyquist,
  annotated phase portrait).

### Modified Capabilities

_None._

## Impact

- New: `complexplorer/ee/__init__.py`, `complexplorer/ee/transfer_function.py`,
  `openspec/specs/transfer-functions/spec.md`, `tests/unit/ee/test_transfer_function.py`.
- Modified: `complexplorer/__init__.py` (expose the `ee` subpackage), `CHANGELOG.md`.
- Dependencies: none added (numpy + matplotlib only; scipy used in tests as an oracle for
  frequency-response spot checks).
- ROADMAP: move `add-transfer-function-explorer` from the 3.1+ backlog into the 3.0 block
  (done in the docs-drift change).
