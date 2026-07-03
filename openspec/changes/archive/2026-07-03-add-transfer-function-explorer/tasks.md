# Tasks — add-transfer-function-explorer

## 1. Math kernel

- [x] 1.1 `complexplorer/ee/transfer_function.py`: `TransferFunction` class — validation,
      `poles`/`zeros`/`gain`, `is_stable` (strict; per system), `__call__` via `polyval`
      ratio, `frequency_response(omega=None)` with auto log range and the `s`/`z` contour
- [x] 1.2 `complexplorer/ee/__init__.py` re-exports; expose `ee` from `complexplorer/__init__.py`

## 2. Views

- [x] 2.1 `pole_zero_plot(tf, ax=None)`: × poles, ○ zeros, stability boundary, equal aspect
- [x] 2.2 `bode_plot(tf, omega=None)`: dB magnitude + degree phase panels, shared semilogx axis
- [x] 2.3 `nyquist_plot(tf, omega=None, ax=None)`: ± frequency locus, −1 marked
- [x] 2.4 `transfer_portrait(tf, domain=None, **plot_kwargs)`: auto domain enclosing
      poles/zeros, standard `plot()` path, marker + boundary overlay

## 3. Tests and docs

- [x] 3.1 Unit tests `tests/unit/ee/test_transfer_function.py`: roots/poles/zeros, callable
      evaluation vs polyval, stability (s and z, marginal cases), frequency_response vs
      `scipy.signal.freqs`/`freqz` for a Butterworth-like example, validation errors, view
      smoke tests (figure/axes structure), transfer_portrait callable through `cp.plot`
- [x] 3.2 `CHANGELOG.md` Added entry
- [x] 3.3 Off-screen visual check: portrait + Bode of `1/(s² + 0.2s + 1)` (resonator)

## 4. Verification

- [x] 4.1 `pytest tests/` green; ruff clean; `openspec validate --specs` passes
