# Tasks — fix-colormap-nonfinite

## 1. Sanitize non-finite inputs in the base colormap
- [x] 1.1 In `Colormap.hsv()` (`core/colormap.py`), compute `bad = ~np.isfinite(z)`, evaluate
      `hsv_tuple()` on a placeholder-substituted `z` (non-finite → 0), then paint the `bad`
      points (combined with `outmask`) using `out_of_domain_hsv`.
- [x] 1.2 Handle the scalar (`z.ndim == 0`) case.
- [x] 1.3 Confirm `rgb()` is finite, in `[0, 1]`, and unchanged for finite inputs.

## 2. Remove the matplotlib-3D facecolor band-aid
- [x] 2.1 Remove the `np.clip(np.nan_to_num(...))` facecolor sanitization at the three sites
      in `plotting/matplotlib/plot_3d.py` (now redundant — the colormap guarantees valid RGB).

## 3. Tests
- [x] 3.1 `rgb()` on input with non-finite entries (e.g. `1/z` at the origin, `exp(1/z)` at 0):
      result is finite, within `[0, 1]`, and equals the out-of-domain color at those points.
- [x] 3.2 Determinism: `rgb(z)` on a non-finite-containing array is identical across repeated
      calls (the previously-flaky pole-node case).
- [x] 3.3 Regression: finite-input RGB is byte-unchanged for a sample (a couple of colormaps).
- [x] 3.4 Cover a pattern colormap (`Chessboard`/`PolarChessboard`/`LogRings`) with a pole to
      confirm no garbage/IndexError from the `astype(int)` index math.

## 4. Close out
- [x] 4.1 Run `pytest tests/` green; `ruff` clean; `openspec validate --specs`.
- [x] 4.2 Update `openspec/ROADMAP.md` (the bug is fixed; gallery is unblocked).
