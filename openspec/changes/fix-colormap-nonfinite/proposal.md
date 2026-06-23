# Fix colormap non-finite RGB

## Why

`Colormap.rgb()` produces **invalid and non-deterministic** RGB when the input is non-finite
at an in-domain point (a pole or essential singularity that lands on a grid node).
`Colormap.hsv()` derives a NaN hue there and passes it to matplotlib's `hsv_to_rgb`, whose
`(h*6).astype(int)` casts NaN to a garbage int that varies run-to-run. The `outmask` path
only covers out-of-*domain* points, not in-domain singularities.

This violates the colormap conversion contract (all channels in `[0, 1]`) and has surfaced
three times: matplotlib 3D **raises** on it (band-aided by clipping facecolors in
`plotting/matplotlib/plot_3d.py`), PyVista renders garbage, and it makes pole-node pinning
tests flaky. It is also a **prerequisite for the deterministic gallery** (Phase 2): pole
presets would otherwise yield non-deterministic, garbage pixels.

## What changes

- **Sanitize non-finite inputs in the base `Colormap.hsv()`** (one fix, all 14 colormaps):
  replace non-finite `z` with a placeholder before `hsv_tuple()` (so subclass index math
  never sees NaN/inf), then paint those points — together with `outmask` — using the
  colormap's `out_of_domain_hsv`. The result: `rgb()` is **always finite, in `[0, 1]`, and
  deterministic** for any input.
- **Drop the matplotlib-3D facecolor band-aid** in `plotting/matplotlib/plot_3d.py` (the
  `np.clip(np.nan_to_num(...))` on facecolors), now redundant since the colormap guarantees
  valid RGB.

## Non-goals

- Not changing the *value-derived* color for finite inputs (unchanged for all existing
  presets/tests away from singularities).
- Not detecting singularities or changing domain masking — only sanitizing the color of
  non-finite points.

## Impact

- Touched: `complexplorer/core/colormap.py` (base `hsv()`); `plotting/matplotlib/plot_3d.py`
  (remove the band-aid).
- Affected specs: `colormaps` — the "Out-of-domain coloring" requirement extends to
  in-domain non-finite points; the `[0, 1]` contract now genuinely holds for any input.
- Unblocks: deterministic 2D gallery images (no garbage pole pixels); lets the surface-kernel
  pole-node regression cases be deterministic.
- Risk: low. Pure base-class sanitization; finite inputs are byte-unchanged (a test pins
  that). Verify the determinism with the previously-flaky pole case.
