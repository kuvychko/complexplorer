# Design — fix colormap non-finite RGB

## Root cause (verified 3×)

```
Colormap.rgb(z)
  └─ hsv(z) ─ hsv_tuple(z): hue = phase(z); for non-finite z, phase = NaN -> H = NaN
              (and pattern colormaps do np.floor(...).astype(int) on NaN -> garbage int)
  └─ mcolors.hsv_to_rgb(hsv): (h*6).astype(int) on NaN -> run-varying garbage int
                              -> non-deterministic, out-of-[0,1] RGB
```

The `outmask` path (`hsv()` lines ~77–83) only recolors *out-of-domain* points. In-domain
non-finite values (a pole/essential on a grid node) flow through unguarded.

## The fix — sanitize in the base `Colormap.hsv()`

One change covers all colormaps, because every subclass routes through `hsv_tuple()` →
`hsv()`:

```python
def hsv(self, z, outmask=None):
    z = np.asarray(z)
    bad = ~np.isfinite(z)                    # in-domain poles / essential singularities
    z_safe = np.where(bad, 0, z) if bad.any() else z   # placeholder so hsv_tuple sees no NaN
    H, S, V = self.hsv_tuple(z_safe)
    mask = bad
    if outmask is not None:
        mask = mask | outmask
    # paint non-finite AND out-of-domain points with the out-of-domain color
    if mask.any():  H,S,V = (paint(c, mask, self.out_of_domain_hsv[i]) ...)
    return stack(H, S, V)
```

Key points:
- **`z_safe` placeholder first.** Replacing non-finite `z` with `0` *before* `hsv_tuple()`
  means subclass index math (`Chessboard`/`PolarChessboard`/`LogRings` do
  `np.floor(...).astype(int)`) never sees NaN/inf and can't produce garbage or IndexErrors.
  The placeholder color is overwritten anyway.
- **Non-finite ≡ out-of-domain color.** A pole has "no representable value," so the neutral
  out-of-domain color is the natural, deterministic choice (consistent with masked points).
- **Scalar case** (`z.ndim == 0`) handled with scalar booleans.
- Result: `rgb()` is finite, in `[0, 1]`, and deterministic for *any* input — making the
  existing "RGB output is gamut-valid for any complex input" contract actually true.

## Removing the band-aid

`plotting/matplotlib/plot_3d.py` clips/`nan_to_num`s facecolors at three sites (added when
matplotlib 3D raised on the NaN colors). With the colormap guaranteeing valid RGB, those are
redundant and can be removed. (This is deprecated mpl-3D code, removed at 3.0; cleanup is
optional but tidy.)

## Determinism check

The pre-existing flakiness: pinning the RGB of a function with an exact in-domain pole node
varied across process runs. After this fix it is stable. The surface-kernel regression cases
that deliberately avoid pole nodes can stay as-is, but a focused test should assert the
previously-flaky case is now deterministic.

## Risk

| Risk | Mitigation |
|---|---|
| Finite-input color changes | `z_safe == z` where finite; a test pins finite output byte-unchanged for a sample |
| A subclass relied on NaN propagation | None do; all route through `hsv_tuple` and are overwritten at `bad` points |
| Band-aid removal changes mpl-3D output | Only at non-finite points, which now get a valid color either way |
