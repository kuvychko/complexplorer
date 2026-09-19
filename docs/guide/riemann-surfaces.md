# Riemann surfaces

A Riemann **surface** is not the same object as a Riemann **sphere**, and the two are easy to
confuse because both are 3D and both involve the word Riemann.

[![The Riemann sphere of 1/z beside the two-sheeted Riemann surface of the square root](../examples/gallery/view/_tour/sphere_vs_surface.png)](../examples/gallery/_tour/sphere_vs_surface.png)

- The **sphere** compactifies the plane, so that a single-valued function can include the point at
  infinity. One function, one value per point, one sphere.
- The **surface** is the multi-sheeted cover on which a *multivalued* expression becomes
  single-valued. `sqrt(z)` has two values at every non-zero point; on its two-sheeted surface it
  has exactly one.

If the question is "what does this function do near infinity", you want the sphere
([3D landscapes and the Riemann sphere](three-dimensions.md)). If the question is "how do the
branches of this expression fit together", you want the surface.

## Drawing one

```python
import complexplorer as cp

cp.riemann_surface_pv("power", n=2)                  # sqrt(z): two sheets
cp.riemann_surface_pv("power", n=3)                  # cube root: three sheets
cp.riemann_surface_pv("log", turns=3, r_max=3.0)     # the logarithm's helicoid
cp.riemann_surface_pv("algebraic", p=[1, 0, -1])     # w^2 = P(z)
```

Three families are available:

- **`power`** — the `n`-th root `z^(1/n)`, with `n` sheets joined at a branch point of order `n` at
  the origin.
- **`log`** — the logarithm, whose surface is an infinite helicoid; `turns` chooses how much of it
  to draw. Because the height grows with every turn, a taller surface wants a wider one:
  `r_max=3.0` keeps it from looking like a drinking straw.
- **`algebraic`** — curves `w² = P(z)`, with `p` giving the polynomial's coefficients in descending
  order, so `[1, 0, -1]` is `w² = z² − 1`.

The rendering arguments are the same as everywhere else in the 3D API: `resolution`, `cmap`,
`camera_position`, `window_size`, `interactive=False` with `filename` to write an image, and
`return_plotter=True` to keep composing.

## Reading one

The sheets are stacked in height and coloured by phase, so a branch point is where they meet. Walk
once around the branch point on the surface and you arrive on the *next* sheet, not back where you
started — which is the whole reason the surface exists. After `n` circuits of an order-`n` branch
point you return to the sheet you began on.

The branch *cut* you see in a flat portrait of `sqrt(z)` — the line where the colour jumps
discontinuously — is the seam where the flat picture was forced to choose one sheet. On the surface
the discontinuity is gone; only the choice of where to cut was ever arbitrary.

## Next

- [The gallery](../gallery/gallery.generated.md) shows the branch-cut presets (`sqrt`, `cbrt`,
  `log`) as flat portraits, which is the contrast that makes the surface worth the trouble.
