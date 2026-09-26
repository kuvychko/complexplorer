# Design

## Why the constant is sampled rather than derived

The closed form is exact for a rational `f`:

```
c = (1 / |A|) * prod_k sqrt(1 + |p_k|^2) / prod_j sqrt(1 + |z_j|^2)
```

over the **finite** zeros `z_j` and poles `p_k`, because `log` of the chordal distance to a fixed
point integrates to `-1/2` over the sphere. Checked against quadrature:

| function | quadrature | closed form |
|---|---|---|
| `z/(z^5-1)` | 5.65686 | 5.65685 |
| `z/(z^10-1)` | 32.00007 | 32.00000 |
| `(z-1)/(z+1)` | 1.00000 | 1.00000 |
| `z^2` | 1.00000 | 1.00000 |
| `3/z` | 0.33333 | 0.33333 |

It is nonetheless the wrong engine for the default path, for two reasons.

**It needs a complete divisor, and nothing in the library has one.** `create_ornament` takes an
arbitrary callable. The tempting shortcut — read the divisor off `FunctionPreset.singularities` —
is unsound, because those answer keys are illustrative. `sin(z)` lists three zeros and has
infinitely many:

| function | true (quadrature) | closed form from the answer key |
|---|---|---|
| `sin(z)` | 0.853 | 0.092 |
| `tan(z)` | 1.313 | 1.052 |

A "kinds are all zero/pole" test does not catch this — `sine` and `tangent` both pass it. Guarding
it properly would mean a completeness flag on every preset and a rationality check on every
callable, to buy an accuracy nobody can see.

**Sampling is accurate enough and free.** `generate_ornament` already evaluates `f` across the
sphere; the constant comes from that existing field with no second pass. Worst observed error is
0.06%, which moves a radius by less than a thousandth of the piece.

The closed form therefore ships as `normalization_constant(zeros, poles, gain)` — useful to a
caller who *does* know their divisor, and, more importantly, the exact oracle the sampled
estimator is tested against.

## The estimator's two traps

Both were measured, and both are one line.

**Area weighting is mandatory.** `sample_sphere` returns a lat/long grid, which crowds the poles.
Averaging `log|f|` unweighted over those samples gets `z/(z^10-1)` **10.1x** wrong. Weight by
`sin(theta)`.

**The duplicated seam meridian must be dropped.** `sphere_coordinates` builds
`phi = linspace(0, 2*pi, resolution)`, whose first and last columns are the same meridian, so that
meridian is counted twice. It lies along the **positive real axis**, exactly where a
real-coefficient function puts its features:

| case | as-is | seam dropped |
|---|---|---|
| feature on the seam, `(z-1)/(z+1)` | 1.05% error | 0.01% |
| feature off the seam, `(z-1j)/(z+1j)` | 0.00% | 0.00% |

A third effect is worth knowing about but not worth fixing: `avoid_poles` clamps `theta` to
`[0.01, pi - 0.01]`, a *fixed* clamp that does not shrink with resolution, so the estimate
converges to a slightly biased value rather than to the true one — error on `z/(z^10-1)` drifts
0.041% -> 0.057% -> 0.085% as resolution goes 100 -> 200 -> 400. It is bounded well below
anything visible. **The consequence for tests is that a sampled-vs-closed-form assertion must use a
loose tolerance (0.5% is comfortable) and must not tighten it with resolution.**

## Where normalization applies

The constant multiplies `|f|` before the transfer, inside the relief build. The `magnitude` scalar
attached to the mesh stays the **raw** `|f|`, because that is the mathematical quantity a viewer
inspects; the applied constant is recorded in mesh metadata instead. Only `radius` reflects
normalization, which is correct — normalization is a statement about geometry, not about the
function.

Scope is the STL/ornament path only. `riemann_pv` and the other relief renderers keep their
current behaviour, so their regression baselines stand. The helper is public, so a visualization
caller can opt in.

## Why `pointiness` forces the default transfer to change

With `r = r_min + delta * sigma(log|f| / k)`, the surface approaches a feature of order `mu` like
`distance ** (mu / k)`. So `mu / k` is the number the eye reads:

```
  mu/k > 1   rounded dome, reads as a blob
  mu/k = 1   exact cone
  mu/k < 1   cusp, sharper than a cone
```

`arctan` has no `k` at all. Its shape reaches 90% of its height at `|f| = 6.3` whatever the
function, which on typical rational functions sits well above `mu/k = 1` — which is why every
default ornament is a rounded pebble and why the gallery pieces all look related regardless of
their mathematics. A knob cannot be bolted onto it.

`ModulusScaling.logarithmic` is already `r_min + delta * sigma(L / ln(base))`, i.e. exactly the
right shape with `k = ln(base)`. `pointiness` is a re-parameterization of an existing mode, not new
mathematics: `k = pointiness * pole_order`, tip exponent `1 / pointiness`. Scaling by `pole_order`
(default 1, caller-supplied) means a double pole prints as sharp as a simple one instead of twice
as blunt.

## Resolved: `pointiness` defaults to 2.0, with `contrast` alongside it

An earlier reading of this change treated the default as an open calibration question, because a
bulk-contrast measure — share of surface area in the middle half of the radial range — makes
`k = 2.0` look worse than the current `arctan` default on five of six functions, while `k` around
0.5 to 0.75 looks better on all six.

**That measure is the wrong proxy, and the reference implementation settles it.** A ten-piece
collection was built at `POINTINESS = 2.0` (`figures_repo/2026/09-20-riemann-ornaments`), rendered
and visually inspected, and is now being printed. The measure fails because it rewards spreading
area across the radial range, which a low `k` achieves by blunting every tip — a well-contoured
blob scores better than a sculpted star. The notes use **feature clearance** instead (how far a
petal rises above the valley floor between adjacent petals), which is what the eye actually reads:
by that measure the pole flowers go from 16% to 47%.

The bulk measure is not worthless — it is what motivated `contrast` — but it must not be optimized
alone.

**`pointiness` and `contrast` are a matched pair, and shipping one without the other is a
mistake.** In the reference collection, 6 of 10 pieces set `contrast=(boost, weight)`:

| piece | contrast |
|---|---|
| quadrupole, pole-flower-5, pole-flower-10 | `(3.0, 0.5)` |
| alternating-crown-6, tetrahedral-dual, notch-filter | `(4.0, 0.6)` |
| dipole, octahedral-crown, cube-octahedron-dual, resonant-filter | none |

The four without it are exactly the cases the notes described as regressing under a *global*
mixture: the two Platonic pieces whose tips are the whole point, the gain-calibrated filter, and
the dipole, which is an egg at every setting. So "it improved four and regressed three" was never
an argument against the parameter — it was an argument against applying it globally. As a per-piece
parameter it is used by the majority of the collection.

Measured: with `k = 2.0`, the mixture recovers most of the bulk contrast it costs.

| piece | `arctan` | `k=2` alone | `k=2` + contrast |
|---|---|---|---|
| quadrupole | 83.2% | 97.3% | 88.4% |
| pole-flower-5 | 45.2% | 77.8% | 58.3% |
| pole-flower-10 | 40.8% | 64.1% | 51.0% |
| alternating-crown-6 | 94.1% | 98.9% | 92.7% |

Raising `pointiness` without a remedy for the bulk it flattens would hand users a knob whose only
direction of travel makes the body blander. `contrast` is that remedy.

## `sharpness` as a direct override

Two pieces in the collection bypass `pointiness` and set the log-modulus scale directly, to
`ln(10)`, so that one unit of relief is exactly a factor of ten in gain — 20 dB. That turns a
transfer function's relief into a readable gain scale, where walking a meridian reads off the Bode
magnitude curve. It is a genuine use case that `pointiness * pole_order` cannot express, so the
scale must be settable directly as well as through `pointiness`.

Note also that `ln(10) = 2.30` sits close to `POINTINESS * 1 = 2.0`, so those two pieces match the
rest of the collection's sharpness while still reading as a calibrated scale — the convenience
that made the choice free.

## Fixing `custom` dispatch is a breaking change

The reference implementation's `mixture_scaling` returns a **radius**, not a value in `[0, 1]`,
and says so explicitly: it bakes `depth` into its own return because `apply_scaling_mode`
short-circuits `ModulusScaling.custom` and uses the callable's output as the radius directly.

Anyone who worked around the bug did the same thing, because it was the only thing that worked.
After the fix their callable's output is clipped to `[0, 1]` — which a radius-returning callable
already satisfies — and then remapped onto `[r_min, r_max]`, whose default is `[0.5, 1.5]`. The
result is silently wrong geometry rather than an error. The migration note must call this out
directly, not bury it in a list of fixes.

## Consequences accepted

- Every default ornament changes shape. That is the release.
- `tests/regression/baselines/ornament.npz` is regenerated deliberately, not repaired.
- `validate_printability` loses a key (`wall_thickness_ok`, `estimated_min_wall_mm`) and gains two
  (`min_radius_mm`, `max_radius_mm`). The removed key was always `False`, so nothing could have
  depended on it being meaningful — but a caller reading it by name will see a `KeyError`, and the
  migration note must say so.
