## Why

A ten-piece printable collection was built against complexplorer 3.0.0 — the generators live in
`figures_repo/2026/09-20-riemann-ornaments`, whose module docstrings carry the derivations and the
measurements referenced below. Two things came out of that build: three real bugs, and the observation that nearly every ornament the library produces is "a sphere with
features poked into it" regardless of the mathematics behind it.

The three bugs were confirmed against the source, and the first was reproduced cold: a
`pyvista.Cube()` — closed, manifold, `n_open_edges == 0`, unit volume — is reported by
`validate_printability` as **neither watertight nor manifold**. `extract_feature_edges` is called
with only `boundary_edges=True`, leaving `feature_edges`, `manifold_edges` and
`non_manifold_edges` at their `True` defaults, so the call returns essentially every edge in the
mesh and any crease reads as a hole. `repair_mesh_simple` and `close_mesh_holes` carry the
identical mistake in their progress counters, which is why repair looks like it is failing when it
is working. The practical cost is that users write redundant repair code against a false report.

The blob problem has a measurable cause, and it is not the transfer function. Every self-dual
transfer puts sea level at `|f| = 1`, so the arbitrary constant in front of a function changes the
*shape* of the relief rather than just its labels. Measuring the share of surface area sitting in
the middle half of the radial range — high means featureless — normalizing `z/(z^10-1)` moves it
from 40.5% to 9.0%, a 4.5x improvement, while swapping the transfer from `arctan` to a logistic
moves it from 40.5% to 47.0%, slightly the wrong way. Normalization is the fix; the transfer is a
knob.

Normalization has an exact closed form for a rational `f`, verified here against quadrature to
five decimals on five functions, including `z/(z^10-1) -> 32` and `3/z -> 1/3`. But the closed
form needs a *complete* divisor, and the catalog's `singularities` answer keys are illustrative,
not complete: fed the three listed zeros of `sin(z)`, the closed form returns 0.092 against a true
0.853, nine times wrong. The constant is therefore computed by sampling, which has no such failure
mode, and the closed form ships as a public helper and as the exact oracle the tests check the
sampled estimator against.

## What Changes

**Bug fixes.**

- Edge counting passes all four `extract_feature_edges` flags explicitly wherever an edge class is
  counted, in `validate_printability`, `repair_mesh_simple` and `close_mesh_holes`. `n_points` and
  `n_cells` stop being used interchangeably between a boolean and its count.
- The wall-thickness check is removed rather than repaired. `save_stl` validates *after*
  `scale_to_size`, so `scale_factor` is always 1.0 and the test reduces to `0.3 >= 0.8` — constant
  `False`, with a "recommend at least" line that always suggests `size_mm * 2.67`. A relief is a
  star-shaped solid about the origin and has no walls. It is replaced by `min_radius_mm` and
  `max_radius_mm`, measured before centring, plus an explicit statement that ridge width between
  adjacent pits — the feature that actually fails on an FDM printer — is not measured.
- `custom` scaling is routed through `ModulusScaling.custom` instead of short-circuiting ahead of
  it, restoring the documented `[r_min, r_max]` mapping and the `np.clip`.
- The save path computes consistent, outward-oriented normals so the STL's facet normals mean
  something to slicers that read them.

**Normalization, on by default.**

- `normalization_constant(zeros, poles, gain=1.0)` ships as a public closed-form helper.
- The ornament generator computes the constant by **sampling**, from the sphere field it has
  already evaluated — no second pass over `f` — and applies it to `|f|` before the transfer.
  `normalize` accepts `"geometric"` (default), `"median"`, an explicit float, or `None`.
- The estimator weights by `sin(theta)` and drops the duplicated seam meridian. Both matter:
  unweighted, the lat/long grid gets `z/(z^10-1)` 10.1x wrong; with the seam counted twice,
  `(z-1)/(z+1)` is 1.05% off instead of 0.01%, and the seam lies along the positive real axis
  where real-coefficient functions put their features.

**A tuning knob for tip shape.**

- `pointiness` exposes the quantity the eye actually reads. With `r = r_min + delta * sigma(log|f| / k)`
  the surface approaches a feature of order `mu` like `distance ** (mu / k)`, so `mu / k` — not
  `k` — decides whether a tip is a dome, an exact cone, or a cusp. `pointiness` sets `k` (scaled by
  an optional `pole_order`, default 1), giving a tip exponent of `1 / pointiness`.
- This requires the default transfer to move from `arctan`, which has no `k` and reaches 90% of its
  height at `|f| = 6.3` whatever the function, to the logistic in log-modulus. That transfer
  already exists as `ModulusScaling.logarithmic`, which is exactly `r_min + delta * sigma(L / k)`
  with `k = ln(base)`; no new mathematics is needed.
- The default relief depth moves from `r_min = 0.5` to `0.2`, matching every entry in
  `SCALING_PRESETS`, which the STL defaults alone diverge from.

**Documentation.** A "why is my ornament a blob" section in `guide/physical-workflow.md` covering
normalization, the tip exponent, and the honest limit: a relief is only as interesting as its
zero/pole divisor is dense. `(z-1)/(z+1)` is an egg at every setting because `|f|` rises
monotonically from one pole of the sphere to the other, and no amount of contrast changes that.

**A two-scale contrast mixture.** `contrast=(boost, weight)` mixes a narrow logistic at
`k / boost` for bulk contrast with the original wide one for the tips. Both terms are odd in
`log|f|`, so self-duality is preserved exactly and peaks and pits stay mirror images. It is smooth
at `|f| = 1`, which matters: a signed power of the log modulus is monotone, odd and tempting, but
has infinite derivative there, creases the surface along every sea-level contour and splits the
mesh seam badly enough that the weld fails.

`pointiness` and `contrast` ship together because they are a matched pair. Raising `pointiness`
sharpens tips at the cost of the bulk, and `contrast` is the only remedy; 6 of the 10 pieces in the
reference collection set it. The notes' "improved four, regressed three" was an argument against a
*global* default, not against the parameter — the three it regressed are exactly the pieces that
leave it unset. It therefore defaults to off.

**Reference implementation.** `figures_repo/2026/09-20-riemann-ornaments` is a working, visually
validated build of all of this against 3.0.0. `meshtools.py` already contains the corrected
`_count_edges` and the `inspect_mesh` radial report, and `ornaments.py` contains the normalization,
the transfer and the mixture. Those are port targets, which removes most of the design risk from
this change; that repo is updated separately after 3.1.0 ships.

## Capabilities

### New Capabilities

_None._

### Modified Capabilities

- `stl-export`: printability validation reports correct topology and radial extent instead of a
  constant-`False` wall-thickness test; ornaments are normalized by default; the default relief is
  deeper and its tip shape is tunable; saved meshes carry oriented normals.
- `modulus-scaling`: `custom` mode honours `r_min`/`r_max`; a normalization constant is part of the
  capability's public surface; the logistic transfer is parameterized by tip exponent.

## Impact

- **Code:** `complexplorer/export/stl/utils.py`, `mesh_repair.py`, `ornament_generator.py`;
  `complexplorer/core/scaling.py`; `complexplorer/utils/mesh_distortion.py`;
  `complexplorer/cli/main.py` (the `"arctan"` default at line 128).
- **Baselines:** `tests/regression/baselines/ornament.npz` must be regenerated — the default
  geometry changes deliberately. The gallery `index.json` contract is unaffected: presets carry
  named `scaling_spec` values such as `"poles_emphasis"`, which the STL defaults do not touch.
- **Breaking:** fixing `custom` dispatch changes what an existing custom callable means. Code
  written against the bug returns a radius directly (the reference implementation does exactly
  this, and documents why); after the fix that value is clipped to `[0, 1]` and remapped onto
  `[r_min, r_max]`, producing wrong geometry silently rather than an error. This needs its own
  migration entry, not a bullet in a list.
- **Behaviour:** every ornament generated with default settings changes shape. This is the point of
  the release, and it needs a migration note. `normalize=None` reproduces the old normalization
  behaviour; the old *look* additionally needs `scaling="arctan"` and `r_min=0.5`.
- **Docs:** `docs/guide/physical-workflow.md`, `docs/migration-3.0.md` sibling note for 3.1.
