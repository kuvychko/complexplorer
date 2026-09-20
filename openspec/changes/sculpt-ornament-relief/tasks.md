## 1. Edge counting

> `figures_repo/2026/09-20-riemann-ornaments/meshtools.py` has a working `_count_edges` and
> `inspect_mesh`. Sections 1 and 2 are a port, not a design.

- [ ] 1.1 Add a single helper that counts one class of edge with all four
  `extract_feature_edges` flags passed explicitly, so no call site can inherit the defaults again.
- [ ] 1.2 Use it in `validate_printability` for both the watertight and the manifold checks, and
  derive each boolean and its count from the same measure rather than mixing `n_points` and
  `n_cells`.
- [ ] 1.3 Use it in `repair_mesh_simple`'s verbose check and in `close_mesh_holes`'s
  `Initial` / `After filling` counters.
- [ ] 1.4 Revisit the `n_boundary_edges < 200` branch in the verbose assessment: with the count
  fixed, "small gaps typical of a Riemann sphere" describes a case that should now be rare, and the
  wording should not excuse a genuinely open mesh.

## 2. Radial extent replaces wall thickness

- [ ] 2.1 Remove the `wall_thickness_ok` / `estimated_min_wall_mm` computation and the
  `recommended_size_mm` line that depends on it.
- [ ] 2.2 Report `min_radius_mm` and `max_radius_mm` from `np.linalg.norm(mesh.points, axis=1)`,
  measured before centring — the order in `save_stl` must change so validation sees the uncentred,
  scaled mesh.
- [ ] 2.3 State in the verbose report that ridge width between adjacent features is not measured.

## 3. Custom scaling dispatch

- [ ] 3.1 Route `custom` through `ModulusScaling.custom` in `apply_scaling_mode`, keeping the
  missing-callable validation error.

## 4. Normalization

- [ ] 4.1 Add `normalization_constant(zeros, poles, gain=1.0)` to `core/scaling.py` and export it.
  Docstring must state that the divisor has to be complete and that a transcendental function or a
  partial list gives a wrong answer.
- [ ] 4.2 Add the sampled estimator: `sin(theta)` weights, seam meridian dropped, non-finite
  `log|f|` excluded. Geometric mean and median share one code path.
- [ ] 4.3 Wire `normalize` into `OrnamentGenerator` and `create_ornament`, accepting
  `"geometric"` (default), `"median"`, a float, or `None`. Compute from the field already sampled
  in `generate_ornament` — no second evaluation of `f`.
- [ ] 4.4 Apply the constant to the moduli before the transfer, keep the `magnitude` scalar raw,
  and record the applied constant in mesh metadata.

## 5. Pointiness and the default transfer

- [ ] 5.1 Move the default ornament transfer to the logistic in log-modulus
  (`ModulusScaling.logarithmic`, where `k = ln(base)`), and change the default depth to
  `r_min = 0.2` to match `SCALING_PRESETS`.
- [ ] 5.2 Add `pointiness` (with optional `pole_order`, default 1) mapping to `k` as
  `k = pointiness * pole_order`.
- [ ] 5.3 Update `cli/main.py:128`, which hardcodes `"arctan"`, and
  `get_default_scaling_params(for_stl=True)`, so the default lives in one place rather than three.
- [ ] 5.4 Default `pointiness` to **2.0**, validated visually across the ten-piece reference
  collection. Do not calibrate against the bulk-contrast measure alone — see `design.md` for why it
  is the wrong proxy. If a quantitative check is wanted, use feature clearance (how far a petal
  rises above the valley floor between adjacent petals).
- [ ] 5.5 Allow the log-modulus scale to be set directly, bypassing `pointiness * pole_order`, for
  the gain-calibrated case where `ln(10)` makes one unit of relief equal 20 dB.

## 5a. Contrast

- [ ] 5a.1 Add the two-scale mixture: `weight * logistic(L / (k / boost)) + (1 - weight) * logistic(L / k)`,
  mapped onto `[r_min, r_max]`. Port from `mixture_scaling` in `ornaments.py`, but return a value
  in `[0, 1]` and let the caller's bounds do the mapping — the reference bakes depth into its own
  return only because of the `custom` dispatch bug this change fixes.
- [ ] 5a.2 Expose it as `contrast=(boost, weight)`, defaulting to off.
- [ ] 5a.3 Confirm it stays odd in `log|f|` so self-duality holds, and document the honest limit:
  asymptotically the wide term still sets the tip exponent, but that asymptote lies below mesh
  resolution, so the visible point is measurably shorter than the exponent implies.

## 6. Tests

- [ ] 6.1 A closed, creased mesh (`pv.Cube().triangulate()` is sufficient and fast) validates as
  watertight and manifold. This is the regression test for the reported bug.
- [ ] 6.2 Validation reports `min_radius_mm` / `max_radius_mm` and no longer reports a wall
  thickness; the radii are measured before centring.
- [ ] 6.3 `custom` mode with `r_min` / `r_max` maps a callable returning `[0, 1]` onto those
  bounds, and clips a callable that overshoots.
- [ ] 6.4 `normalization_constant` against the closed-form values: `z/(z^5-1)` -> `4*sqrt(2)`,
  `z/(z^10-1)` -> `32`, `3/z` -> `1/3`, and `1` for a reciprocal-symmetric divisor.
- [ ] 6.5 The sampled estimator agrees with the closed form to within **0.5%** — loose on purpose.
  Per `design.md`, the fixed `avoid_poles` clamp means accuracy does not improve with resolution,
  so this tolerance must not be tightened.
- [ ] 6.6 The two grid corrections, tested as correctness: an unweighted average gets
  `z/(z^10-1)` roughly ten times wrong, and retaining the seam biases `(z-1)/(z+1)` by around 1%
  while leaving `(z-1j)/(z+1j)` unaffected.
- [ ] 6.7 Scale invariance end to end: ornaments for `f` and `100 * f` have matching geometry.
- [ ] 6.8 Self-duality: the constants for `f` and `1/f` are reciprocal.
- [ ] 6.9 `normalize=None` reproduces the unnormalized mapping.
- [ ] 6.10 Saved meshes have consistent outward normals.
- [ ] 6.11 The contrast mixture is self-dual: the relief of `1/f` mirrors that of `f` about sea
  level, to within sampling error.
- [ ] 6.12 Contrast is off by default, and supplying it changes the radial distribution in the
  expected direction.
- [ ] 6.13 An explicit log-modulus scale overrides `pointiness` and `pole_order`.
- [ ] 6.14 Regenerate `tests/regression/baselines/ornament.npz` and note in the commit that the
  geometry change is intended.

## 7. Docs

- [ ] 7.1 "Why is my ornament a blob" in `docs/guide/physical-workflow.md`: normalization first
  (it carries the weight), then the tip exponent, then the limit.
- [ ] 7.2 The rule of thumb: a relief is only as interesting as its zero/pole divisor is dense.
  `(z-1)/(z+1)` is an egg at every setting because `|f|` rises monotonically from one pole of the
  sphere to the other; a three-feature transfer function is the same. Say so, so users stop
  turning knobs.
- [ ] 7.3 A warning on `custom` mode: a transfer must be **smooth** at `|f| = 1`, not merely
  monotone and odd. A signed power of the log modulus has infinite derivative there, creases the
  surface along every sea-level contour, and prevents the seam from welding — the mesh will not
  close. (Section 3.2 of the notes; recorded so the dead end is not walked twice.)
- [ ] 7.4 **Its own migration entry** for the `custom` dispatch fix — this one silently changes
  geometry rather than raising. Code written against the bug returns a radius directly; after the
  fix that value is clipped to `[0, 1]` and remapped onto `[r_min, r_max]` (default `[0.5, 1.5]`).
  Show the before/after and the one-line correction.
- [ ] 7.5 Migration note for 3.1: default ornament geometry changes; `normalize=None` restores the
  old normalization behaviour and `scaling="arctan"`, `r_min=0.5` restore the old look;
  `wall_thickness_ok` and `estimated_min_wall_mm` are gone from the validation dict.

- [ ] 7.6 Document `contrast` with the guidance the collection demonstrates: use it on pieces whose
  few features leave the body smooth; leave it off where the tips are the point.

## 8. Verification

- [ ] 8.1 `pytest`, `ruff check` / `format`, `openspec validate --specs`, and
  `openspec validate sculpt-ornament-relief`.
- [ ] 8.2 Generate the catalog's printable presets at the new defaults and confirm each is
  watertight, manifold and positive-volume **under the corrected check**.
- [ ] 8.3 Record before/after bulk-contrast numbers for those pieces, so the release can state what
  changed rather than asserting it.
