## 0. Sequencing

> Settle this before starting: 3.1 is already a large release with its migration note written, and
> six new presets plus a regenerated gallery grows it. Shipping this after 3.1 keeps that release
> focused and exercises the tag-triggered release chain on the smaller change first.

- [ ] 0.1 Decide whether this ships in 3.1 or in the release after it, and record the decision here.

## 1. The invariant forms

> Ported from `figures_repo/2026/09-20-riemann-ornaments/ornaments.py`, whose `verify_math.py`
> checks every claim below. Copy the coefficients from there, not from a paper — see 2.1 for why.

- [ ] 1.1 Add `complexplorer/core/polyhedral.py` with the eight forms: tetrahedral vertex and its
  antipodal dual (binary degree 4 each), octahedral vertex (6), face/cube vertex (8) and edge (12),
  and the icosahedral vertex (12), Hessian (20) and edge (30).
- [ ] 1.2 Use descriptive names with the classical letter in the docstring, because `V`, `F`, `E`,
  `H` and `T` collide with everything. Note in each docstring which solid's features are its roots.
- [ ] 1.3 State in the module docstring why the ratios must have **equal binary degree**: a degree-`d`
  rational map has exactly `d` zeros and `d` poles on the sphere, so "features at one solid's
  vertices and nowhere else" does not exist, and only at equal degree do the automorphy factors
  cancel so that `|f|` descends to a genuine invariant function.
- [ ] 1.4 Record which forms are one degree short as polynomials — octahedral and icosahedral vertex
  — because their last root is the vertex **at infinity**, which matters for both the divisor and the
  normalization helper.
- [ ] 1.5 Re-export flat from the package, and add them to the API reference page (the docs-site test
  asserts every exported name is documented).

## 2. Pin the mathematics, not the transcription

> This is the point of the change. The forms in the literature cohere only for one orientation, and
> the commonly-remembered signs mix orientations — a mixed set is three root sets that are each
> individually a perfect solid while being rotated relative to one another, whereupon the ratios stop
> being rotation-invariant by a factor of 1e3 to 1e5. No per-form geometry check notices.

- [ ] 2.1 Test Klein's syzygy: `H^3 - T^2 == 1728 * V^5` for the icosahedral trio. Confirmed to hold
  to a relative 2e-14 with `z(z^10 - 11z^5 - 1)`, and to fail by a factor of ~20 with the `+11`
  variant — assert both directions, so the test demonstrates it discriminates.
- [ ] 2.2 Test rotation invariance of each documented ratio under an element of the corresponding
  rotation group, which is the property a mixed set loses.
- [ ] 2.3 Test each form's root count against its solid, allowing for the root at infinity.
- [ ] 2.4 Test that the tetrahedral pair multiplies to the cube vertex form exactly — the two
  tetrahedra together are the cube.

## 3. Divisors from geometry

> `design.md` decides this: construct the locations by stereographically projecting an explicit
> solid, and do **not** solve the polynomial. Preset records are serialized into a manifest that is
> byte-compared across platforms after quantization to 12 significant digits, and degree-30
> companion-matrix roots do not agree to 12 digits between LAPACK builds.

- [ ] 3.1 Build each solid explicitly — for the icosahedral set, a vertex at the north pole and rings
  at `z = ±1/sqrt(5)` — and project its vertices, face centres and edge midpoints stereographically.
- [ ] 3.2 Verify in the other direction: the monic polynomial over the projected roots reproduces the
  stated integer coefficients. That is the check which would have caught a solved key, so write it
  as a test rather than a one-off script.
- [ ] 3.3 Assemble each preset's `singularities` from those locations with the right multiplicities
  (for the icosidodecahedral piece: 20 triple zeros, 30 double poles).
- [ ] 3.4 For the pieces with a feature at infinity — anything over the octahedral or icosahedral
  vertex form — omit it from the key and say so in `story`, as `exp` already does for its essential
  singularity.
- [ ] 3.5 Check `answer_key_stats().min_separation` on each: it is the tightest peak-next-to-pit
  distance, which is the ridge-width proxy that decides whether a piece prints at a given size.

## 4. Preset-carried relief settings

- [ ] 4.1 Add `pole_order` and `resolution` to `FunctionPreset` as optional plain data, serialized
  with the rest of the record.
- [ ] 4.2 Have `OrnamentGenerator` / `create_ornament` use them when built from a preset, with an
  explicit argument taking precedence.
- [ ] 4.3 Update `cli/main.py`'s `stl` subcommand to honour them for `preset:<id>`, so
  `complexplorer stl preset:icosidodecahedral_star` produces the intended piece without flags.
- [ ] 4.4 Confirm the existing presets are unaffected — the fields are absent, not defaulted to
  something that changes their geometry.

## 5. The six presets

> Settings from the reference manifest, which records what was actually built and printed.

- [ ] 5.1 `tetrahedral_dual` — `pole_order=1`, resolution 250. The one piece with no mirror plane, so
  its halves are genuinely different; say so, it is the point of the piece.
- [ ] 5.2 `octahedral_crown` — `pole_order=2`, resolution 300.
- [ ] 5.3 `cube_octahedron_dual` — `pole_order=3`, resolution 300. Its spikes lie on the cube
  diagonals, which is the piece that made the bounding-box sizing defect visible: it was understated
  by exactly `sqrt(3)`.
- [ ] 5.4 `icosahedral_crown` — `pole_order=5`, resolution 400. Its derived scale hits the cap, which
  is the case the cap exists for; note that in the story.
- [ ] 5.5 `dodecahedron_icosahedron_dual` — `pole_order=3`, resolution 400.
- [ ] 5.6 `icosidodecahedral_star` — `pole_order=2`, resolution 400. The densest piece in the
  collection. Verify it still reproduces the reference exactly once it comes from the catalog:
  sharpness 4.0, base 54.598150, depth 0.2, and `max_radius` 66.77 mm at `size_mm=130`.
- [ ] 5.7 Give each a `story` naming its zeros, poles and **symmetry group** — the group is what
  makes the piece what it is and is not derivable from the expression by inspection.
- [ ] 5.8 Tag them so the family is discoverable as a group, and keep `ornament` on the printable
  ones.
- [ ] 5.9 Write each `expression` out in full and test it against the callable. They are long —
  degree 60 — but a preset's expression is a contract, not a convenience.

## 6. Numerics and portraits

- [ ] 6.1 Choose a 2D portrait domain per preset that does not overflow: these are degree-60
  rationals and `ICO_T(z)**2` overflows float64 past roughly `|z| = 1e5`. The reference uses a
  half-width of 2.0. The sphere sampler is safe — its `avoid_poles` clamp reaches about `|z| = 200`.
- [ ] 6.2 Confirm each preset renders as a 2D portrait, on the sphere, and as an STL, all without
  warnings that a user would have to learn to ignore.

## 7. Gallery

- [ ] 7.1 Regenerate `examples/gallery/` — the manifest enumerates the whole catalog, so `index.json`
  gains six records and its byte-stability test fails until it is regenerated.
- [ ] 7.2 Review the `index.json` diff rather than accepting it: six added records and nothing else
  should change.
- [ ] 7.3 Note the gallery generation time before and after. These are the most expensive functions
  in the catalog; if it becomes a problem, the generator already supports rendering a tag subset.

## 8. Docs

- [ ] 8.1 A section in the physical-workflow guide on building a symmetric relief: why equal binary
  degree is required, the eight forms, and the syzygy as the check to run on any set you transcribe
  yourself.
- [ ] 8.2 Show the star as the worked example, since it is the densest and the one most likely to be
  reproduced.
- [ ] 8.3 Note that `normalization_constant` must not be fed these divisors from the preset's
  `singularities` where a feature sits at infinity — the closed form takes finite divisors only, and
  would return a confidently wrong number.

## 9. Verification

- [ ] 9.1 `pytest`, `ruff check` / `format`, `openspec validate --specs`, and
  `openspec validate polyhedral-ornament-presets`.
- [ ] 9.2 `mkdocs build --strict`.
- [ ] 9.3 Generate all six at print resolution and confirm each is watertight, manifold and
  positive-volume under the corrected check, as the 3.1 work did for the existing catalog.
- [ ] 9.4 Compare all six against the reference manifest — normalization constant, sharpness, depth,
  point and triangle counts, and `max_radius` at 130 mm. Any disagreement is a porting error in this
  change, not a difference of opinion.
