## Why

`cp.catalog` has 17 presets and exactly one tagged `ornament` that is actually a printable piece:
`pole_flower_10`. Meanwhile `figures_repo/2026/09-20-riemann-ornaments` holds a thirteen-piece
collection that has been built, rendered, printed and measured, and six of those pieces — the
polyhedral ones — are not reachable from this library at all. Reproducing
`icosidodecahedral-star`, the piece currently on the printer, means pasting two degree-20 and
degree-30 polynomials into a script by hand.

That is the gap worth closing, and 3.1 is the release that makes closing it worthwhile: with
normalization on by default, `pointiness`, `pole_order` and the cap in place, the library now
reproduces the star to the digit — sharpness 4.0, base 54.598150, depth 0.2, 159,600 points, and
`max_radius` 66.77 mm at 130 mm, matching the reference manifest exactly. The mathematics is
already expressible. What is missing is that nobody can find it.

**There is a second reason, and it is about correctness rather than convenience.** These
invariants are exactly the kind of constant that is remembered wrong. The reference records that
the forms quoted in the literature only cohere as a set for one particular orientation of the
icosahedron, and that the commonly-remembered signs mix orientations: pairing
`V = z(z^10 + 11z^5 - 1)` with the usual `H` gives three root sets that are each *individually* a
perfect icosahedron, dodecahedron and edge set, while being rotated relative to one another.
`|V^5/H^3|` then fails rotation invariance by a factor of 1e3 to 1e5 and Klein's syzygy does not
hold at all — a failure invisible to anyone who checks each form's geometry separately. This was
confirmed here: with the `-11` variant the syzygy `H^3 - T^2 = 1728 V^5` holds to 2e-14, and with
the `+11` variant it is wrong by a factor of 20. A library that ships these forms with a test
asserting the syzygy is worth more than a folder that has them right by luck.

## What Changes

- **Six polyhedral presets** join `cp.catalog`, ported with their stories, divisors and symmetry
  groups: `tetrahedral_dual`, `octahedral_crown`, `cube_octahedron_dual`, `icosahedral_crown`,
  `dodecahedron_icosahedron_dual`, `icosidodecahedral_star`.
- **The Klein relative invariants become library functions** rather than inline polynomials:
  the tetrahedral pair, the octahedral vertex/face/edge forms, and the icosahedral trio. A relief
  carries the full polyhedral rotation symmetry only when `f` is a ratio of relative invariants of
  equal *binary-form* degree — the automorphy factors cancel and `|f|` descends to a genuine
  invariant function on the sphere — so the forms belong together with a note on why the degrees
  must match.
- **A preset can carry its relief settings.** These pieces need `pole_order` (2, 3 and 5 among
  them) and a higher `resolution` than the library default of 150; the reference uses 250–400. A
  preset that records `pole_order=2` but is rendered at the default `pole_order=1` produces an
  exact cone where a cusp was intended, which defeats the point of shipping it.
- **Tests that assert the mathematics, not the output:** Klein's syzygy, rotation invariance of
  each ratio, and the root sets being the right solids. These are the checks that catch a
  mixed-orientation transcription, which a geometry-only check does not.
- The gallery gains six entries, because it renders the whole catalog.

**Not in scope.** The non-polyhedral pieces of the reference collection (the two filters, the
dipole/quadrupole/pole-flower family, `alternating-crown-6`). Some are near-duplicates of existing
presets and the filters belong with `cp.ee`; they are a separate question. Changing the library's
default `resolution` is also out of scope — per-preset resolution is the mechanism here.

## Capabilities

### New Capabilities

_None._

### Modified Capabilities

- `function-presets`: the registry gains the polyhedral family; a preset may carry the relief
  parameters (`pole_order`, `resolution`) that a printable piece needs to come out as intended;
  the polyhedral invariants are part of the capability's surface, with their correctness pinned by
  the syzygy rather than by transcription.

## Impact

- **Code:** `complexplorer/core/presets.py` (the six records plus the invariant forms); possibly a
  new module for the forms if `presets.py` grows unwieldy.
- **Assets:** `examples/gallery/index.json` gains six records and must be regenerated — it is a
  byte-stable contract with a test asserting the committed file reproduces. The `gallery`
  capability's *requirements* do not change: it already specifies that the manifest carries every
  rendered preset and quantizes floats for cross-platform stability, so this is a new asset under an
  unchanged contract, not a new behaviour. Six new portrait PNGs
  come with it; those are reproducible only best-effort and will differ across machines.
- **Numerics:** these are degree-60 rational functions. `ICO_T(z)**2` overflows float64 beyond
  about `|z| = 1e5`, which the sphere's `avoid_poles` clamp keeps well clear of (it reaches about
  `|z| = 200`), but the 2D portrait domains must stay modest and each preset's `expression` string
  has to be numerically safe as written.
- **Answer keys:** the divisors here are 20 to 60 algebraic numbers — roots of the invariant
  polynomials — not the hand-authorable constants the existing presets carry. How
  `singularities` represents them is the main design question, and `design.md` decides it.
- **No breaking changes.** Everything here is additive.
