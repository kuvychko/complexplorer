## Context

`cp.catalog` is a registry of *serializable* presets: each carries a callable, a parseable
`expression` string, plain-dict specs, and a hand-authored `singularities` answer key. The gallery
renders the whole catalog into a byte-stable `index.json`, the CLI resolves `preset:<id>`, and the
STL object cards read the same records. Anything added here is added to all four.

The six pieces being ported are ratios of **Klein relative invariants**. A degree-`d` rational map
has exactly `d` zeros and `d` poles on the sphere, so "poles at the six octahedron vertices and
nothing else" is impossible. For the relief to carry the full polyhedral rotation symmetry, `f` must
be a ratio of relative invariants of equal *binary-form* degree: the automorphy factors then cancel
and `|f|` descends to a genuine invariant function on the sphere. That is why these are ratios like
`H^3 / T^2` (60 over 60) rather than single forms.

Two constraints from the surrounding code shape everything below. `expression` must be a real,
parseable string — so a preset's mathematics is written out in full, not hidden behind a helper.
And `singularities` is specified as hand-authored ground truth, "not values produced by a numerical
detector" — which collides with a divisor of sixty algebraic numbers.

## Goals / Non-Goals

**Goals:**

- The six polyhedral pieces are one call from the library, carrying the relief settings they need.
- The invariant forms are available for building new pieces, with their correctness pinned by
  mathematics rather than by careful transcription.
- The reference collection's geometry stays reproducible to the digit.

**Non-Goals:**

- The non-polyhedral pieces of the reference collection (filters, dipole/quadrupole/pole-flower
  family, `alternating-crown-6`).
- Changing the library's default `resolution`.
- Print-oriented helpers beyond what 3.1 already has (cut planes, sectioning, slicer settings).
- Any 2D-portrait tuning beyond choosing a domain that does not overflow.

## Decisions

### The invariant forms live in their own module, and are public

A new `complexplorer/core/polyhedral.py` holds eight forms: the tetrahedral pair, the octahedral
vertex/face/edge forms, and the icosahedral trio. `presets.py` is already the longest module in
`core/` and these are reusable mathematics rather than preset data.

They are re-exported flat (`cp.icosahedral_hessian` and friends), consistent with the rest of the
public API. The alternative — keeping them importable only from their module — was considered,
because eight degree-4-to-30 polynomials is a lot of surface for a niche feature. It is rejected
because the whole point is that these are hard to get right: a form nobody can find gets retyped
from a half-remembered paper, which is precisely the failure this change exists to prevent.

Names are descriptive rather than classical (`icosahedral_hessian`, not `H`), with the classical
name in the docstring, because `V`, `F`, `E`, `H`, `T` collide with everything.

### Correctness is pinned by Klein's syzygy, not by transcription

This is the decision that matters most. The forms in the literature only cohere as a set for one
orientation of the icosahedron, and the commonly-remembered signs mix orientations. Pairing
`V = z(z^10 + 11z^5 - 1)` with the usual Hessian gives three root sets that are each *individually*
a perfect icosahedron, dodecahedron and edge set while being rotated relative to one another. The
consequence is silent: `|V^5/H^3|` stops being rotation-invariant, by a factor of 1e3 to 1e5, and no
per-form geometry check notices.

Measured here: with `z(z^10 - 11z^5 - 1)` the syzygy `H^3 - T^2 = 1728 V^5` holds to a relative
2e-14; with the `+11` variant it is wrong by a factor of 20. So the syzygy is the test. It is a
single identity that fails loudly for any mixed-orientation set, which is stronger than checking
each form's roots against a solid and much stronger than re-reading the coefficients.

The tests therefore assert: the syzygy; rotation invariance of each ratio under an element of the
group; and, per form, that its roots are the expected count and lie at the expected solid's
vertices. Ported from the reference's `verify_math.py`, which checks the same claims.

### `singularities` records the divisor from the solid's geometry, not from a root solve

The existing contract wants hand-authored keys. A sixty-root divisor cannot be hand-authored
usefully: typed decimals would be less accurate than computing, and unverifiable besides.

The obvious move is to solve the polynomial — `np.roots` on the integer coefficients. **It does not
survive the manifest contract.** `index.json` is byte-compared across platforms, and preset records
are quantized to `MANIFEST_SIGNIFICANT_DIGITS = 12` significant digits. Degree-30 companion-matrix
eigenvalues do not agree to twelve significant digits between LAPACK builds; root accuracy at that
degree is nearer 1e-10 relative. So a solved key would break the byte-stable manifest on somebody
else's machine, and the existing quantization does not save it.

The better source is the one the reference used to *derive* the coefficients in the first place:
explicit geometry. Build the solid — for the icosahedral set, a vertex at the north pole and rings at
`z = ±1/sqrt(5)` — and stereographically project its vertices, face centres or edge midpoints. The
locations then come from `sqrt(5)` and trigonometry, which are deterministic to full double precision
on every platform, and the polynomial becomes the *check* rather than the source: the monic polynomial
over those projected roots must reproduce the stated integer coefficients.

This inverts the dependency in the right direction. The geometry is the ground truth, stated as
"the twenty vertices of the dodecahedron"; the integer coefficients and the recorded locations are
two derived views of it, each verifying the other. That is a stronger answer key than either alone,
and it is genuinely hand-authored at the level that matters — the solid, not the decimals.

This also buys something real: `answer_key_stats().min_separation` becomes the tightest
peak-next-to-pit distance, which is the ridge-width proxy that decides whether a piece prints at a
given size. On the reference collection that number, not the radius, is the binding constraint on
the closest-featured piece.

**A feature at infinity cannot be recorded, and must be said so.** `octahedral_vertex` and
`icosahedral_vertex` are one degree short as polynomials — their last root is the vertex at
infinity. `icosahedral_crown` is `T^2 / V^5`, so it carries a pole of order 5 at infinity that no
`[real, imag]` pair can express. Those presets state it in `story` and omit it from the key, the way
`exp` already omits its essential singularity at infinity.

### A preset may carry `pole_order` and `resolution`

Two new optional fields, both plain data, both serialized. A preset that knows it has order-2
features but is rendered at the default `pole_order=1` produces an exact cone where a cusp was
intended — shipping the piece without its order ships the blunt version.

`resolution` is per-preset rather than a new global default: the reference uses 250 for the simpler
pieces and 400 for the degree-60 ones, and raising the library default to 400 would make every
unrelated render four times slower. `OrnamentGenerator` and the CLI consult the preset when it has
them; an explicit argument still wins.

The alternative of a generic `relief_spec` dict, mirroring `domain_spec`, was considered and
rejected: `pole_order` is not a constructor kwarg for one class, and two named fields are cheaper to
read than a dict whose keys must be documented separately.

### The gallery regenerates, and the portraits need bounded domains

`index.json` enumerates the whole catalog, so six records join it and the committed manifest must be
regenerated — it has a test asserting the committed file reproduces byte for byte. The portrait PNGs
that come with it are reproducible only best-effort and will differ across machines; that is already
the documented contract.

Numerics set the portrait domain. These are degree-60 rationals: `ICO_T(z)**2` overflows float64
beyond roughly `|z| = 1e5`. The sphere sampler's `avoid_poles` clamp reaches about `|z| = 200`, well
clear, which is why the reference renders fine at resolution 400. A 2D portrait over a wide rectangle
is not clear, so each preset gets a modest domain — the reference uses a half-width of 2.0 — and the
`expression` string must be safe as written at those magnitudes.

## Risks / Trade-offs

- **[The syzygy passes but the orientation is still not the one intended]** → The syzygy pins the
  three icosahedral forms to *each other*; it does not fix which rotation of the icosahedron they
  describe. That is acceptable, because a relief is rendered on the sphere and any rotation of it is
  the same object. The rotation-invariance test covers the part that matters.
- **[Eight new flat names for a niche feature]** → Grouped on one API reference page, and the six
  presets are the entry point most readers will use. Revisit if the flat namespace becomes crowded.
- **[Root locations drift across platforms and break the byte-stable manifest]** → This is why the
  locations come from projected geometry rather than from `np.roots`; see the decision above. The
  residual risk is that trigonometry also disagrees in the last place, which the manifest's existing
  12-significant-digit quantization absorbs — libm differences are one ulp, not 1e-10. The test to
  write is the one that would have caught the solve: generate the manifest, and assert the recorded
  locations reproduce the stated integer coefficients, so a platform that computed them differently
  fails loudly.
- **[Six more presets slow the gallery]** → They are the most expensive in the catalog (degree-60 at
  portrait resolution). Measure before and after; if it matters, the gallery already supports
  rendering a tag subset.
- **[The reference is a separate repo and can drift]** → The library becomes the authority once these
  ship. The reference updates itself after 3.1, which its own notes already say.

## Open Questions

- Does this ship in 3.1 or after it? 3.1 is already a large release and its migration note is
  written; adding six presets and a regenerated gallery grows it. Shipping after keeps 3.1 focused
  and lets the release chain be exercised on the smaller change first.
- Do the presets belong in the `ornament` tag alone, or does a `polyhedral` tag earn its place? Six
  pieces sharing construction, symmetry vocabulary and a print workflow is a reasonable tag.
- Should the two non-polyhedral families follow? The filters overlap `cp.ee` and would need a story
  about why a transfer function is in the function catalog.
