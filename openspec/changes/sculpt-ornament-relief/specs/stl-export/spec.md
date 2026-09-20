## MODIFIED Requirements

### Requirement: Printability repair and validation

The library SHALL repair and validate the ornament mesh for 3D printing, reporting topology,
dimensions and radial extent, while tolerating the small polar gaps inherent to a rectangular
sphere mesh.

Wherever a class of edge is counted — in printability validation, in mesh repair, and in
hole-filling progress output — the count SHALL request that class explicitly and SHALL disable the
classes it is not counting. `extract_feature_edges` enables boundary, feature, manifold and
non-manifold extraction by default, so requesting one class alone returns substantially every edge
in the mesh and reports any creased-but-closed surface as open. A boolean and its accompanying
count SHALL be derived from the same measure.

Validation SHALL report the mesh's minimum and maximum radius from the origin in millimetres,
measured **before** the mesh is centred, because centring shifts the star centre away from the
origin and understates the range on a lopsided piece. Validation SHALL NOT report an estimated wall
thickness: the relief is a star-shaped solid about the origin and has no walls, and the quantity
that does fail on a fused-deposition printer — the ridge width between adjacent pits — is not
measured. Validation SHALL state that ridge width is not measured rather than substitute a number
for it.

#### Scenario: A closed mesh with creases is reported as watertight

- **WHEN** printability is validated for a mesh that is closed and manifold but contains sharp
  creases, such as a relief with pronounced tips
- **THEN** it is reported as watertight and manifold, with zero boundary edges and zero
  non-manifold edges

#### Scenario: Repair progress reflects the same measure

- **WHEN** mesh repair or hole filling reports how many boundary edges remain
- **THEN** the count is of boundary edges alone, so a successful repair is reported as successful

#### Scenario: Radial extent replaces wall thickness

- **WHEN** validation runs for a given target print size
- **THEN** the minimum and maximum radius from the origin are reported in millimetres, measured
  before centring, and no wall-thickness estimate or wall-thickness-based size recommendation is
  produced

#### Scenario: The unmeasured quantity is named

- **WHEN** the validation report is printed verbosely
- **THEN** it states that ridge width between adjacent features is not measured

### Requirement: Sized STL file output

The library SHALL write the ornament to an STL file centered at the origin and uniformly scaled so
its largest dimension equals a requested size in millimeters, in binary or ASCII form. Before
writing, the library SHALL compute consistent, outward-oriented point and cell normals, so the
facet normals recorded in the STL are meaningful to consumers that read them.

#### Scenario: Export at a target size

- **WHEN** the ornament is saved with a target size in millimeters
- **THEN** the mesh is centered, scaled so its maximum dimension equals that size, and written to
  the STL path, creating the output directory if needed

#### Scenario: One-call generate-and-save

- **WHEN** the combined generate-and-save entry point (or the `create_ornament` convenience
  function) is called
- **THEN** the mesh is generated and then exported in a single step, returning the saved file path

#### Scenario: Facet normals are oriented

- **WHEN** an ornament is saved
- **THEN** its normals are consistent and point outward

## ADDED Requirements

### Requirement: Ornaments are normalized by default

The relief's sea level sits at `|f| = 1` for every self-dual transfer, so the arbitrary constant in
front of a function changes the shape of the ornament rather than only its labels. The library
SHALL therefore rescale `|f|` by a normalization constant before applying the modulus transfer, by
default.

The default constant SHALL place the area-weighted geometric mean of `|f|` over the sphere at 1.
This is self-dual: the constant for `1/f` is the reciprocal of the constant for `f`, so the relief
of `1/f` is the relief of `f` turned inside out. A caller SHALL be able to select the area-weighted
**median** of `log|f|` instead — also self-dual, more robust where a high-order feature at infinity
occupies a large share of the sphere and drags sea level away from the interesting structure — or
supply the constant as an explicit number, or disable normalization entirely.

The constant SHALL be estimated from the sphere samples the generator has already evaluated,
without a second pass over the function, and SHALL NOT be derived from a function preset's
singularity answer key, because those keys are illustrative rather than complete divisors.

#### Scenario: A rescaled function produces the same ornament

- **WHEN** ornaments are generated for `f` and for `c * f` with default normalization and any
  non-zero constant `c`
- **THEN** the two reliefs have the same geometry to within sampling error

#### Scenario: Normalization is self-dual

- **WHEN** the normalization constant is computed for `f` and for `1/f`
- **THEN** the two constants are reciprocal

#### Scenario: A caller overrides the constant

- **WHEN** an explicit numeric constant is supplied
- **THEN** that constant is applied and no estimate is computed

#### Scenario: A caller disables normalization

- **WHEN** normalization is disabled
- **THEN** `|f|` is passed to the transfer unchanged, reproducing the pre-normalization mapping

#### Scenario: The raw modulus is still reported

- **WHEN** an ornament mesh is generated with normalization active
- **THEN** the mesh's magnitude scalar is the raw `|f|`, the radius scalar reflects the normalized
  value, and the applied constant is recorded in the mesh metadata

### Requirement: Ornament tip shape is tunable

Under a logistic transfer in the log-modulus, the surface approaches a feature of order `mu` like
`distance` raised to the power `mu / k`, so the ratio `mu / k` — rather than `k` — determines
whether a feature reads as a rounded dome, an exact cone, or a cusp. The library SHALL expose this
as a `pointiness` parameter giving a tip exponent of `1 / pointiness`, with larger values producing
sharper tips.

`pointiness` SHALL set the logistic's scale as `k = pointiness * pole_order`, where `pole_order`
defaults to 1 and MAY be supplied by the caller, so that a higher-order feature prints as sharp as
a simple one rather than proportionally blunter. The default ornament transfer SHALL be one that
admits such a scale; the bounded-arctangent transfer does not, because its shape is fixed at 90% of
its height at `|f| = 6.3` regardless of the function.

The default relief depth SHALL match the depth used by the library's named scaling presets rather
than the shallower value previously used only by STL export.

A caller SHALL also be able to set the log-modulus scale directly, bypassing
`pointiness * pole_order`. Setting it to `ln(10)` makes one unit of relief exactly one decade of
gain — 20 dB — which turns a transfer function's relief into a calibrated scale that can be read
like a Bode magnitude curve along a meridian. That is not expressible as a tip exponent.

#### Scenario: Pointiness sharpens tips

- **WHEN** two ornaments are generated for the same function differing only in `pointiness`
- **THEN** the one with the larger value approaches its maximum radius over a smaller
  neighbourhood of the feature

#### Scenario: Feature order is compensated

- **WHEN** ornaments are generated for a simple pole and for a double pole at the same
  `pointiness`, with `pole_order` given for each
- **THEN** the two tips have the same approach exponent

#### Scenario: The default transfer accepts a scale

- **WHEN** an ornament is generated with default settings
- **THEN** the transfer applied is one parameterized by a log-modulus scale, so `pointiness` has an
  effect

#### Scenario: The scale can be set directly

- **WHEN** an explicit log-modulus scale is supplied
- **THEN** it is used as given and `pointiness` and `pole_order` are not consulted

### Requirement: Bulk contrast is separately adjustable

A single logistic makes one parameter do two jobs: lowering the scale adds contrast across the bulk
of the surface but blunts the tips, while raising it sharpens the tips and squeezes the body toward
mid-radius. On a piece with few features the result is a near-perfect sphere with features poked
into it. The library SHALL therefore offer a two-scale transfer, mixing a narrow logistic for bulk
contrast with the original wide one for the tips, under a `contrast` parameter taking a boost and a
weight.

The mixed transfer SHALL remain odd in `log|f|` about zero, so normalization stays self-dual and a
zero of order `k` carves the mirror image of what a pole of order `k` raises. The mixed transfer
SHALL be smooth at `|f| = 1`; a transfer that is merely monotone and odd is not sufficient, because
one with unbounded derivative at sea level creases the surface along every sea-level contour and
prevents the mesh seam from welding.

`contrast` SHALL default to off. Applied globally it degrades pieces whose tips are the point of
the piece, so it is a per-piece choice rather than a default.

#### Scenario: Contrast sculpts an otherwise smooth body

- **WHEN** ornaments are generated for a function with few, widely separated features, with and
  without contrast at the same pointiness
- **THEN** the one with contrast spreads more of its surface area away from mid-radius

#### Scenario: The mixture stays self-dual

- **WHEN** the mixed transfer is applied to a function and to its reciprocal
- **THEN** the two reliefs are reflections of one another about sea level

#### Scenario: Contrast is off by default

- **WHEN** an ornament is generated without requesting contrast
- **THEN** a single logistic is applied
