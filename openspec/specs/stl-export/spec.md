# STL Export

## Purpose

The stl-export capability turns a complex function into a 3D-printable mathematical ornament: a
modulus-scaled Riemann sphere whose surface relief encodes `|f(z)|`, exported as an STL file sized
in millimeters and repaired for printability. It is built on PyVista and is the bridge from
mathematics to a physical object.
## Requirements
### Requirement: STL export is always available

STL export SHALL be available whenever `complexplorer` is importable, because PyVista is a
required dependency. The export modules SHALL import PyVista unconditionally and SHALL NOT define
or expose any PyVista-availability flag or gating check.

#### Scenario: Ornament export works without any availability guard

- **WHEN** an `OrnamentGenerator` is constructed and used in a normal installation
- **THEN** it generates and saves an STL mesh without consulting any capability flag, and no `HAS_PYVISTA` / `check_pyvista_available` symbol is importable from the library

### Requirement: Ornament mesh generation

The library SHALL generate a 3D mesh from a complex function by distorting a Riemann sphere
radially according to `|f(z)|` and attaching per-point color, magnitude, phase, and radius data.

#### Scenario: Generate the ornament mesh

- **WHEN** an ornament's mesh is generated for a function at a resolution and modulus mode
- **THEN** a sphere mesh is produced whose radius is scaled by the modulus mode applied to `|f(z)|`, carrying RGB color, magnitude, phase, and radius arrays

#### Scenario: Saving before generation is rejected

- **WHEN** validation or saving is requested before the mesh has been generated
- **THEN** an error is raised directing the caller to generate the mesh first

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

The library SHALL write the ornament to an STL file centered at the origin and uniformly scaled to a
requested size in millimeters, in binary or ASCII form. Before writing, the library SHALL compute
consistent, outward-oriented point and cell normals, so the facet normals recorded in the STL are
meaningful to consumers that read them.

The requested size SHALL by default mean the object's **true maximum width** — its tip-to-tip
extent, which for a convex body is the diameter of its convex hull — and SHALL NOT by default mean
the largest dimension of its axis-aligned bounding box.

The bounding box depends on how a piece happens to sit in the coordinate frame, so as a size measure
it is not a property of the object. A relief whose spikes point along the cube diagonals
`(±1, ±1, ±1)/sqrt(3)` projects each spike onto a coordinate axis at only `0.577` of its length, so
its box understates it by a factor of `sqrt(3)`. A collection sized by the box therefore comes out
visibly uneven while every piece reports the same nominal size. A caller SHALL still be able to
request bounding-box sizing explicitly, for compatibility and for the cases where the box is what
matters, such as fitting a build plate.

#### Scenario: Export at a target size

- **WHEN** the ornament is saved with a target size in millimeters
- **THEN** the mesh is scaled so its tip-to-tip extent equals that size, then centered, and written
  to the STL path, creating the output directory if needed

#### Scenario: Orientation does not change the size

- **WHEN** the same ornament is saved at one target size, and saved again after being rotated
  arbitrarily
- **THEN** both files describe an object of that same maximum width

#### Scenario: Bounding-box sizing remains available

- **WHEN** a caller explicitly requests sizing by the axis-aligned bounding box
- **THEN** the largest bounding-box dimension equals the requested size, as it did before

#### Scenario: One-call generate-and-save

- **WHEN** the combined generate-and-save entry point (or the `create_ornament` convenience
  function) is called
- **THEN** the mesh is generated and then exported in a single step, returning the saved file path

#### Scenario: Facet normals are oriented

- **WHEN** an ornament is saved
- **THEN** its normals are consistent and point outward

### Requirement: Non-destructive operations

Mesh operations that scale, center, or repair SHALL operate on copies so the generator's internal
mesh is not mutated by an export.

#### Scenario: Export leaves the source mesh intact

- **WHEN** an ornament is saved
- **THEN** the centering and scaling apply to a copy, and the generator's stored mesh is unchanged for reuse

### Requirement: Status output is encodable on any console

The progress and validation messages printed by STL export, mesh repair and printability
validation SHALL be ASCII. These messages are emitted by the library itself during
`generate_and_save`, so a character the console cannot encode aborts the export with
`UnicodeEncodeError` after the mesh has been built and before the file is written. Status markers
SHALL therefore be written as ASCII markers rather than as symbols.

#### Scenario: A verbose export on a legacy code page

- **WHEN** an ornament is generated and saved with verbose output on a console using a legacy code
  page
- **THEN** the repair and validation report is printed in full and the STL file is written

#### Scenario: Markers carry the same meaning

- **WHEN** the repair or validation report reports success, a warning or a failure
- **THEN** each is marked distinguishably in ASCII, so the report stays readable

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

The scale derived from `pointiness * pole_order` SHALL be capped, because the tip-exponent rule
assumes the mesh can deliver the dynamic range a given feature order demands, and beyond roughly
order three it cannot: a feature of order `mu` only drives `log|f|` as far as the nearest sample gets
to it, so a scale demanding saturation that the grid never reaches flattens the whole body instead of
sharpening the tip. The cap SHALL NOT apply to a scale the caller sets directly, which is an explicit
instruction rather than a derivation.

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
- **THEN** it is used as given, `pointiness` and `pole_order` are not consulted, and the cap on the
  derived scale does not apply

#### Scenario: A high feature order does not flatten the body

- **WHEN** an ornament is generated for a feature order high enough that
  `pointiness * pole_order` would exceed what the mesh can resolve
- **THEN** the scale applied is capped, and the relief spans more of its radial range than it would
  at the uncapped scale

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
