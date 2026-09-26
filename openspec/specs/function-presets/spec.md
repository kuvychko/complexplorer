# Function Presets

## Purpose

The function-presets capability provides a catalog of curated, serializable complex
functions for reuse across rendering, export, and prototyping. Each preset bundles a
renderable callable, a human-readable expression, plain serializable domain/colormap/
scaling specs, hand-authored exact singularity answer keys, and descriptive metadata. The
capability also provides spec factories that reconstruct live `Domain` and `Colormap`
objects without modifying core classes, plus a registry for retrieving and filtering
presets. It is defined independently of the optional PyVista backend so presets are usable
as data in any environment.
## Requirements
### Requirement: Serializable function preset

The library SHALL provide a `FunctionPreset` describing a complex function for reuse across
rendering, export, and game prototyping. It SHALL carry a renderable `callable` and a
human/`expression` string, serializable `domain_spec` / `cmap_spec` / `scaling_spec`
dicts, a hand-authored `singularities` list, and `id` / `title` / `story` / `tags`
metadata. It SHALL be defined in a module that does not require PyVista.

#### Scenario: Preset exposes both callable and expression

- **WHEN** a preset is retrieved
- **THEN** it provides a callable that evaluates the function and an `expression` string
  describing the same function (e.g. `"z / (z**10 - 1)"`)

#### Scenario: Preset specs are plain serializable dicts

- **WHEN** a preset's domain, colormap, and scaling are inspected
- **THEN** each is a plain dict whose keys mirror the target constructor's kwargs (e.g.
  `domain_spec={"type": "annulus", "inner_radius": 0.2, "outer_radius": 3}`), not a live
  object, and contains no PyVista-dependent data

#### Scenario: Complex values are encoded as real pairs

- **WHEN** a spec or singularity record holds a complex value (a domain `center`, a
  singularity `at`)
- **THEN** it is encoded as an `[real, imag]` pair so the record is JSON-serializable, and
  the factories convert it back to a complex value

#### Scenario: Presets are importable without the 3D backend

- **WHEN** the preset module is imported in an environment without PyVista
- **THEN** the import succeeds and presets are usable as data

### Requirement: Hand-authored exact singularity answer keys

A preset's `singularities` SHALL be a list of exact, author-provided records, one record
**per location**, each with a `type` in `{zero, pole, essential, branch_point}`, a location
`at` as a `[real, imag]` pair, an `order`, and an optional `label`. `order` is the
multiplicity for zero/pole, the branching order for branch_point, and null for essential
singularities. These SHALL be ground-truth answer keys, not values produced by a numerical
detector.

Where a preset's divisor is a geometrically defined set — the vertices, face centres or edge
midpoints of a polyhedron — the records MAY be **constructed from that geometry** rather than
transcribed as decimals. This is ground truth in the sense the requirement protects: the divisor is
exact and named ("the twenty vertices of the dodecahedron"), and the construction is a documented,
reproducible step, not an analysis of the preset's callable.

Such records SHALL NOT be obtained by numerically solving the polynomial whose roots they are.
Records are serialized into a manifest that is byte-compared across platforms after quantization to a
fixed number of significant digits, and the roots of a high-degree polynomial do not agree to that
many digits between linear-algebra implementations, so a solved key would make the manifest
platform-dependent. Projected geometry depends only on elementary functions, which agree to within
one unit in the last place and therefore survive the quantization.

The constructed locations SHALL be pinned by a check against the polynomial in the other direction:
the monic polynomial over the recorded locations SHALL reproduce the stated exact coefficients. The
count and multiplicities SHALL match the solid.

A singularity that lies **at infinity** SHALL NOT be recorded, because a location is a finite
`[real, imag]` pair. A preset whose function has a zero or pole at infinity SHALL say so in prose
rather than leave the reader to infer that the key is complete.

#### Scenario: Singularity records are structured and exact

- **WHEN** a preset's singularities are read
- **THEN** each record has a `type`, an `at` `[real, imag]` pair, and an `order` (or null),
  describing the function's known zeros/poles/essential/branch points

#### Scenario: A geometric divisor is constructed rather than transcribed or solved

- **WHEN** a preset's zeros or poles are the vertices, face centres or edge midpoints of a polyhedron
- **THEN** its records are constructed from that geometry, with the expected count and multiplicity,
  and the monic polynomial over them reproduces the exact coefficients of the corresponding form

#### Scenario: Serialized keys do not depend on the platform's linear algebra

- **WHEN** a preset carrying such a key is serialized on two platforms
- **THEN** the records are identical after the manifest's quantization

#### Scenario: A feature at infinity is described rather than recorded

- **WHEN** a preset's function has a zero or pole at infinity
- **THEN** no record claims a finite location for it, and the preset's prose states that it is there

### Requirement: JSON-ready serialization

A preset SHALL serialize via `to_dict()` to a JSON-compatible record containing every field
except the live `callable`. The `domain_spec` and `cmap_spec` SHALL reconstruct equivalent
live objects through the provided factories.

#### Scenario: to_dict omits the callable and is JSON-serializable

- **WHEN** `preset.to_dict()` is called
- **THEN** the result contains the expression, specs, singularities, and metadata, excludes
  the callable, and is serializable to JSON

#### Scenario: Specs round-trip through the factories

- **WHEN** `domain_from_spec(preset.domain_spec)` and `cmap_from_spec(preset.cmap_spec)`
  are called
- **THEN** they return live `Domain` and `Colormap` objects equivalent to the preset's
  intended domain and colormap

### Requirement: Spec factories build live objects without modifying core classes

The library SHALL provide `domain_from_spec` and `cmap_from_spec` factories that map a
spec dict's `type` to the corresponding `Domain` / `Colormap` subclass and instantiate it.
The factories SHALL cover the subclasses used by the curated presets and SHALL raise the
library's domain exception (`ValidationError`) for an unknown `type`. The core `Domain` /
`Colormap` classes SHALL NOT be modified.

#### Scenario: Unknown spec type is rejected

- **WHEN** a spec with an unrecognized `type` is passed to a factory
- **THEN** a `ValidationError` is raised naming the unsupported type and the supported ones

### Requirement: Registry access and tag filtering

The library SHALL provide a registry to retrieve presets by `id`, list available presets,
and filter presets by `tag`. It SHALL ship a curated set of canonical presets, each with
its singularity answer keys, recommended specs, story, and tags.

#### Scenario: Retrieve a preset by id

- **WHEN** `catalog.get("pole_flower_10")` is called
- **THEN** the corresponding `FunctionPreset` is returned

#### Scenario: Filter presets by tag

- **WHEN** `catalog.filter(tag="singularity-detective")` is called
- **THEN** every returned preset carries that tag

#### Scenario: Missing id is reported clearly

- **WHEN** `catalog.get` is called with an unknown id
- **THEN** a clear error is raised (or `None`/sentinel per the documented contract), not a
  silent wrong result

### Requirement: Derived answer-key statistics

A preset SHALL expose derived geometry computed from its hand-authored `singularities`,
via an `answer_key_stats()` method, and SHALL include the same record in `to_dict()`. The
statistics are a pure function of the authored answer key (never computed by analyzing the
callable) and SHALL contain:

- `count` — the total number of singularity records,
- `count_by_type` — a mapping of singularity type to count,
- `min_separation` — the smallest Euclidean distance, in the z-plane, between any two
  singularity locations, or `null` when the preset has fewer than two singularities.

#### Scenario: Stats summarize the answer key

- **WHEN** `preset.answer_key_stats()` is called for a preset with multiple singularities
- **THEN** `count` equals the number of records, `count_by_type` tallies the records by `type`, and `min_separation` is the distance between the closest pair of singularity locations

#### Scenario: Separation is null without a pair

- **WHEN** a preset has zero or one singularity
- **THEN** `min_separation` is `null`

#### Scenario: Stats are part of the serialized record

- **WHEN** `preset.to_dict()` is called
- **THEN** the result includes an `answer_key_stats` record (count, count_by_type, min_separation) and remains JSON-serializable

### Requirement: The interchange shapes are declared

`domain_spec`, `cmap_spec` and `scaling_spec`, and the singularity records, SHALL have declared
shapes that a type checker can verify, rather than being bare `dict`. The declarations SHALL
describe the same structure the manifest serializes, so the interchange record and the in-memory
record cannot drift apart silently.

#### Scenario: A malformed spec is visible to a type checker

- **WHEN** a preset is constructed with a spec dictionary whose keys do not match the declared
  shape
- **THEN** a type checker reports it

### Requirement: Polyhedral relative invariants

The library SHALL provide the Klein relative invariants of the tetrahedral, octahedral and
icosahedral groups as callables: the tetrahedral vertex form and its antipodal dual, the octahedral
vertex, face and edge forms, and the icosahedral vertex, Hessian and edge forms.

These exist so that a relief carrying a polyhedral symmetry can be built from named, verified forms.
A rational map of degree `d` has exactly `d` zeros and `d` poles on the sphere, so a function with
features at one solid's vertices and nowhere else does not exist; the symmetric reliefs are ratios of
relative invariants of **equal binary-form degree**, where the automorphy factors cancel and `|f|`
descends to a genuine invariant function on the sphere. The documentation SHALL state that
requirement, because a ratio of unequal degree silently fails to be invariant.

The correctness of the icosahedral trio SHALL be pinned by Klein's syzygy relating the three forms,
and the invariance of each documented ratio SHALL be pinned by a rotation of the sphere that belongs
to the group.

Transcription is the failure mode being guarded against, and it is silent. The forms quoted in the
literature cohere as a set only for one orientation of the solid, and the commonly-remembered signs
mix orientations: a mixed set yields three root sets that are each individually a perfect
icosahedron, dodecahedron and edge set while being rotated relative to one another, whereupon the
ratios stop being rotation-invariant by orders of magnitude. Checking each form's roots against its
solid does not detect this; the syzygy does.

#### Scenario: The icosahedral forms satisfy Klein's syzygy

- **WHEN** the three icosahedral forms are evaluated at arbitrary points
- **THEN** the syzygy relating the Hessian, the edge form and the vertex form holds to numerical
  precision

#### Scenario: A mixed-orientation set is rejected

- **WHEN** an icosahedral vertex form of the opposite sign convention is substituted
- **THEN** the syzygy fails by orders of magnitude, so the check distinguishes the two

#### Scenario: Each documented ratio is rotation invariant

- **WHEN** a documented invariant ratio is evaluated at a point and at that point rotated by an
  element of the corresponding rotation group
- **THEN** the modulus of the ratio is unchanged to numerical precision

#### Scenario: Root sets match their solids

- **WHEN** a form's roots are computed
- **THEN** their number matches the solid's feature count, allowing for the root at infinity where
  the form's polynomial degree is one short of its binary degree

### Requirement: A preset may carry the relief parameters it needs

A preset SHALL be able to record the feature order of its singularities and a rendering resolution,
as plain serializable data alongside its other specs. Consumers that build a relief from a preset
SHALL use them when present, and an explicit argument SHALL take precedence over them.

A printable preset that does not carry its feature order is rendered as though its features were
simple, which produces a blunter shape than intended: the tip exponent is the feature order divided
by the transfer's scale, and that scale is derived from the order. A preset whose features are dense
also needs a finer mesh than a general-purpose default, and raising that default globally would slow
every unrelated render instead.

#### Scenario: A preset's feature order reaches the transfer

- **WHEN** an ornament is built from a preset that records a feature order
- **THEN** the transfer's scale reflects that order, so the tip exponent matches the order-one case

#### Scenario: A preset's resolution is used

- **WHEN** an ornament is built from a preset that records a resolution, without an explicit
  resolution argument
- **THEN** the preset's resolution is used

#### Scenario: An explicit argument wins

- **WHEN** a caller passes a resolution or feature order explicitly
- **THEN** that value is used rather than the preset's

### Requirement: The polyhedral ornament family is in the catalog

The catalog SHALL include the polyhedral reliefs built from the invariants above: a tetrahedral pair,
two octahedral/cube pieces, two icosahedral/dodecahedral pieces, and the icosidodecahedral piece.
Each SHALL carry the metadata every preset carries — a parseable `expression`, serializable specs, a
divisor, a story and tags — plus its feature order and resolution.

Each SHALL also record, in prose, the symmetry group it carries, because that is what makes the piece
what it is and it is not derivable from the expression by inspection.

#### Scenario: A polyhedral preset is one call

- **WHEN** a polyhedral preset is retrieved from the catalog by id
- **THEN** it provides a callable, a parseable expression equal to that callable's function, its
  divisor, its feature order and its resolution

#### Scenario: The family is discoverable as a group

- **WHEN** the catalog is filtered by the tag these presets share
- **THEN** the polyhedral family is returned

#### Scenario: Expressions evaluate to the same function as the callable

- **WHEN** a polyhedral preset's `expression` is evaluated over a domain where it is finite
- **THEN** it agrees with the preset's callable
