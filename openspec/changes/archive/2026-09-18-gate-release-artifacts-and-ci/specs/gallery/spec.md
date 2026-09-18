## MODIFIED Requirements

### Requirement: The manifest is a deterministic, self-contained contract

`index.json` SHALL contain a schema version, the complexplorer version, a generator marker,
and the full record of every rendered preset (each equal to its `card.json`), with records
sorted by id. The manifest SHALL contain no timestamps. Per-preset `card.json` SHALL equal
`preset.to_dict()` plus a `files` mapping. All file references SHALL be relative to the
bundle root, so the bundle is relocatable.

Manifests SHALL be byte-identical for the same selection and library version **on any platform**,
not only across repeated runs on one machine. Floating-point values in a preset record are derived
from libm (roots of unity, cube roots, multiples of pi), which disagrees by one unit in the last
place between platforms, so records SHALL be quantized to a fixed number of significant digits when
serialized. The quantization applies to the serialized record only; live preset attributes SHALL
keep full precision. The manifest SHALL NOT depend on render settings such as `dpi` or
`resolution`, because it describes the catalog rather than the rendered pixels.

#### Scenario: index.json is self-contained and relocatable

- **WHEN** `index.json` is read
- **THEN** it lists every rendered preset's full record inline (id, title, expression, tags, domain/cmap/scaling specs, singularities, and a `files` mapping with relative paths) under a schema version and the complexplorer version

#### Scenario: Manifests are byte-identical across runs

- **WHEN** `generate_gallery` runs twice for the same selection and library version into two directories
- **THEN** `index.json` and every `card.json` are byte-for-byte identical between the two runs

#### Scenario: Manifests are byte-identical across platforms

- **WHEN** the committed `index.json` is compared against one freshly generated on a different
  operating system with the same library version
- **THEN** the two are byte-for-byte identical, and a one-unit-in-the-last-place difference in any
  derived coordinate does not change the serialized record

#### Scenario: The manifest does not depend on render settings

- **WHEN** `generate_gallery` runs at different `dpi` or `resolution` settings
- **THEN** `index.json` is unchanged
