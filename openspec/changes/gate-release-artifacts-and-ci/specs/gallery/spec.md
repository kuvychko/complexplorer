## MODIFIED Requirements

### Requirement: Manifests are deterministic and byte-stable

`index.json` and every `card.json` SHALL be written deterministically (sorted keys, fixed indent,
LF newlines, trailing newline) and SHALL be byte-identical for the same selection and library
version **on any platform**, not only across repeated runs on one machine.

Floating-point values in a preset record are derived from libm (roots of unity, cube roots,
multiples of pi), which disagrees by one unit in the last place between platforms. Records SHALL
therefore be quantized to a fixed number of significant digits when serialized, so that such a
difference cannot reach the manifest. The quantization applies to the serialized record only; the
live preset attributes SHALL keep full precision.

#### Scenario: Manifests are byte-identical across runs

- **WHEN** `generate_gallery` runs twice for the same selection and library version into two
  directories
- **THEN** `index.json` and every `card.json` are byte-for-byte identical between the two runs

#### Scenario: Manifests are byte-identical across platforms

- **WHEN** the committed `index.json` is compared against one freshly generated on a different
  operating system with the same library version
- **THEN** the two are byte-for-byte identical, and a one-unit-in-the-last-place difference in any
  derived coordinate does not change the serialized record

#### Scenario: The manifest does not depend on render settings

- **WHEN** `generate_gallery` runs at different `dpi` or `resolution` settings
- **THEN** `index.json` is unchanged, because the manifest describes the catalog rather than the
  rendered pixels
