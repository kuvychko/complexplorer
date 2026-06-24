# Gallery

## ADDED Requirements

### Requirement: Gallery generation from a preset selection

The library SHALL provide a `generate_gallery` function that renders a selection of catalog
presets into an output directory as a self-describing asset bundle. The selection MAY be a
list of preset ids, a single tag, or all presets. Presets SHALL be processed in a stable
order (sorted by id), and the function SHALL return the manifest it wrote.

#### Scenario: A bundle is produced for a selection

- **WHEN** `generate_gallery` is called with a tag (or id list, or no selection) and an output directory
- **THEN** for each selected preset it writes `<id>/portrait.png` (a 2D phase portrait) and `<id>/card.json`, plus a top-level `index.json`, and returns the manifest

#### Scenario: Selection by tag matches the catalog filter

- **WHEN** a tag is given
- **THEN** exactly the presets the catalog returns for that tag are rendered, in id-sorted order

### Requirement: The manifest is a deterministic, self-contained contract

`index.json` SHALL contain a schema version, the complexplorer version, a generator marker,
and the full record of every rendered preset (each equal to its `card.json`), with records
sorted by id. The manifest SHALL contain no timestamps. Per-preset `card.json` SHALL equal
`preset.to_dict()` plus a `files` mapping. All file references SHALL be relative to the
bundle root, so the bundle is relocatable.

#### Scenario: index.json is self-contained and relocatable

- **WHEN** `index.json` is read
- **THEN** it lists every rendered preset's full record inline (id, title, expression, tags, domain/cmap/scaling specs, singularities, and a `files` mapping with relative paths) under a schema version and the complexplorer version

#### Scenario: Manifests are byte-identical across runs

- **WHEN** `generate_gallery` runs twice for the same selection and library version into two directories
- **THEN** `index.json` and every `card.json` are byte-for-byte identical between the two runs

### Requirement: Images are reproducible best-effort

Rendered `portrait.png` files SHALL be written with image metadata stripped (no software or
timestamp tags) so they are reproducible within an environment. Pixel bytes are NOT
guaranteed across environments or rendering-library versions; the byte-stable guarantee
applies to the manifest, not the images.

#### Scenario: Portrait metadata carries no timestamp

- **WHEN** a portrait PNG is written
- **THEN** it contains no embedded creation timestamp or software-version tag

### Requirement: Gallery is exposed as a CLI subcommand

The CLI SHALL expose a `gallery` subcommand that drives `generate_gallery`, accepting a tag
or explicit preset ids and an output directory.

#### Scenario: CLI generates a gallery

- **WHEN** `complexplorer gallery --tag <tag> -o <dir>` is run
- **THEN** the bundle is written under `<dir>` (including `index.json`) and the command exits 0

#### Scenario: CLI requires an output directory

- **WHEN** `complexplorer gallery` is run without an output directory
- **THEN** it exits with an error rather than writing anywhere implicit
