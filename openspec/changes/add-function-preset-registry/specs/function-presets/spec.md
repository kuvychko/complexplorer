# Function Presets

## ADDED Requirements

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

#### Scenario: Singularity records are structured and exact

- **WHEN** a preset's singularities are read
- **THEN** each record has a `type`, an `at` `[real, imag]` pair, and an `order` (or null),
  describing the function's known zeros/poles/essential/branch points

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
The factories SHALL cover the subclasses used by the curated presets and SHALL raise a
domain-specific error for an unknown `type`. The core `Domain` / `Colormap` classes SHALL
NOT be modified.

#### Scenario: Unknown spec type is rejected

- **WHEN** a spec with an unrecognized `type` is passed to a factory
- **THEN** a `ComplexplorerError` is raised naming the unsupported type

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
