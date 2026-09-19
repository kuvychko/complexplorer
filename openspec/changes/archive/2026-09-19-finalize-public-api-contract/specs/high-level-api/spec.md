## MODIFIED Requirements

### Requirement: Named configuration presets

The library SHALL provide named presets that return bundled colormap and resolution settings for
common rendering goals, usable as keyword arguments to the plotting entry points. They SHALL be
exposed as `PlotPresets`, a name that states what they configure and cannot be confused with the
function registry `catalog`. Each SHALL declare its return type.

#### Scenario: Presets return ready-to-use settings

- **WHEN** the publication, interactive, or high-contrast preset is requested from `PlotPresets`
- **THEN** a settings bundle is returned with a configured `Phase` colormap and a resolution tuned for that goal (higher resolution and more phase sectors for publication, balanced for interactive, many sectors and tighter modulus scaling for high contrast)

#### Scenario: The older name is gone

- **WHEN** `complexplorer.Presets` is accessed
- **THEN** it is not part of the public surface, because the published 2.x releases never exposed
  it and 3.0 is where the name is settled

## ADDED Requirements

### Requirement: The public typing contract describes what the library actually calls

The package ships `py.typed`, so its annotations are a promise to downstream type checkers. A
user-supplied function SHALL be typed by a `ComplexFunction` protocol describing what the
renderers actually pass: an array of complex values in, an array of complex values out, with
scalar input accepted. `Callable[[complex], complex]` SHALL NOT be used for such parameters,
because it describes a scalar contract the library does not use.

Every public callable SHALL declare an explicit return type.

#### Scenario: A vectorized function satisfies the annotation

- **WHEN** a user passes a function that maps an array of complex values to an array of complex
  values to a public entry point
- **THEN** a type checker accepts it

#### Scenario: A downstream program type-checks

- **WHEN** a small program that imports complexplorer and uses its public entry points is checked
  with pyright
- **THEN** the check passes, and it is run in CI so it keeps passing
