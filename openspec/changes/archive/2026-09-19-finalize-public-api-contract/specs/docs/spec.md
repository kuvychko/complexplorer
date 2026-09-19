## ADDED Requirements

### Requirement: The site carries an API map

The documentation SHALL include a map of the public surface organised by task, giving for each
entry point what it returns — an `Axes`, a `Figure`, a `Plotter`, a mesh, a path, or nothing — so
a reader can find the right function and know what they get back without reading its signature.

Every name in `complexplorer.__all__` SHALL have a docstring.

#### Scenario: An undocumented public name fails the suite

- **WHEN** a name is added to `__all__` without a docstring
- **THEN** the test suite fails
