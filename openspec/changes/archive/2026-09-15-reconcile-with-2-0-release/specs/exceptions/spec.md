## ADDED Requirements

### Requirement: Colormap configuration errors

The library SHALL provide `complexplorer.exceptions.ColormapError`. It is raised for invalid
colormap configuration and is a subclass of `ValidationError`, and therefore of both
`ComplexplorerError` and `ValueError`. It SHALL be exported from the top-level `complexplorer`
package, as it was in 2.0.0.

#### Scenario: ColormapError is a ValidationError

- **WHEN** a `ColormapError` is raised
- **THEN** `except ValidationError`, `except ComplexplorerError`, and `except ValueError` each catch it

#### Scenario: ColormapError is a top-level export

- **WHEN** a user imports `ColormapError` from `complexplorer`
- **THEN** the import succeeds and `ColormapError` appears in `complexplorer.__all__`
