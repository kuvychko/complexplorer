# Exceptions

## Purpose

The exceptions capability is the library-wide error contract: a single `ComplexplorerError`
base class from which every error the library raises deliberately derives, a
`ValidationError` that is simultaneously a `ComplexplorerError` and a `ValueError` so that
handlers written against earlier releases keep working, and `ColormapError` for invalid
colormap configuration. It lets callers wrap any
complexplorer call in one `except ComplexplorerError` handler without enumerating
lower-level types.

## Requirements

### Requirement: Common exception base class

The library SHALL provide `complexplorer.exceptions.ComplexplorerError`, an `Exception`
subclass serving as the base class for all errors the library raises deliberately, and SHALL
export it (together with `ValidationError`) from the top-level `complexplorer` package.

#### Scenario: Library errors are catchable via the base class

- **WHEN** any deliberate library error is raised (invalid argument, unknown mode/preset, malformed expression, invalid STL parameters, mesh/RGB shape mismatch)
- **THEN** `except ComplexplorerError` catches it

#### Scenario: Top-level export

- **WHEN** a user imports `ComplexplorerError` or `ValidationError` from `complexplorer`
- **THEN** the import succeeds and both names appear in `complexplorer.__all__`

### Requirement: Backward-compatible ValidationError

`ValidationError` SHALL subclass both `ComplexplorerError` and `ValueError`, and SHALL remain
importable from its historical location `complexplorer.utils.validation`.

#### Scenario: Existing ValueError handlers keep working

- **WHEN** code written against 2.x catches `ValueError` around a call that raises `ValidationError`
- **THEN** the exception is caught exactly as before

#### Scenario: Historical import path preserved

- **WHEN** `from complexplorer.utils.validation import ValidationError` is executed
- **THEN** it imports the same class as `complexplorer.exceptions.ValidationError`

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
