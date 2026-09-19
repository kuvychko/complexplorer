# Add ComplexplorerError Exception Hierarchy

## Why

CLAUDE.md and `openspec/config.yaml` both document a `ComplexplorerError` hierarchy in
`complexplorer/exceptions.py` — but no such module exists. The library's real error type is
`ValidationError(ValueError)` in `utils/validation.py`, plus seven stray bare `ValueError`
raises. 3.0.0 is the right moment to make the documented contract true: introducing a common
base class later would churn `except` clauses again, while doing it now (with dual
`ValueError` inheritance) is fully backward-compatible.

## What Changes

- New module `complexplorer/exceptions.py` defining `ComplexplorerError(Exception)` — the
  base class for all library-domain errors.
- `ValidationError` is re-parented to `class ValidationError(ComplexplorerError, ValueError)`
  and moves to `exceptions.py`; `utils/validation.py` re-imports it so every existing
  `from complexplorer.utils.validation import ValidationError` import keeps working.
- The seven bare `raise ValueError(...)` sites in library code (`api.py`, `core/scaling.py`,
  `mesh/surface.py`, `export/stl/utils.py`, `export/stl/ornament_generator.py` ×2,
  `plotting/pyvista/utils.py`) become `ValidationError`, so `except ComplexplorerError`
  catches everything the library raises deliberately. (Genuine `ImportError`/`SystemExit`
  sites are left untouched.)
- `ComplexplorerError` and `ValidationError` are exported at package top level (`__all__`).
- Not breaking: `ValidationError` remains a `ValueError` subclass, and every converted site
  previously raised `ValueError`, so all existing `except ValueError` / `except
  ValidationError` code continues to work.

## Capabilities

### New Capabilities

- `exceptions`: the library-wide error contract — a `ComplexplorerError` base class, a
  `ValidationError` that is both a `ComplexplorerError` and a `ValueError`, and the guarantee
  that deliberate library errors derive from the base class.

### Modified Capabilities

_None — no existing capability's requirements change; this adds a cross-cutting contract._

## Impact

- New: `complexplorer/exceptions.py`; new spec `openspec/specs/exceptions/spec.md`.
- Modified: `utils/validation.py` (definition moves, import-path preserved),
  `complexplorer/__init__.py` (exports), the seven `ValueError` sites listed above,
  `CHANGELOG.md` (Added entry).
- Tests: new unit test for the hierarchy; existing tests unaffected (subclass relationships
  preserved).
