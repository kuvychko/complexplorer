# Tasks — add-exception-hierarchy

## 1. Core implementation

- [x] 1.1 Create `complexplorer/exceptions.py` with `ComplexplorerError(Exception)` and
      `ValidationError(ComplexplorerError, ValueError)` (docstrings per project style)
- [x] 1.2 Re-export `ValidationError` from `complexplorer/utils/validation.py` (remove local
      definition, import from `exceptions`)
- [x] 1.3 Export `ComplexplorerError` and `ValidationError` from `complexplorer/__init__.py`
      (`__all__`)

## 2. Convert bare ValueError sites

- [x] 2.1 `api.py` (unknown mode), `core/scaling.py` (unknown preset), `mesh/surface.py`
      (RGB row mismatch), `export/stl/utils.py` (invalid axis),
      `export/stl/ornament_generator.py` ×2 (no mesh generated),
      `plotting/pyvista/utils.py` (export format) → `ValidationError`

## 3. Tests and docs

- [x] 3.1 New `tests/unit/test_exceptions.py`: hierarchy relationships, top-level exports,
      historical import path, converted sites catchable via `ComplexplorerError`
- [x] 3.2 `CHANGELOG.md`: Added entry for the exception hierarchy
- [x] 3.3 `pytest tests/` green; ruff clean; `openspec validate --specs` passes
