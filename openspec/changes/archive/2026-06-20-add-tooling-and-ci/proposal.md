# Add tooling and CI

## Why

The roadmap's Phase 0 calls for linting and a CI matrix, but no change owned it — and it is
a prerequisite, not a nicety. `add-pyvista-surface-kernel` introduces output-pinning
regression tests for all five PyVista entry points; those can only run in an environment
with PyVista installed and offscreen rendering configured. Without a CI lane that installs
PyVista and runs headless, the kernel refactor has no automated safety net. This change
establishes that lane (and basic lint/format) so later phases land on green CI.

## What changes

- Add `ruff` (lint + format) configuration and bring the tree to a clean baseline.
- Add a `uv`-based GitHub Actions workflow (matching the project's local toolchain) running,
  on push/PR: lint, then tests on **Linux + Windows** in **two configurations** — base (no
  PyVista) and with PyVista (`complexplorer[pyvista]`, `PYVISTA_OFF_SCREEN`). The base lane
  passes by skipping PyVista tests (they use `pytest.importorskip`). Dependencies install
  from `pyproject` ranges (not `uv.lock`) so CI catches upstream breakage.
- Optionally add a lightweight type check (`mypy`/`pyright`) — non-blocking to start.

## Non-goals

- Version/license reconciliation (`reconcile-versioning-and-license`).
- Backend policy declaration (`establish-backend-and-release-policy`).
- Enforcing 100% type coverage — start advisory.

## Impact

- New: `.github/workflows/ci.yml`, `[tool.ruff]` config (and `ruff` in the dev extra).
- Touches: formatting across the tree (ruff format), `pyproject.toml` dev dependencies.
- Affected specs (additive): `packaging` gains CI and lint requirements.
- Risk: low. Two watch-items: the first `ruff format` pass produces a large diff (land it
  as its own commit), and `vtk` may lack 3.13 wheels (PyVista lane treats 3.13 as
  `continue-on-error`).
