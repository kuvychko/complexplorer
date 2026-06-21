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
- Add a GitHub Actions workflow running, on push/PR: lint, then tests in **two
  configurations** — base (no PyVista) and with PyVista (`complexplorer[pyvista]`), the
  latter with offscreen rendering enabled (e.g. `PYVISTA_OFF_SCREEN`, a virtual
  framebuffer if needed).
- Optionally add a lightweight type check (`mypy`/`pyright`) — non-blocking to start.

## Non-goals

- Version/license reconciliation (`reconcile-versioning-and-license`).
- Backend policy declaration (`establish-backend-and-release-policy`).
- Enforcing 100% type coverage — start advisory.

## Impact

- New: `.github/workflows/ci.yml`, `[tool.ruff]` config (and dev extras additions).
- Touches: formatting across the tree (ruff format), `pyproject.toml` dev dependencies.
- Affected specs (additive): `packaging` gains CI and lint requirements.
- Risk: low, but the first `ruff format` pass produces a large diff — land it as its own
  commit, separate from logic changes.
