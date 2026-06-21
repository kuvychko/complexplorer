# Tasks — add-tooling-and-ci

## 1. Linting and formatting
- [ ] 1.1 Add `[tool.ruff]` config (line length, target version, rule selection) to
      `pyproject.toml`; add `ruff` to the `dev` extra.
- [ ] 1.2 Run `ruff check --fix` and `ruff format`; land the format pass as its own commit
      (large, mechanical diff kept separate from logic).
- [ ] 1.3 Resolve or explicitly ignore remaining lint findings.

## 2. CI workflow
- [ ] 2.1 Add `.github/workflows/ci.yml` triggered on push and pull_request.
- [ ] 2.2 Job: lint (`ruff check`, `ruff format --check`).
- [ ] 2.3 Job: tests in the **base** configuration (no PyVista) — confirms the 2D/core
      paths import and pass without the 3D backend.
- [ ] 2.4 Job: tests **with PyVista** (`pip install -e ".[pyvista,dev]"`), offscreen
      rendering enabled (`PYVISTA_OFF_SCREEN=true`; add a virtual framebuffer / `xvfb` on
      Linux if required) — this is the lane the surface-kernel regression tests need.
- [ ] 2.5 Run across supported Python versions (3.11–3.13).

## 3. Optional type checking
- [ ] 3.1 Add `mypy` or `pyright` config; run advisory (non-blocking) in CI.

## 4. Close out
- [ ] 4.1 Confirm CI is green on a trial PR.
- [ ] 4.2 Update `openspec/ROADMAP.md` STATUS for this change.
