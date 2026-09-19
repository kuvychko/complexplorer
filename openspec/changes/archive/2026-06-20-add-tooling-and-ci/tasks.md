# Tasks — add-tooling-and-ci

## 1. Linting and formatting
- [x] 1.1 Add `ruff` to the `dev` extra and install it (not currently installed). Add a
      `[tool.ruff]` config to `pyproject.toml` (line length, `target-version = "py311"`,
      rule selection). → `ruff>=0.6`; select `E,F,I,UP,B`; ignore `E501,E731`;
      per-file-ignores for `__init__.py` (F401/F403/F405) and `tests/**` (E402/F841).
- [x] 1.2 Eyeball `ruff format --diff` to gauge size, then run `ruff check --fix` and
      `ruff format`; land the format pass as its **own commit** (large, mechanical diff
      kept separate from logic). This establishes a formatted baseline *before* the
      `add-pyvista-surface-kernel` refactor, so that PR diffs cleanly. → 53 files
      formatted (~6.9k lines), 320 lint fixes auto-applied.
- [x] 1.3 Resolve or explicitly `ignore` remaining lint findings (per-rule, with a reason).
      → ignores for E731 + init/test idioms; manually fixed 14 library findings (B028
      stacklevel ×5, B904 raise-from ×3, E722 bare-except, F841 unused ×3, F401 Qt
      probes via noqa). `ruff check` now passes clean; 339 tests still pass.

## 2. CI workflow (GitHub Actions — uv, Linux + Windows)
- [x] 2.1 Add `.github/workflows/ci.yml` on `push` + `pull_request`, with a `concurrency`
      group (cancel superseded runs) and pinned actions (`actions/checkout@v4`,
      `astral-sh/setup-uv@v6`, with uv caching enabled).
- [x] 2.2 Lint job (ubuntu only): `ruff check` + `ruff format --check`.
- [x] 2.3 Test matrix: OS {`ubuntu-latest`, `windows-latest`} × config {base, pyvista}.
      - **base:** `uv pip install -e ".[dev]"` (no PyVista) — confirms 2D/core paths pass
        without the 3D backend (the PyVista test modules skip via `pytest.importorskip`).
      - **pyvista:** `uv pip install -e ".[pyvista,dev]"` with `PYVISTA_OFF_SCREEN=true`.
        Mesh/STL tests need no display; for any real offscreen render test use
        `pyvista/setup-headless-display-action` (handles xvfb on Linux; Windows renders
        offscreen without it).
- [x] 2.4 Python versions: base lane **3.11–3.13**; PyVista lane **3.11–3.12**, with
      **3.13 as `continue-on-error`** until `vtk` 3.13 wheels are confirmed available
      (vtk historically lags new Python releases — do not let it block the matrix).
- [x] 2.5 Install from `pyproject.toml` dependency ranges, **not** `uv.lock`, so CI catches
      upstream dependency breakage (a library should test against unpinned ranges).

## 3. Type checking — deferred (decision)
- [x] 3.1 Type checking deferred. An advisory mypy run surfaced ~101 errors on the
      untyped, numpy/pyvista-heavy codebase — almost all noise (optional-import
      `module = None` patterns, missing third-party stubs, numpy generics). A perpetually
      red non-blocking job provides no signal, so the job + `[tool.mypy]` config were
      removed. Gradual typing should be a dedicated future change. The one-off run did
      surface two genuine bugs, now fixed: `dict[str, any]` → `dict[str, Any]`
      (`export/stl/utils.py`) and a no-op `compute_normals(consistency=...)` →
      `consistent_normals=...` (`export/stl/mesh_repair.py`).

## 4. Close out
- [x] 4.1 Confirm CI is green on a trial PR (note: base lane should show PyVista tests as
      skipped, pyvista lane as run). → **green**. Surfaced and fixed four real bugs en
      route (uv `--system`/PEP 668, `HAS_STL_EXPORT` detection, non-finite matplotlib
      facecolors, `consistent_normals` no-op).
- [x] 4.2 Update `openspec/ROADMAP.md` STATUS for this change.
