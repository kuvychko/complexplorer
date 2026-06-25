# Require PyVista for the 3D backend

## Why

This is **the** breaking change that defines 3.0. The roadmap's architecture spine draws a
sharp 2D/3D line: matplotlib for 2D, PyVista for 3D. matplotlib's 3D rendering is low quality
and a maintenance drag, and its 3D paths (`plot_landscape`, `pair_plot_landscape`, the 3D
`riemann` surface) have **already emitted deprecation warnings since 2.1**
(`establish-backend-and-release-policy`). 3.0 completes that deprecation: the matplotlib 3D
paths are removed and **PyVista becomes a required dependency**, so the library has exactly
one, high-quality 3D backend. This is justified by — and released together with — the feature
that genuinely needs a real 3D backend, `add-riemann-surfaces` (a separate additive change;
see *Release strategy*).

## What changes (breaking)

- **PyVista is a core dependency** — `pyvista>=0.47` moves from `[project.optional-dependencies]`
  into `dependencies`. STL export (which needs PyVista) is therefore always available.
- **The matplotlib 3D functions are removed** — `plot_landscape`, `pair_plot_landscape`, and
  the 3D `riemann` surface (`plotting/matplotlib/plot_3d.py`) and their public exports.
  - **KEPT:** `riemann_chart` / `riemann_hemispheres` — those are *2D* stereographic charts
    (matplotlib), unaffected by the 3D policy.
- **The capability flags are removed** — `HAS_PYVISTA` and `HAS_STL_EXPORT` (always `True`
  once PyVista is required) are deleted, imports become unconditional, and the internal
  `if not HAS_PYVISTA: raise` guards (CLI, STL utils, PyVista utils, mesh) go away.
- **`quick_plot` loses its matplotlib 3D fallback** — 3D/Riemann modes always use PyVista;
  `backend="matplotlib"` for those modes raises a clear "removed in 3.0" error. 2D is
  unchanged.
- **CI collapses to a single full-install config** — the `base` (no-PyVista) lane tested an
  install that can no longer exist; the matrix becomes `os × python`.
- **Version → `3.0.0`** in `_version.py` (semver: this is the breaking change).

## Release strategy (bundled 3.0)

3.0 ships the removal **and** `add-riemann-surfaces` together (stick + carrot). This change
bumps `_version` to `3.0.0` because semver requires it, but the **PyPI 3.0.0 release is held**
until `add-riemann-surfaces` also lands — so in-tree the version honestly reads 3.0.0-dev
between the two changes, and the first published 3.0.0 carries the justifying feature.

## Non-goals

- Not `add-riemann-surfaces` — that's the paired additive feature, its own change.
- No change to 2D (matplotlib stays the 2D backend) or to the 2D stereographic charts.
- No new PyVista functionality — this is removal + dependency + housekeeping only.

## Impact

- **Specs:** retire `plotting-3d-mpl` (whole capability); `plotting-3d-pyvista` drops its
  "Optional-dependency gating" requirement (PyVista is standard now); `high-level-api`'s
  backend-dispatch requirement tightens (3D/Riemann are PyVista-only, no fallback).
- **Source:** `pyproject.toml`, `__init__.py`, `plotting/matplotlib/plot_3d.py`,
  `plotting/matplotlib/__init__.py`, `api.py`, `cli/main.py`, `export/stl/utils.py`,
  `plotting/pyvista/utils.py`, `utils/mesh.py`, `_version.py`.
- **Tests:** remove the mpl-3D suites (`test_plot_3d.py`, `test_plot_3d_modulus.py`,
  `test_deprecation.py`) and trim integration tests; convert `skipif(not HAS_PYVISTA)` to
  unconditional and delete the inverse `skipif(HAS_PYVISTA)` (now-impossible) tests.
- **Docs:** README / CLAUDE.md references to the removed functions and the optional-PyVista
  framing.
- **Breaking for users** who call the removed functions or check the flags — but the
  functions have warned since 2.1, and this is a major version bump.
