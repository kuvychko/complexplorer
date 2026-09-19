# Tasks — require-pyvista-3d-backend

## 1. Dependency
- [x] 1.1 `pyproject.toml`: move `pyvista>=0.47` from `[project.optional-dependencies]` into
      core `dependencies`. Keep `[pyvista]` / `[3d]` as empty no-op aliases (back-compat for
      existing install commands).

## 2. Remove matplotlib 3D
- [x] 2.1 DELETE the whole file `plotting/matplotlib/plot_3d.py` (it is 100% 3D:
      `_warn_mpl3d_deprecated`, class `Matplotlib3DPlotter`, `plot_landscape`,
      `pair_plot_landscape`, `riemann`). Do NOT touch `utils/mesh_distortion`
      (`get_default_scaling_params`) — it is shared by STL / mesh / PyVista, not dead.
- [x] 2.2 `plotting/matplotlib/__init__.py`: remove `from .plot_3d import *` and drop
      `"plot_3d"` from `__all__`. `complexplorer/__init__.py`: remove `plot_landscape` /
      `pair_plot_landscape` / `riemann` from imports + `__all__`. KEEP `riemann_chart` /
      `riemann_hemispheres` (they live in `plot_2d.py`).

## 3. Remove the capability flags
- [x] 3.1 `__init__.py`: delete `HAS_PYVISTA` / `HAS_STL_EXPORT`; make the STL + PyVista
      imports unconditional; remove from `__all__`; flatten the `if HAS_*` export blocks.
- [x] 3.2 `cli/main.py`: remove `from .. import HAS_PYVISTA` (now broken) + `_need_pyvista()` +
      its call sites (pyvista is always present). SCOPE NOTE: the module-LOCAL pyvista
      detection in `export/stl/utils.py`, `plotting/pyvista/utils.py`, `utils/mesh.py` feeding
      `check_pyvista_available()` (14 call sites) is independent defensive infrastructure, not
      the public package flags — left intact to avoid disproportionate churn in this breaking
      change.

## 4. quick_plot fallback
- [x] 4.1 `api.py`: drop the `plot_3d_landscape` / `plot_riemann` imports and the `use_pyvista`
      flag; 3D/Riemann always dispatch to PyVista; `backend="matplotlib"` for 3D/Riemann raises
      a clear "removed in 3.0" `ValidationError`. 2D unchanged.

## 5. CI
- [x] 5.1 Collapse the workflow matrix `config: [base, pyvista]` to a single full-install
      config (`os × python`); remove the base-only install step; keep the Linux
      headless-display action, the Windows offscreen skip, and the 3.13 wheel-lag guard.

## 6. Tests
- [x] 6.1 Delete `tests/unit/plotting/matplotlib/test_plot_3d.py`, `test_plot_3d_modulus.py`,
      `test_deprecation.py`.
- [x] 6.2 Trim mpl-3D calls from `test_basic_workflows.py` / `test_end_to_end.py`.
- [x] 6.3 Convert every `@skipif(not HAS_PYVISTA)` to unconditional; delete the inverse
      `@skipif(HAS_PYVISTA, "is installed")` tests (`test_utils.py`, `test_mesh.py`) and the
      `test_api_quick_plot` HAS_PYVISTA-agreement regression.
- [x] 6.4 Add a test: `quick_plot(mode="3d", backend="matplotlib")` raises the 3.0 error.

## 7. Version & docs
- [x] 7.1 `_version.py` → `3.0.0` (PyPI release held for `add-riemann-surfaces`).
- [x] 7.2 Update README / CLAUDE.md: drop removed functions and the optional-PyVista framing.

## 8. Close out
- [x] 8.1 `pytest tests/` green; `ruff` clean; `openspec validate --specs`.
- [x] 8.2 Update `openspec/ROADMAP.md` (status; note PyPI release bundled with riemann surfaces).
