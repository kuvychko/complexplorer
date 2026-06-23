# Tasks — fix-quick-plot-backend

## 1. Fix quick_plot
- [x] 1.1 In `api.quick_plot`, `kwargs.pop("backend", None)` up front (never forward it).
- [x] 1.2 For `mode in {3d, riemann}`, default to PyVista when `HAS_PYVISTA` and
      `backend != "matplotlib"`; otherwise call the (deprecated) matplotlib renderer.
- [x] 1.3 Confirm `mode="2d"` is unchanged.

## 2. Tests
- [x] 2.1 Audit existing `quick_plot` / api tests for an assumption that the default 3D path
      returns matplotlib; update them (request `backend="matplotlib"` where a matplotlib
      result is intended, or assert the PyVista result).
- [x] 2.2 New tests (PyVista-gated): `quick_plot(f, mode="riemann", modulus_mode="arctan",
      interactive=False, return_plotter=True)` succeeds (was `TypeError`); `backend="pyvista"`
      does not leak into the renderer.
- [x] 2.3 With `HAS_PYVISTA` patched False, `quick_plot(mode="3d")` falls back to the
      matplotlib renderer (and emits its deprecation warning).

## 3. Close out
- [x] 3.1 Run `pytest tests/` green; `ruff` clean; `openspec validate --specs`.
- [x] 3.2 Update `openspec/ROADMAP.md` (note the fix).
