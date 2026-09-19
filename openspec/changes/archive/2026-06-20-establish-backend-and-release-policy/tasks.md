# Tasks — establish-backend-and-release-policy

## 1. Backend policy document
- [x] 1.1 Add `docs/development/backend-policy.md`: the 2D=matplotlib / 3D=PyVista
      capability line, the rationale (matplotlib 3D quality/maintenance), and the 3.0
      migration plan (PyVista required, mpl-3D removed).
- [x] 1.2 Add a short pointer to the policy from `README.md` and the 3D usage docs
      (`docs/pyvista_usage_guide.md`).

## 2. Deprecate matplotlib 3D entry points
- [x] 2.1 Emit a `DeprecationWarning` from `plot_landscape` naming `plot_landscape_pv` as
      the replacement and stating removal at 3.0. → shared `_warn_mpl3d_deprecated` helper.
- [x] 2.2 Same for `pair_plot_landscape` → `pair_plot_landscape_pv`.
- [x] 2.3 Same for the 3D `riemann()` surface → `riemann_pv`.
- [x] 2.4 Confirm `riemann_chart` / `riemann_hemispheres` (2D stereographic) are NOT
      touched. → they live in `plot_2d.py`, unmodified; covered by a regression test.
- [x] 2.5 Add docstring "Deprecated since 2.1; removed in 3.0" notes to the three
      functions. → `.. deprecated:: 2.1` notes added.

## 3. Dependency ergonomics
- [x] 3.1 Add a `[3d]` extra to `pyproject.toml` that aliases `[pyvista]`
      (`["complexplorer[pyvista]"]`).
- [x] 3.2 Document in the install docs that `[3d]`/`[pyvista]` is the 3D backend and that
      new 3D features are PyVista-only; PyVista becomes required at 3.0.

## 4. Tests
- [x] 4.1 Add tests asserting each deprecated entry point raises `DeprecationWarning`
      (`pytest.warns(DeprecationWarning)`). → `tests/.../test_deprecation.py` (5 tests,
      incl. the negative case for the 2D charts).
- [x] 4.2 Update/adjust existing tests that call the deprecated functions so the new
      warning does not fail the suite. → no-op: `pytest.ini` has no `error` filter, so the
      warnings are captured, not fatal.
- [x] 4.3 Run `pytest tests/` green. → 344 passed, 2 skipped.

## 5. Close out
- [x] 5.1 Update `openspec/ROADMAP.md`: flip this change's STATUS to `in-progress`, then
      `archived` when synced.
