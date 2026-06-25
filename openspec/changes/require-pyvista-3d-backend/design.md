# Design — require PyVista for the 3D backend

## Removal surface (grounded)

```
DEPS    pyproject.toml: pyvista>=0.47  optional → core `dependencies`
        [pyvista]/[3d] extras → keep as empty no-op aliases (back-compat for
        `pip install complexplorer[pyvista]`) OR drop. Decision: keep as empty
        aliases so existing install commands don't error.

REMOVE  plotting/matplotlib/plot_3d.py:
          plot_landscape, pair_plot_landscape, riemann  (+ the _warn_mpl3d_deprecated
          helper and the deprecated wrapper class)  → delete the module's 3D content
        plotting/matplotlib/__init__.py: drops them from the star-export
        __init__.py: remove plot_landscape / pair_plot_landscape / riemann from
          imports and __all__
        KEEP riemann_chart / riemann_hemispheres (2D, live in plot_2d.py)

FLAGS   __init__.py: HAS_PYVISTA / HAS_STL_EXPORT deleted; the two try/except
          import blocks become unconditional `from … import …`; both removed
          from __all__; the `if HAS_*` conditional-export blocks become plain.
        Internal guards now dead (PyVista always present) — delete:
          cli/main.py    _need_pyvista() + its call sites
          export/stl/utils.py:21, plotting/pyvista/utils.py:20, utils/mesh.py:72
```

## api.quick_plot after removal

```python
backend = kwargs.pop("backend", None)
if mode == "2d":
    return plot_2d(domain, func, **kwargs)            # unchanged
# 3D / Riemann: PyVista only (it's a required dependency now)
if backend == "matplotlib":
    raise ValidationError(
        "The matplotlib 3D backend was removed in 3.0; 3D/Riemann use PyVista."
    )
if mode == "3d":
    return plot_landscape_pv(domain, func, **kwargs)
return riemann_pv(func, **kwargs)                      # mode == "riemann"
```

No more `use_pyvista` flag, no `plot_3d_landscape`/`plot_riemann` imports. The lazy PyVista
imports become unconditional (they always succeed).

## Spec deltas — three capabilities

| Capability | Delta | What |
|---|---|---|
| `plotting-3d-mpl` | **retire** | Remove all four requirements; the capability spec file is deleted on archive. |
| `plotting-3d-pyvista` | **REMOVE** one requirement | "Optional-dependency gating" (missing-PyVista error + "imports without PyVista" + flag) describes an impossible state now. Purpose text drops "optional"/"degrades cleanly" on sync. |
| `high-level-api` | **MODIFY** | "Mode- and backend-dispatching plot": 3D/Riemann are **PyVista-only**; the matplotlib-fallback scenario is replaced with "matplotlib 3D was removed; requesting it errors". 2D unchanged. |

## Version & release

`_version.py` → `3.0.0` here (semver). The **PyPI 3.0.0 release waits** for
`add-riemann-surfaces` (the bundled-3.0 decision). Between the two changes, in-tree is
3.0.0-dev — unreleased, so a breaking change at that version is fine (PyPI is the contract,
per `reconcile-versioning`).

## CI

Matrix `config: [base, pyvista]` → single config (PyVista always installed). Keep the
headless-display action for the now-always-present 3D tests on Linux; keep the Windows-CI
offscreen skip from `add-cli`. Drop the `base`-only install step.

## Test migration

- **Delete** `tests/unit/plotting/matplotlib/test_plot_3d.py`, `test_plot_3d_modulus.py`,
  `test_deprecation.py` (the functions and their warnings are gone).
- **Trim** `test_basic_workflows.py` / `test_end_to_end.py` of mpl-3D calls (e.g. the
  `quick_plot(mode="3d", backend="matplotlib")` path → now an error test).
- **Convert** every `@skipif(not HAS_PYVISTA)` → unconditional (PyVista is always there).
- **Delete** the inverse `@skipif(HAS_PYVISTA, "is installed")` tests in `test_utils.py` and
  `test_mesh.py` (they assert behavior that can no longer occur), and the
  `test_api_quick_plot` HAS_PYVISTA-agreement regression (flag gone).
- **Add** a test: `quick_plot(mode="3d", backend="matplotlib")` raises the clear 3.0 error.

## Risks

| Risk | Mitigation |
|---|---|
| Users calling removed functions | Deprecated since 2.1; major bump; clear errors. |
| Removing flags breaks downstream `HAS_PYVISTA` checks | Documented breaking change; the flags were always-true post-requirement anyway. |
| Bigger blast radius (many files) | Mechanical; full suite + the collapsed CI verify. |
| 3.13 PyVista/VTK wheel lag (the old `continue-on-error`) | Keep that guard on the single lane until wheels are confirmed. |
