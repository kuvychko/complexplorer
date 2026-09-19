# Fix quick_plot 3D backend selection

## Why

`api.quick_plot` (the mode-dispatching high-level plot, also aliased `visualize`/`explore`)
has two bugs in its 3D/Riemann paths, discovered while building the CLI:

1. **It defaults 3D/Riemann to matplotlib** (`kwargs.get("backend", "matplotlib")`), the
   path that `establish-backend-and-release-policy` just **deprecated** (removed at 3.0).
   So `quick_plot(f, mode="riemann", modulus_mode="arctan")` silently hits the deprecated
   matplotlib `riemann()`, which doesn't accept `modulus_mode` → `TypeError`. This
   contradicts the backend policy (3D = PyVista).
2. **It leaks the `backend` kwarg.** `backend` is read with `.get` but never popped, so it
   rides along in `**kwargs` into the renderer (e.g. into the PyVista plotter, which does
   not accept it).

## What changes

In `quick_plot` only (the pattern is not repeated elsewhere):

- **Pop `backend`** so it never leaks into the renderer.
- **Default 3D/Riemann to PyVista when available** (per the backend policy). Use matplotlib
  only when PyVista is absent or `backend="matplotlib"` is explicitly requested — and that
  path is the deprecated matplotlib 3D renderer, which already emits its `DeprecationWarning`.

```python
backend = kwargs.pop("backend", None)
use_pv = HAS_PYVISTA and backend != "matplotlib"
return (plot_landscape_pv if use_pv else plot_3d_landscape)(...)
```

## Non-goals

- No change to 2D (`mode="2d"` is unaffected).
- No removal of the matplotlib 3D fallback — that happens at 3.0
  (`require-pyvista-3d-backend`); until then it remains as a deprecated fallback.
- Not the CLI — the CLI already dispatches directly to the PyVista functions and is
  unaffected.

## Impact

- Touched: `complexplorer/api.py` (`quick_plot`).
- Affected specs: `high-level-api` — the backend-selection scenario is tightened (PyVista is
  the default for 3D/Riemann when available, not only "when requested").
- **Behavior change:** `quick_plot(mode="3d"|"riemann")` now returns a PyVista plotter by
  default (when PyVista is installed) instead of a matplotlib axes. Existing tests/callers
  that assumed a matplotlib result for the default 3D path must be checked/updated.
- Risk: low, scoped to one function; covered by a new test (riemann + `modulus_mode` works;
  `backend` not leaked; matplotlib fallback when PyVista absent).
