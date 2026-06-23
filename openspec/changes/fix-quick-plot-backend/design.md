# Design — fix quick_plot 3D backend

## The bug (current code)

```python
elif mode == "3d":
    if HAS_PYVISTA and kwargs.get("backend", "matplotlib") == "pyvista":  # (1) mpl default
        return plot_landscape_pv(domain, func, **kwargs)                   # (2) backend leaks
    else:
        return plot_3d_landscape(domain, func=func, **kwargs)   # deprecated mpl-3D
elif mode == "riemann":
    # same shape
```

1. Default backend is `"matplotlib"` → 3D/Riemann use the deprecated matplotlib renderer
   unless the caller explicitly passes `backend="pyvista"`. Against the backend policy, and
   the matplotlib `riemann()` doesn't accept `modulus_mode` (→ `TypeError`).
2. `backend` is never popped from `kwargs`, so it leaks into the renderer call.

## The fix

```python
backend = kwargs.pop("backend", None)          # never leak
...
elif mode == "3d":
    use_pv = HAS_PYVISTA and backend != "matplotlib"
    if use_pv:
        return plot_landscape_pv(domain, func, **kwargs)
    return plot_3d_landscape(domain, func=func, **kwargs)   # deprecated; emits its warning
elif mode == "riemann":
    use_pv = HAS_PYVISTA and backend != "matplotlib"
    if use_pv:
        return riemann_pv(func, **kwargs)
    return plot_riemann(func, **kwargs)
```

Selection table:

```
  backend arg   PyVista installed   →  renderer
  (none)        yes                    PyVista          (the policy default)
  (none)        no                     matplotlib (deprecated, warns)
  "pyvista"     yes                    PyVista
  "pyvista"     no                     matplotlib (deprecated, warns)   [best effort]
  "matplotlib"  any                    matplotlib (deprecated, warns)   [explicit opt-out]
```

## Behavior change + test scope

`quick_plot(mode="3d"|"riemann")` now returns a **PyVista plotter by default** (PyVista
installed) rather than a matplotlib axes. Implementation must:

- Check the existing `quick_plot`/api tests for assumptions that the default 3D path returns
  matplotlib; update them to request `backend="matplotlib"` where a matplotlib result is
  intended, or assert the PyVista result.
- Add a test: `quick_plot(f, mode="riemann", modulus_mode="arctan", interactive=False,
  return_plotter=True)` succeeds (previously `TypeError`); `backend="pyvista"` does not leak;
  with PyVista absent (patched `HAS_PYVISTA`) the matplotlib fallback is used.

## Risk

| Risk | Mitigation |
|---|---|
| Existing tests assume mpl default for 3D | Audit + update in this change |
| Behavior change surprises a caller | Documented; aligns with the backend policy; matplotlib still reachable via `backend="matplotlib"` until 3.0 |
