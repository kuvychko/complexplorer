# Backend policy: matplotlib for 2D, PyVista for 3D

Complexplorer draws a **sharp capability line by dimensionality** and does not maintain
feature parity across it.

| Visualization | Backend |
|---|---|
| 2D phase portraits, pair plots, static educational figures | **matplotlib** |
| 2D stereographic charts (`riemann_chart`, `riemann_hemispheres`) | **matplotlib** |
| 3D analytic landscapes | **PyVista** |
| Riemann relief / Riemann sphere | **PyVista** |
| Riemann surfaces, multi-sheet geometry (future) | **PyVista** |
| STL / mesh export | **PyVista** |
| High-quality screenshots / movies | **PyVista** |

## Why

matplotlib is excellent for 2D and for publication-quality static figures. Its 3D engine,
however, is slow, low-quality, and a maintenance burden — advanced 3D is fundamentally a
mesh, camera, lighting, clipping, scalar-field, and export problem, which is PyVista's
domain. Trying to support advanced 3D equally in both backends would mean duplicated APIs,
weaker interactions, lower quality, more edge cases, and pressure to simplify the
mathematical design to fit the weaker backend.

> Use matplotlib where matplotlib is excellent; use PyVista where the problem is actually a
> 3D mesh/geometry problem.

## Migration plan

- **2.1 (now):** matplotlib 3D entry points — `plot_landscape`, `pair_plot_landscape`, and
  the 3D `riemann()` surface — emit a `DeprecationWarning` pointing to their PyVista
  equivalents (`plot_landscape_pv`, `pair_plot_landscape_pv`, `riemann_pv`). PyVista
  remains optional, installable via `complexplorer[pyvista]` or its alias
  `complexplorer[3d]`. New 3D features are PyVista-only.
- **3.0:** PyVista becomes a **required** dependency. The matplotlib 3D paths above are
  **removed**. matplotlib remains the 2D backend, including the 2D stereographic charts.

The 2D stereographic charts `riemann_chart` and `riemann_hemispheres` are matplotlib **2D**
features and are **not** affected by this policy — they are supported indefinitely.

## For contributors

- Add new 3D functionality to `complexplorer/plotting/pyvista/` (and the forthcoming 3D
  surface kernel), never to the matplotlib 3D modules.
- Do not extend `complexplorer/plotting/matplotlib/plot_3d.py`; it is frozen pending
  removal at 3.0.
