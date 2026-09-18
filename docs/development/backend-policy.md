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

- **2.1:** matplotlib 3D entry points — `plot_landscape`, `pair_plot_landscape`, and
  the 3D `riemann()` surface — emitted a `DeprecationWarning` pointing to their PyVista
  equivalents (`plot_landscape_pv`, `pair_plot_landscape_pv`, `riemann_pv`). PyVista
  remained optional, installable via `complexplorer[pyvista]` or its alias
  `complexplorer[3d]`. New 3D features were PyVista-only.
- **3.0 (current):** PyVista is a **required** dependency and the sole 3D backend. The matplotlib
  3D paths above are **removed**, along with the `HAS_PYVISTA` / `HAS_STL_EXPORT` capability flags
  that existed to guard them — those features are now always available. The `[pyvista]` and `[3d]`
  extras survive only as empty no-op aliases, so an existing
  `pip install complexplorer[pyvista]` keeps working; they install nothing. matplotlib remains the
  2D backend, including the 2D stereographic charts.

The 2D stereographic charts `riemann_chart` and `riemann_hemispheres` are matplotlib **2D**
features and are **not** affected by this policy — they are supported indefinitely.

## For contributors

- Add new 3D functionality to `complexplorer/plotting/pyvista/` (and the forthcoming 3D
  surface kernel), never to the matplotlib 3D modules.
- There is no matplotlib 3D module to extend; `plotting/matplotlib/` is 2D only.
- Do not reintroduce a capability flag or a `try: import pyvista` guard. PyVista is a hard
  dependency, so importing it unconditionally is correct, and a guard would recreate the
  two-backend branching this policy exists to remove.

## What requiring PyVista costs

The question of whether a required PyVista is too heavy was settled with measurements rather than
estimates, taken from the release artifact gate: a fresh virtual environment containing only the
built wheel and its runtime dependencies (Windows, CPython 3.12, complexplorer 3.0.0). The same
`artifact` job prints its own Linux equivalents on every CI run — the same measurements, taken the
same way — so these can be re-checked rather than trusted.

| Measurement | Value |
|---|---|
| Wheel | 92 KB |
| Installed `site-packages`, everything | 533 MB |
| VTK (`vtk.libs` + `vtkmodules`) | 296 MB — 56% of the install |
| scipy | 87 MB |
| matplotlib + numpy + their bundled libraries | 74 MB |
| pyvista itself | 14 MB |
| Cold `import complexplorer` | 0.54 s |
| Cold `import pyvista` alone | 0.27 s |
| Cold `import matplotlib.pyplot` alone | 0.38 s |

`import complexplorer` loads pyvista, 32 `vtkmodules` submodules and `matplotlib.pyplot` eagerly.

**The decision: PyVista stays required.** The install is dominated by VTK, and that is real — but
the alternative is worse in the ways that matter here. Making it optional means every 3D entry
point, the STL export and the CLI acquire an import guard and a second failure mode ("installed,
but the interesting half does nothing"), and the capability flags come back. The library's stated
purpose includes 3D landscapes, Riemann surfaces and 3D-printable ornaments; an install that cannot
do those is not a smaller complexplorer, it is a different one. Half a second of import time is not
the constraint, and anyone who wants only 2D phase portraits is well served by matplotlib directly.

This is revisited only if the numbers change materially — a VTK that ships slimmer wheels, or an
import cost that grows past a second or two.
