# Design — Riemann surfaces

## The core idea: parametrize the cover, not the z-plane

A Riemann surface is multivalued over `z` but **single-valued over a different base**. Flip
which variable is sampled, and the surface becomes an ordinary rectangular grid that maps
straight onto `SurfaceMesh`.

```
  power  w = z^(1/n)  ⟺  z = w^n
     sample w on a (polar) grid over a disk of radius R = r_max^(1/n)
     z = w^n           (sweeping w once covers the n sheets over |z| ≤ r_max)
     point = ( Re z, Im z, Re w )      ← honest height (see height note below)
     as arg(w): 0→2π,  arg(z): 0→2πn   → self-intersects along the cut (emergent)

  log  w = log z = ln r + iθ
     sample (r, θ) grid,  r ∈ (0, R],  θ ∈ [0, 2π·turns]
     point = ( r cosθ, r sinθ, θ )     ← θ = Im(w) = honest height
     z = r e^{iθ} repeats every 2π but the height climbs → helicoid (no self-intersection)
```

Both reduce to **rectangular parameter grid → embedded points → `SurfaceMesh`**: one builder,
two parametrization functions, no mode branching (honest only).

## Pipeline (reuses the kernel)

```
build_riemann_surface(family, params) -> SurfaceMesh
   1. parameter grid (2D, structured) for the family
   2. embed → X, Y, Z arrays  →  pv.StructuredGrid(X, Y, Z)
   3. SurfaceMesh(grid)                                   # kernel, unchanged
   4. .attach_colors(cmap, w)                             # w = the value array; phase coloring
   return it

riemann_surface_pv(family, *, n=2 | turns=3, r_max=1.5, resolution=..., cmap=Phase(),
                   interactive, notebook, camera_position, window_size, title,
                   filename, return_plotter, **kwargs) -> pv.Plotter | None
   build_riemann_surface(...) → render with the same plotter plumbing as riemann_pv
```

`SurfaceMesh.__init__` takes a `pv.DataSet` (StructuredGrid/PolyData), and
`attach_colors(cmap, w)` writes `RGB` + `phase` — exactly the shape produced here. No kernel
change.

## Decisions

| Question | Decision | Why |
|---|---|---|
| Embedding | **honest only** | faithful textbook picture; stacked dropped for tightness |
| Power height | **`Re(w)`** | spike-confirmed: `Re w` puts the self-intersection on the **negative** real axis = the conventional principal-branch cut; `Im w` would put it on the positive axis (unconventional). Fixed, not a parameter, for tightness. |
| Log height | `Im(w) = θ` | the helicoid height is the argument — conventional and the only sensible choice |
| Extent | `r_max` (radius in the `z`-plane) | power samples `w` over radius `r_max^(1/n)`; log over the `z`-disk of radius `r_max` |
| Coloring | phase of the **value** `w` (`Phase` cmap) | domain-coloring *on the surface*; consistent with the library |
| Grid for power | **polar in `w`** (ρ, φ) | natural disk sampling; concentrates detail near the branch point `w=0` |
| Grid for log | `(r, θ)` rectangular | the natural helicoid chart |
| `ComplexField`? | **bypass it** | it models "f over plane/sphere"; a cover is a different object — build `SurfaceMesh` directly |
| STL | **non-goal** | honest power self-intersects → non-manifold → unprintable |

## Family model

A small internal descriptor per family — enough for v1's two, extensible later:

```
power(n, r_max):   R = r_max**(1/n);  grid(ρ∈[0,R], φ∈[0,2π])  (φ endpoint included → seam closes)
                   w = ρ e^{iφ};  z = w**n;  height = w.real;  value = w
log(turns, r_max): grid(r∈(0,r_max], θ∈[0,2π·turns]) → w = ln r + iθ; z = r e^{iθ}; height = θ; value = w
```

Spike-validated (n=2,3): the `φ` seam closes (first/last column coincide), `z == w**n`, there
are exactly `n` distinct sheet-heights over a generic `z`, the branch point (`w=0`) is finite,
and the log helicoid spans `[0, 2π·turns]` and is `2π`-periodic in `(x,y)`.

`n` (power) corresponds to the catalog `branch_point` order (sqrt=2, cbrt=3); `log` is the
order-∞ case. The API is standalone (`family=` + params), not catalog-driven, because a
preset's callable is single-valued — but the answer-key orders line up for future linkage.

## Spike (de-risk before building)

Build the `n=2` power grid and the `turns=2` log grid in a throwaway script and confirm:
- the power mesh is **continuous** across `arg(w)=0↔2π` (closes up, self-intersection is a
  crossing not a tear), and the sheet count matches `n`;
- phase coloring is continuous on each sheet and wraps correctly;
- the log mesh is a clean helicoid over the requested turns;
- vertices near the branch point (`w=0` / `r→0`) are well-behaved (no NaN/degenerate faces).

## Tests

- `build_riemann_surface("power", n=2/3)`: point/face counts match the grid; `z ≈ w**n` at
  sample points; height `== Im(w)`; mesh closes across the seam (first/last `φ` column
  coincide or connect).
- `build_riemann_surface("log", turns=k)`: height spans `[0, 2πk]`; `(x,y)` periodic in `θ`.
- Colors: `attach_colors` populates `RGB`/`phase`; finite and in `[0,1]` (the colormap
  guarantee from `fix-colormap-nonfinite`).
- `riemann_surface_pv(..., return_plotter=True, interactive=False)` returns a plotter without
  rendering (PyVista-gated; Windows-CI offscreen-safe, mirroring existing tests).

## Risks

| Risk | Mitigation |
|---|---|
| Seam/closure of the power surface | spike + a closure test on the `φ` seam |
| Detail loss / degeneracy at the branch point | polar grid concentrates points near `w=0`; guard `r→0` for log |
| Phase coloring discontinuity across sheets | color by `arg(w)` (continuous in `w`); validated in the spike |
| Real-render crash on headless Windows CI | use `return_plotter` (no screenshot), as in `add-cli` |
