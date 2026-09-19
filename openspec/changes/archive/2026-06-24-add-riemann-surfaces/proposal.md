# Add Riemann surfaces

## Why

This is the **feature that justifies the 3.0 breaking change** — the visualization that
genuinely needs a real 3D backend — and the last thing gating the published 3.0.0 release
(bundled with `require-pyvista-3d-backend`).

A *Riemann surface* is the multi-sheeted cover on which a multivalued function (`√z`, `log z`)
becomes single-valued. This is distinct from the **Riemann sphere** (already shipped as
`riemann_pv`), which compactifies the plane for a *single*-valued function. The library
specializes in the visual grammar of complex functions — sheets, branch points, and cuts are
a core part of that grammar that is currently unrepresented.

The surface kernel (`SurfaceMesh`) built in Phase 1 is the right foundation: it is
parametrization-agnostic (points + faces + colors), so multivalued surfaces map onto it
without changing it.

## What changes

- **A new PyVista renderer** `riemann_surface_pv(family, …)` for the canonical multivalued
  families, with the standard wrapper options (`interactive`, `notebook`, `camera_position`,
  `window_size`, `title`, `filename`, `return_plotter`, `cmap`):
  - `family="power", n=2` → `z^(1/n)` (sqrt, cbrt, …): an `n`-sheeted surface, built by
    inverting to `z = w^n` and sampling `w`.
  - `family="log", turns=3` → `log z`: the helicoid, sampled over `(r, θ)` with
    `θ ∈ [0, 2π·turns]`.
- **A surface builder** `build_riemann_surface(family, params)` (in the mesh layer) that maps
  a rectangular *parameter* grid to embedded 3D points and returns a `SurfaceMesh`.
- **The "honest" embedding only** — the power surface uses height `Re(w)` (so its
  self-intersection falls on the negative real axis, the conventional principal-branch cut,
  per the spike); `log` uses height `Im(w) = θ`, the helicoid. The faithful picture: sheets
  and cuts are *emergent* from the geometry, not separate machinery.
- **Domain-colored by phase of the value** `w` via the existing `Phase` colormap (the shared
  `SurfaceMesh.attach_colors` path).
- A new `riemann-surfaces` capability spec. PyVista-only (3D — consistent with the 3.0
  backend policy).

## Non-goals

- **No `stacked`/separated-sheet embedding** — honest only (faithful self-intersecting
  geometry; the stylized stack is dropped).
- **No STL export** — the honest power surface self-intersects (non-manifold, unprintable);
  these are visualization objects. The "physical" pillar is served by the ornament generator.
- **No general algebraic curves `w² = P(z)`** — multi-branch-point gluing is research-grade
  and is the roadmap's 3.1+ backlog item.
- Not a new colormap or scaling mode; reuses `Phase` and the kernel.

## Impact

- New module(s): `mesh/riemann_surface.py` (builder) + `plotting/pyvista/riemann_surface.py`
  (wrapper); `riemann_surface_pv` exported from the package.
- Reuses unchanged: `SurfaceMesh`, `Phase` colormap, the PyVista plotter plumbing, grid/
  triangulation utilities. `ComplexField` is intentionally bypassed (a Riemann surface is an
  embedded cover, not "f sampled over a domain").
- New `riemann-surfaces` capability spec.
- Additive / non-breaking. Completes the 3.0 release set; after it lands, the PyPI 3.0.0 tag
  can be cut.
- Risk: moderate — the math (inversion `z = w^n`, the helicoid) is well understood and the
  kernel is proven; the main care is grid resolution near the branch point and correct phase
  coloring across sheets. A spike validates the parametrization before building.
