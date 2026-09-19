# Design — add-algebraic-curves

## Context

`mesh/riemann_surface.py` parametrizes each family's *cover* as an ordinary rectangular grid
feeding the shared `SurfaceMesh` kernel; `riemann_surface_pv` dispatches on `family`. For
`w² = P(z)` no global inverse parametrization exists (unlike `z = w^n`), so the cover must be
assembled from sheets.

## Goals / Non-Goals

**Goals**: faithful two-sheeted surface of `w² = P(z)` for polynomial `P`, cuts/monodromy
emergent, additive to the existing families, branch points exposed as metadata.

**Non-Goals**: general algebraic curves `w^n = P(z)` or `F(z, w) = 0` (per the project
principle: support `sqrt`, `z^(1/n)`, `w² = P(z)` very well rather than a grand API that fails
on hard cases); catalog preset / showcase / gallery integration (would churn the byte-stable
`index.json` pre-release; own change later); STL export (self-intersecting, non-manifold —
excluded for all Riemann surfaces by the existing spec).

## Decisions

1. **Sheets as graphs of `±Re(√P(z))`, colored by `phase(±√P(z))`.** Key observation:
   `Re(√ζ) = √((|ζ|+Re ζ)/2)` is continuous in `ζ`, so each sheet's *geometry* is a continuous
   graph over the disk — no spurious vertical walls at the principal-branch cut of `√`. The
   two graphs intersect exactly where `Re(√P) = 0`, i.e. along `{z : P(z) ≤ 0}` — curves
   joining the branch points. The *color* (phase) does jump across those curves on each sheet,
   which is the honest picture: following the surface across a cut continues onto the other
   sheet where the phase continues smoothly. This mirrors how the `n`-sheeted power surface
   shows its self-intersection. Alternative considered: cut-aware sheet gluing with explicit
   seams — far more complex meshing for the same visual, rejected.
2. **Polar disk grid in `z`** (`rho ∈ [0, r_max]`, `phi ∈ [0, 2π]` inclusive endpoint so the
   seam closes), same construction and cell-shape conventions as the power family.
3. **Two `StructuredGrid`s, colors attached per sheet, then merged** with
   `merge(merge_points=False)`. Colors must be attached *before* merging because
   `SurfaceMesh.attach_colors` requires the value array's ravel order to match the mesh point
   order — which no longer holds after merge. `merge_points=False` keeps the sheets'
   intersection an emergent crossing rather than welding it.
4. **`p` in `numpy.polyval` order** (highest degree first) — matches the numpy convention
   users already know; validated as sequence, length ≥ 2, `p[0] != 0`.
5. **Branch points via `numpy.roots(p)`** recorded in metadata (values only; multiplicity
   analysis deferred until a consumer needs it).

## Risks / Trade-offs

- [Roots of `P` outside `r_max` make the visible surface look cut-free] → documented: choose
  `r_max` to enclose the interesting branch points; metadata carries the roots so callers can
  check.
- [Perfect-square `P` (reducible curve) yields two tangent/crossing planes rather than a
  connected surface] → mathematically correct behavior; not an error.

## Migration Plan

Additive; single commit; rollback = revert.

## Open Questions

_None._
