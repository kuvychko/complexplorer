# Design — PyVista surface kernel

## Context

This captures the design worked out for Phase 1. It is grounded in the current code, not
the idealized API in `complexplorer_phased_implementation_plan.md` §5 — which over-states
the duplication in one place and under-states two inconsistencies.

### What is actually true today

Two mesh topologies, four entry points:

| Entry point | Topology | Builds | Scaling semantics |
|---|---|---|---|
| `plot_landscape_pv`, `pair_plot_landscape_pv` (via `create_complex_surface`) | planar grid | `StructuredGrid`, `X=Re, Y=Im, Z=height` | `height = scale(\|f\|)`, `z_max` clip, `log_z` |
| `riemann_pv` | distorted sphere | sphere → radial distort | `radius = scale(\|f\|)`, `r_min/r_max` |
| `OrnamentGenerator` (STL) | distorted sphere | sphere → radial distort | same |

Already shared (the plan's "viz vs STL fork" fear is half-wrong):
- `utils/mesh_distortion.py::apply_modulus_distortion` — the single radial-scaling kernel,
  used by **both** `riemann_pv` and `OrnamentGenerator`.
- STL post-processing (`scale_to_size`, `center_mesh`, `validate_printability`,
  `repair_mesh_simple`) — topology-agnostic; operates on any finished `PolyData`.

The real duplication is narrow: each of the three builders re-does, inline:
1. evaluate `f`, compute `|f|` and `arg f`;
2. `cmap.rgb(f)` → `"RGB"`;
3. attach `"magnitude"` / `"phase"` scalars.

### Two latent inconsistencies (must be pinned, or unification silently changes output)

1. **Projection mirror.** `riemann_pv` projects from the *south* pole
   (`sphere_to_complex(..., from_north=False)`, `z=0` at top). `OrnamentGenerator`
   projects from the *north* (`compute_riemann_sphere_distortion(..., from_north=True)`).
   The printed ornament is an upside-down mirror of the rendered sphere.
2. **Duplicate function.** `core.functions.inverse_stereographic` and a different
   `utils.mesh.inverse_stereographic` (aliased to `complex_to_sphere`) coexist with
   different signatures; the two sphere paths use different ones.

## Decisions

### D1 — Scope: `ComplexField` + `SurfaceMesh` (not thin, not the full quartet)

Introduce a lightweight `ComplexField` AND `SurfaceMesh`. Defer `FieldOnSurface` /
`RiemannSurface` to Phase 4. Rationale: the field seam is exactly what Phase 2
presets/gallery, Phase 4 surfaces, and Phase 5 EE all consume; building it now avoids a
retrofit, while `FieldOnSurface` would be guessed-at blind before the surface work.

### D2 — Unify the projection on the visualization convention; flip STL to match

Standardize all relief on the convention `riemann_pv` currently uses; the STL ornament
orientation flips to match the rendered sphere. Rationale: the roadmap's "math → matter"
thesis means **the object you print should match the sphere you saw**. Viz is the
more-seen, screenshotted artifact; the STL is regenerated on demand, so the shift is
cheap. Collapse the two `inverse_stereographic` into one while here. This is the **only**
intentional behavior change in the refactor; everything else is output-preserving.

### D3 — Facade the five public functions, output-preserving, regression-locked

`plot_landscape_pv`, `pair_plot_landscape_pv`, `riemann_pv`, `OrnamentGenerator`,
`create_ornament` keep their signatures and become thin facades over the kernel. Write
output-pinning regression tests for all five **before** refactoring; the STL orientation
flip (D2) is the one expected diff.

## The seam

```
   sample()  ──▶  ComplexField           z (or sphere params), w=f(z),
   (core/field.py)                       |w|, arg w, mask, metadata
                       │
          ┌────────────┴────────────┐
   build_landscape            build_relief        (mesh/builders.py)
   (height = scale|f|)    (radius = scale|f|)      ← only topology-specific code
          └────────────┬────────────┘
                  SurfaceMesh                       (mesh/surface.py)
                  PolyData + .attach_colors() + metadata
                       │
        ┌──────────────┼───────────────┐
   .to_pyvista()   .save_stl()    .screenshot()     ← shared; already topology-agnostic
```

Outcome: the 3× decorate-block collapses to one `SurfaceMesh.attach_colors()`; STL
post-processing becomes reachable from any `SurfaceMesh` (landscapes can export STL for
free); `apply_modulus_distortion` + `ModulusScaling` remain the geometry kernel unchanged.

## Proposed module layout

```
core/field.py            ComplexField, sample()          numpy-only, no PyVista import
mesh/__init__.py
mesh/surface.py          SurfaceMesh
mesh/builders.py         build_landscape, build_relief
mesh/sphere.py           (absorbed RectangularSphereGenerator)
mesh/distortion.py       (re-homed apply_modulus_distortion — unchanged math)
mesh/repair.py, mesh/io.py   (re-homed STL post-proc: scale/center/validate/repair/save)
```

`SurfaceMesh` wraps a `pv.DataSet`; `.save_stl()` triangulates (`extract_surface` /
`triangulate`) before writing. `ComplexField` must not import PyVista (keeps 2D/core
importable without the 3D backend until 3.0).

## Open questions

- Exact home of the STL post-proc (`export/stl/` vs `mesh/io.py`). Leaning `mesh/` so it
  serves any surface, with `export/stl/` keeping the user-facing `OrnamentGenerator`
  facade. Resolve during implementation.
- Whether `pair_plot_landscape_pv`'s two-viewport plumbing stays in the facade or moves
  into a small `SurfaceMesh` multi-mesh plot helper. Facade is fine for now.

## Risks

| Risk | Mitigation |
|---|---|
| Refactor silently changes a rendered/exported output | Output-pinning regression tests for all 5 entry points before refactor (D3) |
| Projection flip surprises existing STL users | Documented intentional change; note in release notes + docstring |
| `mesh/` vs `utils/mesh.py` import churn | Single move commit; keep old import paths as shims if anything external depends on them |
