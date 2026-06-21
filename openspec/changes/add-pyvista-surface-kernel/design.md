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

1. **Projection mirror — `riemann_pv` is the lone outlier.** The documented canonical
   convention is `core.functions.stereographic_projection`, whose default maps `z=0` to
   the **south** pole (`stereographic_projection(0+0j) == [0,0,-1]`, "for consistent
   zero/pole visualization"). Tracing all paths:
   - matplotlib `riemann()` — `stereographic_projection(project_from_north=False)` → `z=0`
     at south. **Follows core.**
   - STL `OrnamentGenerator` — `inverse_stereographic(project_from_north=True)`, denom
     `(1-z)` → `z=0` at south. **Follows core.**
   - `riemann_pv` — `utils.mesh.sphere_to_complex(from_north=False)`, denom `(1+z)` →
     `z=0` at **north**. **Diverges** from the core convention, the matplotlib sphere, and
     the STL ornament — it renders the sphere mirrored.
2. **Duplicate function.** `core.functions.inverse_stereographic` and a different
   `utils.mesh.inverse_stereographic` (aliased to `complex_to_sphere`) coexist with
   different signatures; `riemann_pv`'s divergence traces directly to using the latter.

## Decisions

### D1 — Scope: `ComplexField` + `SurfaceMesh` (not thin, not the full quartet)

Introduce a lightweight `ComplexField` AND `SurfaceMesh`. Defer `FieldOnSurface` /
`RiemannSurface` to Phase 4. Rationale: the field seam is exactly what Phase 2
presets/gallery, Phase 4 surfaces, and Phase 5 EE all consume; building it now avoids a
retrofit, while `FieldOnSurface` would be guessed-at blind before the surface work.

### D2 — Unify on the canonical core convention; fix `riemann_pv` (the outlier)

Standardize all relief on the documented core convention (`z=0` at the **south** pole),
which the matplotlib sphere and the STL ornament already follow. `riemann_pv` is the lone
divergent path, so **it is the one that changes** — its rendered sphere flips to agree
with the rest of the library. Rationale: this is a bugfix (the PyVista sphere has been
mirrored relative to core + matplotlib + STL), it honors the documented "consistent
zero/pole visualization" intent, and — importantly — **existing STL files stay valid**
(the ornament does not move). The roadmap's "math → matter" goal (print matches screen) is
satisfied either way once unified; converging on the majority/documented convention is the
lower-risk direction. Collapse the two `inverse_stereographic` into one canonical function
while here. The `riemann_pv` orientation flip is the **only** intentional behavior change;
everything else is output-preserving.

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

**Import-layering note (sphere sampling).** The sphere-sampling path in `sample()` must
stay pure-numpy: the θ/φ → Cartesian sphere coordinates are plain numpy, and the canonical
`stereographic_projection` (south-pole convention) maps them to `w`. PyVista enters *only*
in the `mesh/` builder when those coordinates are wrapped into a `pv.StructuredGrid` /
distorted. The current `RectangularSphereGenerator` couples coordinate generation with
`pv.StructuredGrid` construction — split it so the numpy coordinate generation lives at the
field/core layer and the PyVista wrapping lives in `mesh/sphere.py`.

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
