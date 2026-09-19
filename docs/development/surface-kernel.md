# 3D surface kernel

The PyVista 3D paths share a small kernel so geometry, decoration, and export live in one
place instead of being re-implemented per entry point.

```
   sample() / sample_sphere()  ──▶  ComplexField        core/field.py (PyVista-free)
   (plane or sphere)                 z|sphere_xyz, w=f(z), |w|, arg w, mask, metadata
                                          │
                        ┌─────────────────┴─────────────────┐
                build_landscape                       build_relief          mesh/builders.py
                (height = scale|f|)              (radius = scale|f|)
                        └─────────────────┬─────────────────┘
                                    SurfaceMesh                              mesh/surface.py
                          PyVista dataset + .attach_colors() + metadata
                                          │
                       ┌──────────────────┼───────────────────┐
                 .to_pyvista()       .save_stl()         .screenshot()
```

## Layers

- **`ComplexField`** (`core/field.py`) — the sampled function, backend-agnostic and
  **PyVista-free** (keeps core/2D importable without the 3D backend). Records where `f` is
  non-finite but does not clamp it.
- **builders** (`mesh/builders.py`) — the only topology-specific code. They own the
  geometry-specific handling: `build_landscape` blanks masked cells to NaN and maps
  `|f|`→height; `build_relief` removes out-of-domain cells and maps `|f|`→radius via the
  shared `apply_modulus_distortion`.
- **`SurfaceMesh`** (`mesh/surface.py`) — wraps the PyVista dataset; one decorate path
  (`attach_colors`, faithful — no NaN sanitization, since PyVista tolerates NaN scalars);
  STL export reuses the existing `export/stl` post-processing in place.

## Conventions

- **Projection.** Sphere sampling uses the canonical convention (`z = 0` at the south
  pole) via `core.functions.inverse_stereographic(project_from_north=True)`. All sphere
  renderers and the STL ornament share this single projection (see the `riemann-sphere`
  capability and `backend-policy.md`).
- **Additive, no relocation.** The kernel imports the existing `utils.mesh` /
  `utils.mesh_distortion` / `export.stl` utilities in place. The legacy
  `RectangularSphereGenerator` + divergent sphere projection helpers are now unused by the
  kernel and are slated for removal at 3.0.

## Public API

`plot_landscape_pv`, `pair_plot_landscape_pv`, `riemann_pv`, `OrnamentGenerator`, and
`create_ornament` are thin facades over this kernel; their signatures are unchanged.
