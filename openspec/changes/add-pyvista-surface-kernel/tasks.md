# Tasks — add-pyvista-surface-kernel

## 1. Lock current behavior first (regression safety net)
- [ ] 1.1 Add output-pinning tests for `plot_landscape_pv` / `pair_plot_landscape_pv`
      (mesh point/scalar arrays for a fixed function + resolution).
- [ ] 1.2 Add output-pinning tests for `riemann_pv` (points, `RGB`, `magnitude`, `phase`).
- [ ] 1.3 Add output-pinning tests for `OrnamentGenerator.generate_ornament` and STL bytes
      for a fixed preset. Note: these will be intentionally updated once D2 lands.

## 2. ComplexField (core/field.py)
- [ ] 2.1 Implement `ComplexField` container: `z`/sphere params, `w = f(z)`, `|w|`,
      `arg w`, mask, metadata. Must not import PyVista.
- [ ] 2.2 Implement `sample(func, domain=..., resolution=...)` for the planar case
      (reuse `Domain.mesh` / `Domain.outmask`) and a sphere-sampling path.
- [ ] 2.3 Handle infinities/NaNs once, here (the logic currently duplicated in callers).
- [ ] 2.4 Unit tests for field sampling, masking, infinity handling.

## 3. mesh/ package
- [ ] 3.1 Create `mesh/` package; move `RectangularSphereGenerator` → `mesh/sphere.py` and
      `apply_modulus_distortion` → `mesh/distortion.py` (math unchanged). Keep import shims
      at old paths if needed.
- [ ] 3.2 Implement `SurfaceMesh` (`mesh/surface.py`): wraps `pv.DataSet`; `.attach_colors`
      (single decorate path: `RGB`/`magnitude`/`phase`), `.to_pyvista`, `.save_stl`
      (triangulate first), `.screenshot`, `.validate_printability`.
- [ ] 3.3 Implement `build_landscape(field, ...)` (height = `scale(|f|)`, `z_max`, `log_z`).
- [ ] 3.4 Implement `build_relief(field, ...)` (radius = `scale(|f|)`, `r_min/r_max`).
- [ ] 3.5 Re-home STL post-proc (`scale_to_size`, `center_mesh`, `validate_printability`,
      `repair_mesh_simple`) so any `SurfaceMesh` can use it.
- [ ] 3.6 Unit tests: builders produce finite vertices + faces, correct scalar lengths,
      STL export non-empty, no NaNs in exported mesh.

## 4. Projection unification (D2)
- [ ] 4.1 Collapse the two `inverse_stereographic` definitions into one canonical function.
- [ ] 4.2 Make `build_relief` use a single pole convention (the `riemann_pv` viz
      convention); update the sphere sampling accordingly.
- [ ] 4.3 Update the ornament regression baseline (1.3) to the new orientation; confirm the
      printed orientation now matches `riemann_pv`'s rendered sphere.

## 5. Facade the public entry points (D3)
- [ ] 5.1 Reimplement `plot_landscape_pv` / `pair_plot_landscape_pv` over
      `sample` + `build_landscape` + `SurfaceMesh`; signatures unchanged.
- [ ] 5.2 Reimplement `riemann_pv` over `sample` + `build_relief` + `SurfaceMesh`.
- [ ] 5.3 Reimplement `OrnamentGenerator` / `create_ornament` as facades over
      `build_relief` + `SurfaceMesh.save_stl`; signatures unchanged.
- [ ] 5.4 Confirm regression tests 1.1–1.2 pass unchanged; 1.3 passes against new baseline.

## 6. Docs & close out
- [ ] 6.1 Document the STL orientation change in release notes and `OrnamentGenerator`
      docstring.
- [ ] 6.2 Add a short "3D surface kernel" architecture note (feeds the Phase 0
      backend-policy doc / `docs/development/`).
- [ ] 6.3 Run `pytest tests/` green; `openspec validate --specs`.
- [ ] 6.4 Update `openspec/ROADMAP.md` STATUS for this change.
