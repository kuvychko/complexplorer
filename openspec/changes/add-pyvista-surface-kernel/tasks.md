# Tasks — add-pyvista-surface-kernel

## 1. Lock current behavior first (regression safety net)
- [ ] 1.1 Add output-pinning tests for `plot_landscape_pv` / `pair_plot_landscape_pv`
      (mesh point/scalar arrays for a fixed function + resolution).
- [ ] 1.2 Add output-pinning tests for `riemann_pv` (points, `RGB`, `magnitude`, `phase`).
- [ ] 1.3 Add output-pinning tests for `OrnamentGenerator.generate_ornament` using mesh
      **arrays** (vertices, `RGB`, `magnitude`, `phase`, `radius`) plus a geometry hash
      (bounds, vertex/face counts, sorted-coordinate checksum) — NOT raw STL bytes, which
      are not stable across pyvista/vtk versions. The ornament output is unchanged by D2,
      so this baseline stays fixed through the refactor.

## 2. ComplexField (core/field.py)
- [ ] 2.1 Implement `ComplexField` container: `z`/sphere params, `w = f(z)`, `|w|`,
      `arg w`, mask, metadata. Must not import PyVista.
- [ ] 2.2 Implement `sample(func, domain=..., resolution=...)` for the planar case
      (reuse `Domain.mesh` / `Domain.outmask`) and a sphere-sampling path. The sphere path
      must stay pure-numpy (θ/φ → Cartesian → canonical `stereographic_projection`); split
      `RectangularSphereGenerator` so PyVista `StructuredGrid` construction moves to
      `mesh/sphere.py` and only the numpy coordinates live at the field layer.
- [ ] 2.3 Handle infinities/NaNs once, here (the logic currently duplicated in callers).
- [ ] 2.4 Unit tests for field sampling, masking, infinity handling.

## 3. mesh/ package
- [ ] 3.1 Create `mesh/` package; move `RectangularSphereGenerator` → `mesh/sphere.py` and
      `apply_modulus_distortion` → `mesh/distortion.py` (math unchanged). Keep import shims
      at old paths if needed.
- [ ] 3.2 Implement `SurfaceMesh` (`mesh/surface.py`): wraps `pv.DataSet`; `.attach_colors`
      (single decorate path: `RGB`/`magnitude`/`phase`), `.to_pyvista`, `.save_stl`
      (triangulate first), `.screenshot`, `.validate_printability`. `.attach_colors` must
      handle both the regular-grid case and the domain-filtered irregular-points case
      (preserve the reshape special-casing currently in `OrnamentGenerator`).
- [ ] 3.3 Implement `build_landscape(field, ...)` (height = `scale(|f|)`, `z_max`, `log_z`).
- [ ] 3.4 Implement `build_relief(field, ...)` (radius = `scale(|f|)`, `r_min/r_max`).
- [ ] 3.5 Re-home STL post-proc (`scale_to_size`, `center_mesh`, `validate_printability`,
      `repair_mesh_simple`) so any `SurfaceMesh` can use it.
- [ ] 3.6 Unit tests: builders produce finite vertices + faces, correct scalar lengths,
      STL export non-empty, no NaNs in exported mesh.

## 4. Projection unification (D2 — fix the outlier `riemann_pv`)
- [ ] 4.1 Collapse the two `inverse_stereographic` definitions into one canonical function
      using the documented core convention (`z = 0` at south pole).
- [ ] 4.2 Make sphere sampling + `build_relief` use that canonical convention everywhere.
      `riemann_pv` (the outlier) flips to match; the STL ornament and matplotlib `riemann`
      are already on this convention and do **not** change.
- [ ] 4.3 Update the `riemann_pv` regression baseline (1.2) to the corrected orientation;
      confirm `riemann_pv`, matplotlib `riemann`, and the STL ornament now share one
      orientation. The ornament baseline (1.3) stays unchanged.

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
