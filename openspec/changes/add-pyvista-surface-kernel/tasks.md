# Tasks — add-pyvista-surface-kernel

## 1. Lock current behavior first (regression safety net)
- [x] 1.1 Add output-pinning tests for `plot_landscape_pv` / `pair_plot_landscape_pv`
      (mesh point/scalar arrays for a fixed function + resolution). → via the shared
      `create_complex_surface` seam; two cases (modulus + domain-mask), golden `.npz`.
- [x] 1.2 Add output-pinning tests for `riemann_pv` (points, `RGB`, `magnitude`, `phase`).
      → faithful extraction via `return_plotter`; golden `.npz` (regenerated after the D2
      flip).
- [x] 1.3 Add output-pinning tests for `OrnamentGenerator.generate_ornament` using mesh
      **arrays** (vertices, `RGB`, `magnitude`, `phase`) — NOT raw STL bytes. Golden
      `.npz`; unchanged by D2. (`tests/regression/`, deterministic across 3 runs.)
      NOTE: exact in-domain pole hits give non-deterministic RGB (colormap `nan->int`
      cast) — a latent colormap bug; regression cases avoid exact pole nodes.

## 2. ComplexField (core/field.py)
- [ ] 2.1 Implement `ComplexField` container: `z`/sphere params, `w = f(z)`, `|w|`,
      `arg w`, mask, metadata. Must not import PyVista.
- [ ] 2.2 Implement `sample(func, domain=..., resolution=...)` for the planar case
      (reuse `Domain.mesh` / `Domain.outmask`) and a sphere-sampling path. The sphere path
      must stay pure-numpy (θ/φ → Cartesian → canonical `stereographic_projection`); split
      `RectangularSphereGenerator` so PyVista `StructuredGrid` construction moves to
      `mesh/sphere.py` and only the numpy coordinates live at the field layer.
- [ ] 2.3 Record non-finiteness (NaN/inf) in the field's mask/metadata — do NOT pre-clamp
      moduli here. The inf→geometry mapping is topology-specific (relief clamps inf→`r_max`
      via `apply_modulus_distortion`; landscape uses `z_max` clip / NaN-blank), so the
      builders own clamping.
- [ ] 2.4 Unit tests for field sampling, masking, infinity handling.

## 3. mesh/ package (additive — wrap existing utilities in place, no moves)
- [ ] 3.1 Create the `mesh/` package with NEW code only. It IMPORTS the existing
      `RectangularSphereGenerator` (`utils/mesh.py`) and `apply_modulus_distortion`
      (`utils/mesh_distortion.py`) unchanged — no relocation, no shims. (Extract only the
      θ/φ→Cartesian coordinate math into a pure-numpy helper shared by `sample()` and the
      generator, per the import-layering note.)
- [ ] 3.2 Implement `SurfaceMesh` (`mesh/surface.py`): wraps `pv.DataSet`; `.attach_colors`
      (single decorate path: `RGB`/`magnitude`/`phase`), `.to_pyvista`, `.save_stl`
      (triangulate first), `.screenshot`, `.validate_printability`.
      `.attach_colors`: (a) passes `field.mask` as `outmask` to `cmap.rgb`; (b) handles
      both the regular-grid and the domain-filtered irregular-points cases (preserve the
      reshape special-casing currently in `OrnamentGenerator`); (c) stays **faithful** — do
      NOT sanitize non-finite RGB (PyVista tolerates NaN scalars; sanitizing would change
      pole pixels and break output preservation — unlike the matplotlib path).
- [ ] 3.3 Implement `build_landscape(field, ...)` (height = `scale(|f|)`, `z_max`, `log_z`;
      preserves landscape's NaN-blank masking).
- [ ] 3.4 Implement `build_relief(field, ...)` (radius = `scale(|f|)`, `r_min/r_max`;
      preserves relief's cell-removal masking; canonical south-pole projection per D2).
- [ ] 3.5 `SurfaceMesh.save_stl()` imports and calls the existing `export/stl` post-proc
      (`scale_to_size`, `center_mesh`, `validate_printability`, `repair_mesh_simple`) — no
      move — and drops NaN vertices so landscape→STL exports are watertight-capable.
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
