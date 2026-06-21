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
- [x] 2.1 Implement `ComplexField` container: `z`/sphere params, `w = f(z)`, `|w|`,
      `arg w`, mask, metadata. Must not import PyVista. → dataclass; `test_field_is_pyvista_free`
      guards the import boundary.
- [x] 2.2 Implement `sample(func, domain=..., resolution=...)` for the planar case
      (reuse `Domain.mesh` / `Domain.outmask`) and a sphere-sampling path. → `sample()` +
      `sample_sphere()`; sphere stays pure-numpy via `sphere_coordinates()` +
      canonical `inverse_stereographic(project_from_north=True)` (z=0 at south).
- [x] 2.3 Record non-finiteness (NaN/inf) in the field's mask/metadata — do NOT pre-clamp
      moduli here. → `ComplexField.nonfinite` property; raw `w`/`modulus` preserved.
- [x] 2.4 Unit tests for field sampling, masking, infinity handling. → `test_field.py`,
      10 tests incl. canonical-projection (south-pole→0) and pyvista-free guard.

## 3. mesh/ package (additive — wrap existing utilities in place, no moves)
- [x] 3.1 Create the `mesh/` package with NEW code only, importing existing utilities
      unchanged. → θ/φ→Cartesian extracted to `core.field.sphere_coordinates` (pure-numpy);
      `apply_modulus_distortion` imported in place.
- [x] 3.2 Implement `SurfaceMesh` (`mesh/surface.py`): `.attach_colors` (single
      `RGB`/`phase` path, threads `outmask`, faithful — no NaN sanitization), `.attach_scalar`,
      `.to_pyvista`, `.save_stl`, `.screenshot`, `.validate_printability`. Domain-filtered
      irregular case handled in `build_relief` via carry-through cell removal.
- [x] 3.3 Implement `build_landscape(field, ...)` — reproduces `create_complex_surface`
      exactly (verified, incl. NaN-blank masking).
- [x] 3.4 Implement `build_relief(field, ...)` — reproduces the ornament exactly for both
      no-domain and domain (cell-removal) cases; canonical south-pole projection via the
      field. KEY FIX: attach arrays AFTER `extract_surface` (C-order), not on the
      StructuredGrid (VTK-native order).
- [x] 3.5 `SurfaceMesh.save_stl()` calls the existing `export/stl` post-proc in place and
      drops non-finite vertices (landscape→STL works).
- [x] 3.6 Unit tests: `tests/unit/mesh/test_kernel.py` (faces/scalars, faithfulness vs
      `create_complex_surface` + ornament, STL non-empty & finite). 11 tests.

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
