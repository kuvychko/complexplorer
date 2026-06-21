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
- [x] 4.1 Active paths unified on the canonical `core.functions.inverse_stereographic`
      (`z = 0` at south pole) via `sample_sphere`. The legacy `utils.mesh` projection
      helpers + `RectangularSphereGenerator` + `compute_riemann_sphere_distortion` are now
      unused by the kernel (only their own tests reference them); retained per the additive
      principle, slated for removal at 3.0 with the rest of the legacy mesh layer.
- [x] 4.2 Sphere sampling + `build_relief` use the canonical convention; `riemann_pv`
      flips to match (verified: |w|≈0 at south, large at north); ornament + matplotlib
      `riemann` unchanged.
- [x] 4.3 Regenerated the `riemann_pv` regression baseline to the corrected orientation;
      all 4 baselines pass. Ornament baseline unchanged (build_relief reproduces it).

## 5. Facade the public entry points (D3)
- [x] 5.1 `create_complex_surface` (backing `plot_landscape_pv` / `pair_plot_landscape_pv`)
      delegates to `sample` + `build_landscape`; signatures unchanged; regression holds.
- [x] 5.2 `riemann_pv` reimplemented over `sample_sphere` + `build_relief`.
- [x] 5.3 `OrnamentGenerator.generate_ornament` reimplemented over `sample_sphere` +
      `build_relief`; `save_stl` unchanged (already uses the shared `export/stl` post-proc).
- [x] 5.4 Regression 1.1 (landscape) + 1.3 (ornament) pass unchanged; 1.2 (riemann_pv)
      passes against the regenerated baseline. Full suite 367 passed.

## 6. Docs & close out
- [ ] 6.1 Document the STL orientation change in release notes and `OrnamentGenerator`
      docstring.
- [ ] 6.2 Add a short "3D surface kernel" architecture note (feeds the Phase 0
      backend-policy doc / `docs/development/`).
- [ ] 6.3 Run `pytest tests/` green; `openspec validate --specs`.
- [ ] 6.4 Update `openspec/ROADMAP.md` STATUS for this change.
