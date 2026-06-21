# Add PyVista surface kernel

## Why

Three PyVista entry points (`create_complex_surface` for landscapes, `riemann_pv`,
`OrnamentGenerator`) each re-implement the same three steps inline — evaluate `f`, compute
`|f|`/`arg f`, attach `RGB`/`magnitude`/`phase` scalars — and only genuinely differ in one
thing: planar height vs. spherical radius. The radial-scaling kernel
(`apply_modulus_distortion`) and the STL post-processing (`scale_to_size`, `center_mesh`,
`validate_printability`, `repair_mesh_simple`) are *already* shared and topology-agnostic;
they are just wired to one caller each.

Without a unifying abstraction, every Phase 2+ feature (gallery, presets) and especially
Phase 4 (Riemann surfaces) would fork a fourth and fifth copy of the decorate-and-export
logic. This change builds the kernel once, so later phases are cheap. It is the
foundation called for in `openspec/ROADMAP.md` Phase 1 (v2.2).

Two latent inconsistencies are also resolved (see `design.md`):
- the **`riemann_pv` sphere is rendered mirrored** relative to the documented core
  projection convention, the matplotlib sphere, and the STL ornament (it is the lone path
  using a divergent `(1+z)` projection);
- there are **two `inverse_stereographic` definitions** with different signatures, which is
  the root of that divergence.

## What changes

- Add `ComplexField` (+ `sample()`) in `core/field.py`: a backend-agnostic container for a
  sampled complex field (`z`/sphere params, `w = f(z)`, `|w|`, `arg w`, mask, metadata).
- Add a `mesh/` package: `build_landscape(field)` and `build_relief(field)` builders, and
  a `SurfaceMesh` wrapper that owns the `PolyData`, color/scalar attachment, metadata, and
  export (`.to_pyvista()`, `.save_stl()`, `.screenshot()`, `.validate_printability()`).
- Absorb `utils/mesh.py` (sphere generator) and re-home `utils/mesh_distortion.py` (the
  scaling kernel is reused unchanged) and the STL post-processing under the kernel.
- Refactor the five public entry points (`plot_landscape_pv`, `pair_plot_landscape_pv`,
  `riemann_pv`, `OrnamentGenerator`, `create_ornament`) into thin facades over the kernel,
  preserving their signatures.
- **Unify the projection convention** on the documented core convention (`z=0` at south
  pole), which the matplotlib sphere and STL ornament already follow; `riemann_pv`'s
  rendered sphere flips to match (it is the outlier). Existing STL files stay valid.
  Collapse the duplicate `inverse_stereographic` into one canonical function.

## Non-goals

- `FieldOnSurface` / `RiemannSurface` / multi-sheet geometry — Phase 4.
- Camera/lighting preset *library* expansion beyond what current functions already do —
  can be folded in here only if cheap; otherwise its own small change.
- Removing matplotlib 3D — that is the 3.0 `require-pyvista-3d-backend` change.

## Impact

- New: `complexplorer/core/field.py`, `complexplorer/mesh/` package.
- Refactored (signatures preserved): `plotting/pyvista/plot_3d.py`,
  `plotting/pyvista/riemann.py`, `export/stl/ornament_generator.py`.
- Re-homed: `utils/mesh.py`, `utils/mesh_distortion.py`, `export/stl/utils.py` helpers.
- **Behavior change (intentional, documented):** the `riemann_pv` rendered sphere
  orientation flips to match the documented core convention, the matplotlib sphere, and
  the STL ornament (it was previously mirrored). The STL ornament does **not** change, so
  existing exported files stay valid.
- Affected specs: new `surface-mesh` capability; `riemann-sphere` gains a canonical-
  projection requirement and tightens the backend-parity requirement (projection pole is
  no longer an allowed difference).
- Risk: medium. Mitigated by writing output-pinning regression tests for all five entry
  points *before* refactoring (the `riemann_pv` orientation flip is the one expected diff).
