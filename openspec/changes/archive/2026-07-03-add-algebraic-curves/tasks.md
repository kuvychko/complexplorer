# Tasks — add-algebraic-curves

## 1. Mesh layer

- [x] 1.1 Add `_algebraic_grid(p, r_max, resolution)` to `mesh/riemann_surface.py`: polar disk
      grid, `w = √(polyval(p, z))`, returns per-sheet `(X, Y, ±Re(w), ±w)`
- [x] 1.2 Register `family="algebraic"` in `build_riemann_surface` with `p` parameter:
      validate `p` (sequence, len ≥ 2, nonzero leading coefficient), build both sheets,
      attach colors per sheet, merge with `merge_points=False`, set topology tag +
      `branch_points` metadata from `numpy.roots(p)`

## 2. Renderer

- [x] 2.1 `riemann_surface_pv`: accept/forward `p`, add to `_OWN_KWARGS`, docstring, height
      axis label `Re w` for the algebraic family

## 3. Tests and docs

- [x] 3.1 Mesh tests: two-sheet point count (2× single grid), branch points in metadata,
      RGB finite and in [0, 1], sheets' heights are ±mirrors, validation errors (missing/short
      `p`, zero leading coefficient, unknown family unchanged)
- [x] 3.2 Renderer smoke tests: off-screen `family="algebraic", p=[1, 0, -1, 0]`
      (elliptic curve `w² = z³ − z`) returns a plotter; screenshot export path exercised
- [x] 3.3 `CHANGELOG.md` Added entry
- [x] 3.4 Off-screen visual check of the elliptic-curve surface

## 4. Verification

- [x] 4.1 `pytest tests/` green; ruff clean; `openspec validate --specs` passes
