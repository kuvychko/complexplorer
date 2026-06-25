# Tasks — add-riemann-surfaces

## 1. Spike: validate the parametrizations  (DONE during review — see design.md)
- [x] 1.1 Verified numerically: power grid (`z = w**n`) closes across the `φ` seam, has `n`
      distinct sheet-heights, branch point finite; log is a clean helicoid spanning
      `[0, 2π·turns]`, `2π`-periodic in `(x,y)`. Key finding: height `Re(w)` puts the
      self-intersection on the negative real axis (principal-branch convention) — adopted as
      the power-family height (was `Im(w)`).

## 2. Surface builder (mesh layer)
- [ ] 2.1 New `mesh/riemann_surface.py`: a small family model (power, log) producing a
      structured parameter grid → embedded `X, Y, Z` → `pv.StructuredGrid`.
- [ ] 2.2 `build_riemann_surface(family, *, n / turns, resolution) -> SurfaceMesh`: wrap the
      grid in `SurfaceMesh` and `attach_colors(cmap, w)` (phase of the value). Bypass
      `ComplexField`. PyVista-only.

## 3. PyVista renderer
- [ ] 3.1 New `plotting/pyvista/riemann_surface.py`: `riemann_surface_pv(family, *, n=2 |
      turns=3, r_max=1.5, resolution, cmap=Phase(), interactive, notebook, camera_position,
      window_size, title, filename, return_plotter, **kwargs)` mirroring the `riemann_pv`
      wrapper; build → render with the shared plotter plumbing.
- [ ] 3.2 Export `riemann_surface_pv` from the package public API (`__init__.py` + `__all__`).

## 4. Tests (PyVista-gated)
- [ ] 4.1 `build_riemann_surface("power", n=2/3)`: point/face counts match the grid; `z ≈ w**n`
      at samples; height `== Re(w)` (self-intersection on the negative real axis); the `φ` seam
      closes (first/last column coincide); `n` distinct sheet-heights over a generic `z`.
- [ ] 4.2 `build_riemann_surface("log", turns=k)`: height spans `[0, 2πk]`; `(x,y)` periodic in `θ`.
- [ ] 4.3 Colors: `RGB`/`phase` populated, finite, within `[0, 1]`.
- [ ] 4.4 `riemann_surface_pv(..., return_plotter=True, interactive=False)` returns a plotter
      without rendering (Windows-CI offscreen-safe).

## 5. Docs
- [ ] 5.1 README + CLAUDE.md: add `riemann_surface_pv` under the PyVista plot types; note the
      Riemann *surface* (multivalued cover) vs Riemann *sphere* (single-valued) distinction.

## 6. Close out
- [ ] 6.1 `pytest tests/` green; `ruff` clean; `openspec validate --specs`.
- [ ] 6.2 Update `openspec/ROADMAP.md` (status; note 3.0.0 PyPI release is now unblocked once
      this lands — only `migrate-examples-and-docs` remains for the release).
