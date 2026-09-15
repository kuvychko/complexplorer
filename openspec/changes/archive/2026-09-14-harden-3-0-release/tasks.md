## 1. Remove the HAS_PYVISTA / import-guard machinery

- [x] 1.1 Replace the `try/except ImportError` + `HAS_PYVISTA` block with plain `import pyvista as pv` in `plotting/pyvista/plot_3d.py`, `plotting/pyvista/riemann.py`, and `plotting/pyvista/utils.py`; delete `check_pyvista_available()` and all its call sites.
- [x] 1.2 Same in `utils/mesh.py`, `export/stl/utils.py`, `export/stl/mesh_repair.py`, and `export/stl/ornament_generator.py`; delete the `HAS_PYVISTA` definitions and `check_pyvista_available` calls, and remove the bare `raise ImportError` fallbacks.
- [x] 1.3 In `mesh/builders.py` and `mesh/riemann_surface.py`, drop the `HAS_PYVISTA`/`pv is None` guards and the `raise ImportError` branches; import PyVista unconditionally.
- [x] 1.4 Grep the whole repo (incl. tests, examples, notebooks, docs) for `HAS_PYVISTA` / `check_pyvista_available`; update or delete every remaining reference, including tests that import the flag.
- [x] 1.5 Confirm no PyVista `raise ImportError` remains and that no PyVista-availability symbol is importable from the package.

## 2. Delete dead code before it freezes into 3.0

- [x] 2.1 Delete `export/base.py` and its test `tests/unit/export/test_base_export.py`; make `export/__init__.py` re-export only the real STL surface (drop `from .base import *` and the broken `__all__`).
- [x] 2.2 Delete `Matplotlib2DPlotter` from `plotting/matplotlib/plot_2d.py` and its tests; confirm `plot`/`pair_plot` are the only 2D entry points.
- [x] 2.3 Delete the reversed-direction `stereographic_projection`/`inverse_stereographic` aliases and `RectangularSphereGenerator` from `utils/mesh.py`; remove `from .mesh import *` leakage from `utils/__init__.py` and retire the now-obsolete parts of `tests/unit/utils/test_mesh.py`.
- [x] 2.4 Delete `ensure_consistent_normals` (`export/stl/mesh_repair.py`, and its `export/stl/__init__.py` export), `compute_riemann_sphere_distortion` (`utils/mesh_distortion.py`), and `warn_deprecated` + the unused `validate_*` helpers in `utils/validation.py` (keep `validate_resolution` and the `ValidationError` re-export).
- [x] 2.5 Delete the `stereographic = stereographic_projection` back-compat alias in `core/functions.py` and remove it from `core/__init__.py`.
- [x] 2.6 Update every `__all__` / `__init__` touched by 2.1–2.5 and confirm `python -c "import complexplorer as cp"` and `import complexplorer.utils, complexplorer.export` all succeed.

## 3. Fix correctness bugs

- [x] 3.1 `core/domain.py`: make `Rectangle.contains` test against the actual `re_length`/`im_length` about `center`, independent of the square-padded viewing window; verify `Rectangle(4, 2).contains(0+1.5j)` is `False` and `Rectangle(4, 4)` behavior is unchanged.
- [x] 3.2 `plotting/matplotlib/plot_2d.py`: replace `riemann_chart`'s dead `mask_list` branch with masking via `domain.contains()` (out-of-domain samples get the out-of-domain color); no-domain path unchanged.
- [x] 3.3 `plotting/matplotlib/plot_2d.py`: honor `filename` in `plot` even when `ax` is supplied (save `ax.figure`); fix the docstring so the documented return type matches the always-returns-`Axes` behavior.
- [x] 3.4 `plotting/pyvista/riemann.py` and `plotting/pyvista/plot_3d.py`: stop forwarding `**kwargs` into `pv.Plotter`; validate against the documented signature and raise `ValidationError` for unknown kwargs, naming the 3.0 replacement for known-removed names (`n_theta`/`n_phi` → `resolution`, `show` → `interactive`, `project_from_north` removed). Remove the now-unused kwarg blocklists and the dead unit-sphere-point computations that only fed the dead lat/long-grid params.
- [x] 3.5 `plotting/pyvista/plot_3d.py`: make `pair_plot_landscape_pv`'s `title` a figure-level title; keep `Codomain f(z)` as the codomain panel label.
- [x] 3.6 `api.py`: forward a caller-supplied `domain` to `riemann_pv` in `quick_plot(mode="riemann")`.

## 4. CLI fixes

- [x] 4.1 `cli/main.py`: make `render --show` open a window in 2D mode (call the matplotlib show path after `plot_2d`, using the interactive-backend helper); confirm 3D/Riemann behavior is unchanged.
- [x] 4.2 `cli/main.py`: in `cmd_stl`, forward the resolved preset's domain (and colormap where `OrnamentGenerator` accepts one) instead of discarding `preset`, matching `cmd_render` and the `cli` spec.
- [x] 4.3 `cli/main.py`: catch `ComplexplorerError` (not only `ValidationError`) in `main`; have `generate_gallery` raise on an empty selection so the CLI can drop the private `gallery._resolve` import and the double resolution.
- [x] 4.4 `cli/main.py`: warn (don't silently ignore) when `--scaling` is passed with `--mode 2d`, and when both `--preset` and `--tag` are given to `gallery`.

## 5. Packaging and docs hygiene

- [x] 5.1 `pyproject.toml`: bump `[build-system].requires` to `setuptools>=77`; add `license = "MIT"` (SPDX) to `[project]`, remove the `License :: OSI Approved :: MIT License` classifier, and confirm `LICENSE`/`LICENSE.art` still ship.
- [x] 5.2 `pyproject.toml`: add `Development Status :: 5 - Production/Stable` classifier and a `keywords` list; add `complexplorer/py.typed` plus a `[tool.setuptools.package-data]` entry so it ships in the wheel (note `include-package-data = false`).
- [x] 5.3 `pyproject.toml`: make `all` user-facing (`complexplorer[qt]`, not `[dev,pyvista,qt]`); reconcile the README's description of `[all]`.
- [x] 5.4 `README.md`: rewrite image/asset and doc-link URLs to absolute `raw.githubusercontent.com/kuvychko/complexplorer/main/...` / GitHub URLs; fix the placeholder `github.com/user/complexplorer` URL (also in `docs/README.md`).
- [x] 5.5 `README.md`: add short sections for the 3.0 headline features — `quick_plot`/`Presets`, `cp.catalog`, the `complexplorer` CLI, and `cp.ee` — mirroring the CLAUDE.md quick-reference snippets.
- [x] 5.6 `CHANGELOG.md`: add the `HAS_PYVISTA` removal and the `Rectangle` membership fix under 3.0.0 Breaking Changes; reconcile the `1.0`→`3.0` gap (one-line `[2.0.0]` note or reword the preamble).
- [x] 5.7 Clear minor doc drift: add `asteval` to the CLAUDE.md/README dependency list, correct the preset count (17, not ~20) in `core/presets.py` comments, and fix the stale `interactive_showcase` "8 functions/8 schemes" claim in `docs/README.md`.

## 6. Internal consolidation (behavior-preserving)

- [x] 6.1 Factor the duplicated modulus-scaling dispatch (the `custom`/unknown-mode handling in `mesh/builders.py` and `utils/mesh_distortion.py`) into one shared helper and call it from both; confirm identical error text and outputs.
- [x] 6.2 Extract a shared domain/`z`/`f` input-resolution helper used by both `plotting/matplotlib/plot_2d.py` and `plotting/pyvista/plot_3d.py`.
- [ ] 6.3 Delegate `OrnamentGenerator.save_stl` to `SurfaceMesh.save_stl` (keep the `SurfaceMesh` built in `generate_ornament` instead of discarding it). **DEFERRED**: `SurfaceMesh.save_stl` does extra vertex-cleaning/triangulation, so delegating would change STL output and risk the ornament regression suite; not worth the risk in a release-hardening pass.
- [x] 6.4 `plotting/pyvista/utils.py`: extract one `_finalize(plotter, filename, interactive, return_plotter)` helper for the export/show tail (replacing the identical-branch `if interactive: show(); export else: export`); remove the `str(backend).startswith("<MagicMock")` test-detection; drop `add_axes_widget`'s unused `position` param and fix its `-> None` annotation.
- [x] 6.5 `plotting/pyvista/plot_3d.py`: removed the leftover kwarg blocklists (with 3.4). **Partially deferred**: the `create_complex_surface` `(grid, rgb)` tuple is kept — several unit/regression tests assert on `rgb`, so changing the return type would churn tests for a cosmetic gain not worth the risk pre-release.
- [x] 6.6 `plotting/pyvista/__init__.py`: make `__all__` list the real exported functions (incl. `riemann_surface_pv`) instead of module names, with explicit imports.
- [x] 6.7 Canonicalize `ValidationError` imports on `complexplorer.exceptions` across the package (e.g. `api.py`, `mesh/builders.py`, and the core modules currently importing from `utils.validation`).
- [x] 6.8 `export/stl/utils.py`: make `validate_printability`'s failure warning unconditional (not gated on `verbose`).

## 7. Tests and verification

- [x] 7.1 Remove or rewrite tests that assert removed behavior (mocked-kwarg forwarding, `export.base`, `Matplotlib2DPlotter`, deleted helpers, `HAS_PYVISTA` imports).
- [x] 7.2 Add at least one real off-screen PyVista render test (`interactive=False`, `off_screen=True`) exercising a landscape and a Riemann sphere without mocking `pyvista.Plotter`, plus tests for the new `ValidationError` on unknown kwargs.
- [x] 7.3 Add tests for the fixed bugs: non-square `Rectangle.contains`, `riemann_chart` domain masking, `plot(ax=, filename=)` saving, `quick_plot(mode="riemann", domain=...)` forwarding, `render --mode 2d --show`, and `stl preset:` domain forwarding.
- [x] 7.4 Run the full gate: `pytest`, `ruff check`, `openspec validate --specs`, `uv build`, and a fresh-venv wheel install smoke test (`complexplorer list`; `import complexplorer`; one real off-screen render; wheel METADATA carries the MIT license and `py.typed`). Confirm the gallery `index.json` is byte-unchanged.
