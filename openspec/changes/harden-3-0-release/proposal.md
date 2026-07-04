## Why

A deep pre-release review of the 3.0.0 candidate found no crashes on the documented happy
paths, but a cluster of small correctness bugs, stale-since-3.0 dead code, and
packaging/docs gaps that should not be frozen into a major release. Because 3.0 has not
been published yet, this is the last cheap moment to delete dead surface and fix behavior
before it becomes a compatibility contract.

## What Changes

- **Remove the `HAS_PYVISTA` / `check_pyvista_available()` machinery entirely.** PyVista is
  a required dependency as of 3.0, so the `try/except ImportError` guards, capability flags,
  and bare `raise ImportError` fallbacks across `plotting/pyvista/*`, `utils/mesh.py`,
  `export/stl/*`, `mesh/builders.py`, and `mesh/riemann_surface.py` are dead code that
  contradicts the shipped CHANGELOG. Replace with plain `import pyvista as pv`. **BREAKING**
  for anyone importing `HAS_PYVISTA` (an internal flag, never a documented API).
- **Delete dead code before it freezes into the public surface:** `export/base.py` (a
  330-line exporter framework nothing subclasses) and its test; `Matplotlib2DPlotter`;
  `utils/mesh.py`'s reversed-direction `stereographic_projection`/`inverse_stereographic`
  aliases and `RectangularSphereGenerator`; `ensure_consistent_normals`;
  `compute_riemann_sphere_distortion`; `warn_deprecated` and the unused `validate_*`
  helpers; and the `stereographic` back-compat alias in `core/functions.py`.
- **Fix correctness bugs:**
  - `Rectangle(re_length, im_length).contains()` currently tests against the square-padded
    viewing window, so a non-square rectangle reports membership over an inflated region.
    Membership will use the actual `re_length`/`im_length`. **BREAKING** (behavioral).
  - `riemann_chart`'s `domain` argument is a silent no-op (guards on a nonexistent
    `mask_list` attribute); it will actually mask via `domain.contains()`.
  - `riemann_pv` / `plot_landscape_pv` / `pair_plot_landscape_pv` leak unknown/removed
    keyword arguments into `pv.Plotter` (raw `TypeError`) or silently drop them (`show=`);
    they will reject unknown kwargs with a `ValidationError` that names the 3.0 replacement.
  - `quick_plot(..., mode="riemann")` will forward a caller-supplied `domain`.
  - `plot(..., ax=, filename=)` will honor `filename`, and `plot`'s return-type docstring
    will match its always-returns-Axes behavior.
  - `pair_plot_landscape_pv`'s `title` will become a figure-level title instead of
    replacing the codomain panel label.
- **Fix CLI behavior gaps:** `render --show` is a silent no-op in 2D mode (it never calls
  `plt.show()`); it will open a window in every mode. `stl preset:<id>` discards the preset's
  recommended domain/colormap — a violation of the existing `cli` spec — and will forward
  them. `main()` catches only `ValidationError`; it will report any `ComplexplorerError`
  cleanly. The `gallery` subcommand (shipped but absent from the spec) is documented, and the
  private `gallery._resolve` double-resolution in the CLI is removed.
- **Packaging & docs hygiene:** adopt an SPDX `license = "MIT"` expression (bumping the
  setuptools build requirement and removing the now-deprecated `License ::` classifier), add
  a PEP 561 `py.typed` marker, add a `Development Status` classifier and `keywords`; make the
  `all` extra user-facing (it currently expands to `dev,pyvista,qt`, pulling in test/lint
  tooling); rewrite README asset/link URLs to be absolute so the PyPI project page renders;
  fix the placeholder `github.com/user/...` URL; add README sections for the 3.0 headline
  features (`quick_plot`/`Presets`, `cp.catalog`, the CLI, `cp.ee`); reconcile the CHANGELOG
  `1.0`→`3.0` gap; and clear minor doc drift (`asteval` dependency, preset count, stale
  `interactive_showcase` reference).
- **Internal consolidation (no behavior change):** factor the duplicated modulus-scaling
  dispatch (`mesh/builders.py` vs `utils/mesh_distortion.py`) into one helper; share the
  domain/`z`/`f` input-resolution logic across the matplotlib and pyvista backends; delegate
  `OrnamentGenerator.save_stl` to `SurfaceMesh.save_stl`; de-duplicate the PyVista
  kwarg-blocklist / export-and-show tail and the `add_axes_widget` branches; drop the
  `create_complex_surface` return value nothing consumes and the `str(backend).startswith
  ("<MagicMock")` test-detection baked into `plotting/pyvista/utils.py`; make
  `plotting/pyvista/__init__.py`'s `__all__` coherent; canonicalize `ValidationError` imports
  on `complexplorer.exceptions`; and demote `validate_printability`'s verbose-gated warning to
  an unconditional one.
- **Close the test gap:** the flagship PyVista paths are tested only under a mocked
  `pyvista.Plotter`, so kwarg/render breakage slips through. Add at least one real
  off-screen render test.

Public-API argument-order and default-value inconsistencies (e.g. `plot(domain, func)` vs
`riemann_chart(func, domain=)`) are deliberately **left as-is**: reordering the flagship 2D
signature is high-churn, low-value churn even at a major bump. See design.md.

## Capabilities

### New Capabilities

_None._ This is a hardening change against existing capabilities.

### Modified Capabilities

- `domains`: Rectangle membership is defined by its `re_length`/`im_length`, independent of
  any square-padding applied to the viewing window.
- `plotting-2d`: `plot` honors `filename` even when an `ax` is supplied and documents its
  return contract; `riemann_chart`'s optional `domain` masks out-of-domain points.
- `plotting-3d-pyvista`: landscape renderers reject unknown/removed keyword arguments with a
  clear error naming the replacement; `pair_plot_landscape_pv`'s `title` is a figure title.
- `riemann-sphere`: `riemann_pv` rejects unknown/removed keyword arguments with a clear
  error and honors an explicit request for no interactive window.
- `high-level-api`: `quick_plot` forwards a caller-supplied `domain` in Riemann mode.
- `stl-export`: PyVista is a required dependency; STL export is always available — the
  optional-dependency gating requirement is retired (no capability flags, no import guards).
- `cli`: `render --show` opens a window in every mode (not just 3D/Riemann); `stl preset:<id>`
  honors the preset's recommended domain/colormap; the entry point documents the `gallery`
  subcommand; `main` reports any `ComplexplorerError`; and the "works without the 3D backend"
  requirement is retired (PyVista is required).
- `packaging`: the distribution declares an SPDX `license` expression (dropping the deprecated
  license classifier), ships a PEP 561 `py.typed` marker, and adds a development-status
  classifier plus keywords; the `all` extra installs only user-facing features; and the PyPI
  long description renders with absolute asset URLs.

## Impact

- **Code:** `core/domain.py`, `core/functions.py`, `plotting/matplotlib/plot_2d.py`,
  `plotting/pyvista/{plot_3d,riemann,riemann_surface,utils}.py`, `mesh/builders.py`,
  `mesh/riemann_surface.py`, `utils/{mesh,mesh_distortion,validation}.py`,
  `export/base.py` (deleted), `export/stl/{ornament_generator,mesh_repair,utils}.py`,
  `api.py`, `gallery.py`/`cli/main.py` (only if they touch removed helpers).
- **Public API removals (all internal or unused):** `HAS_PYVISTA`, `check_pyvista_available`,
  the `export.base` exporter classes, `Matplotlib2DPlotter`, `ensure_consistent_normals`,
  `compute_riemann_sphere_distortion`, the `utils.mesh` projection aliases +
  `RectangularSphereGenerator`, `warn_deprecated`, unused `validate_*` helpers, and the
  `stereographic` alias.
- **Behavioral (breaking) changes:** non-square `Rectangle` membership; unknown-kwarg
  rejection on the PyVista entry points.
- **Packaging:** `pyproject.toml` (`license`, `package-data` for `py.typed`, `all` extra),
  new `complexplorer/py.typed`, `README.md`, `CHANGELOG.md`, `CLAUDE.md`, `docs/`.
- **Tests:** delete `tests/unit/export/test_base_export.py` and mocked-kwarg tests that
  assert removed behavior; add a real off-screen PyVista render test; update tests importing
  `HAS_PYVISTA`.
- **Specs:** delta specs for the seven modified capabilities above.
