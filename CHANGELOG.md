# Changelog

All notable changes to complexplorer will be documented in this file.

## [3.0.0] - Unreleased

Consolidates all work since 2.0.0, the published baseline (PyPI, 2025-10-19). The 2.1–2.4
version bumps were internal milestones and were never released, so the notes below describe the
upgrade from 2.0.0 directly. A task-oriented version of the same material is in the
[migration guide](https://kuvychko.github.io/complexplorer/migration-3.0/).

### Breaking Changes
- **PyVista (>= 0.47) is now a required core dependency** and the sole 3D backend
- **Removed the matplotlib 3D paths**: `plot_landscape()`, `pair_plot_landscape()`, and the
  3D `riemann()` surface. Use `plot_landscape_pv()`, `pair_plot_landscape_pv()`, and
  `riemann_pv()` instead. The 2D stereographic charts (`riemann_chart`,
  `riemann_hemispheres`) remain matplotlib-based
- Removed the `HAS_PYVISTA` / `HAS_STL_EXPORT` capability flags **and their internal
  machinery** (`check_pyvista_available`, the `try/except ImportError` guards, and the
  `ImportError` fallbacks) — PyVista is imported unconditionally and those features are
  always available
- **`Presets` is now `PlotPresets`.** The render-settings class was too easily confused with
  the `catalog` function registry: `PlotPresets` configures a render, `catalog` supplies a
  function. 2.0.0 exposed neither name (it had `publication_preset()` and friends), so this
  affects only unreleased 3.0 builds
- **`setup_matplotlib_backend` and `ensure_interactive_plots` left the public API.** They were
  wrappers over `matplotlib.use()` and `plt.ion()`, and backend selection is matplotlib's
  business. Both were exported by 2.0.0; the migration guide gives the two-line replacement
- **Colormap configuration is validated at construction.** `phase_sectors` must be a positive
  integer, `r_log_base` greater than 1, and `r_linear_step` and `scale_radius` positive. 2.0.0
  accepted values that produced nonsense: `phase_sectors=0` raised `ZeroDivisionError` from
  inside the constructor, `-1` and `2.5` were accepted silently, and `r_log_base=1` rendered
  **every pixel as NaN** without raising anything
- **The PyVista renderers reject unknown keyword arguments** with a `ValidationError`
  instead of leaking them into `pyvista.Plotter` (a raw `TypeError`) or silently dropping
  them. Removed 2.x names are reported with their replacement: `n_theta`/`n_phi` →
  `resolution`, `show` → `interactive`
- Removed unused/dead public surface that was never wired up: the `complexplorer.export.base`
  exporter framework, `Matplotlib2DPlotter`, `ensure_consistent_normals`, the reversed
  `complexplorer.utils.mesh` projection aliases + `RectangularSphereGenerator`, the
  `core.functions.stereographic` alias, and most `utils.validation` helpers (only
  `validate_resolution` remains)
- **Curated the high-level API surface** — removed never-implemented stubs and redundant
  aliases so everything exported actually works:
  - `create_animation()` and `compare_functions()` (raised `NotImplementedError`) — no
    replacement yet; candidates for a future 3.x as real features
  - `analyze_function()` (its zero/pole "detection" was an unimplemented stub) — use
    `quick_plot()`; curated singularity answer keys live on `cp.catalog` presets
  - the `visualize` / `explore` aliases — use `quick_plot()`
  - the unused plotter base-class scaffolding `complexplorer.plotting.base`
    (`BasePlotter`, `Base2DPlotter`, `Base3DPlotter`, `PlotConfig`)

### Added
- **Riemann surfaces**: `riemann_surface_pv()` renders the multi-sheeted cover on which a
  multivalued family becomes single-valued — `power` roots `z^(1/n)`, `log`, and the
  algebraic family `w² = P(z)` (`family="algebraic"`, `p=[...]` polynomial coefficients;
  e.g. the elliptic curve `w² = z³ − z`) — with branch points, branch cuts, and sheet
  structure made explicit; algebraic branch points are recorded in the mesh metadata
- **Function preset registry**: `cp.catalog` with 17 curated presets. Each
  `FunctionPreset` carries a callable, an expression string, plain-dict
  domain/colormap/scaling specs, a hand-authored singularity answer key
  (`{type, at, order}`), and derived answer-key stats (counts by type, min separation)
- **Gallery generator**: `cp.gallery` renders the catalog into a reproducible asset bundle
  with a byte-stable `index.json` manifest (the machine-readable interchange for
  downstream consumers)
- **Command-line interface**: `complexplorer render | stl | list | gallery`, accepting
  `preset:<id>` references or expression strings via a safe, portable expression
  evaluator (`core/expression.py`)
- **PyVista surface kernel**: a shared `SurfaceMesh` pipeline underpinning 3D landscapes,
  Riemann relief, Riemann surfaces, and STL export
- `[examples]` optional-dependency group (`nbmake`, `nbconvert`, `ipykernel`); included
  in `[dev]`
- **Exception hierarchy**: `complexplorer.exceptions` with `ComplexplorerError` as the base
  class for all deliberate library errors; `ValidationError` now derives from it (and still
  from `ValueError`, so existing handlers keep working). Both are exported at top level
- **Phase-wheel legend**: `plot(..., legend=True)` and `pair_plot(..., legend=True)` draw a
  unit-disk inset colored by the active colormap (faithful for enhanced phase portraits,
  chessboards, and log rings), giving figures an in-image key for hue → phase and
  shading → modulus
- **Transfer-function explorer** (`cp.ee`): `TransferFunction(num, den, system="s"|"z")`
  with `poles`/`zeros`/`is_stable`/`frequency_response()`; the object is a plain complex
  callable, so it works with every renderer (`cp.plot`, `plot_landscape_pv`, `riemann_pv`,
  STL export). Companion views: `pole_zero_plot`, `bode_plot`, `nyquist_plot`, and
  `transfer_portrait` (phase portrait with poles/zeros and the stability boundary overlaid)

### Fixed
- Colormaps now emit finite, deterministic RGB at non-finite inputs (poles and essential
  singularities landing on in-domain grid nodes previously produced out-of-range,
  run-to-run-varying colors)
- `quick_plot` 3D/Riemann modes now dispatch to PyVista per the backend policy and no
  longer leak the `backend` kwarg into renderers
- `quick_plot(..., mode="riemann", domain=...)` now forwards a caller-supplied domain to
  `riemann_pv` (it was silently discarded)
- `riemann_chart(domain=...)` now actually masks out-of-domain samples with the colormap's
  out-of-domain color (it previously guarded on a nonexistent attribute and did nothing)
- `plot(..., ax=..., filename=...)` now saves the figure even when an `ax` is supplied, and
  `plot` consistently returns the drawn `Axes`
- `pair_plot_landscape_pv(title=...)` now renders `title` as a figure-level title instead of
  overwriting the codomain panel's label
- CLI: `render --show` opens a window in 2D mode (previously a silent no-op); `stl
  preset:<id>` now applies the preset's recommended domain and colormap; `main` reports any
  `ComplexplorerError` (not only `ValidationError`)

### Changed
- Version and license metadata reconciled: `complexplorer/_version.py` is the single
  version source; the code license is declared as an SPDX `license = "MIT"` expression
  (the deprecated `License ::` classifier was dropped), and `LICENSE`/`LICENSE.art` ship in
  the distribution
- **The typing contract now describes what the library actually calls.** A `ComplexFunction`
  protocol replaces `Callable[[complex], complex]` on the public entry points, which had
  rejected correctly written vectorized functions — and the library's own `TransferFunction` and
  catalog presets. The spec dictionaries and singularity records have declared shapes, every
  public callable has a return type, and CI type-checks the public surface plus a program that
  imports complexplorer as a consumer does
- The package now ships a PEP 561 `py.typed` marker so downstream type checkers honor its
  annotations; added a `Development Status` classifier and `keywords`; the `all` extra is
  now user-facing (`complexplorer[qt]`) rather than pulling in dev/test tooling
- Internal consolidation (no behavior change): a single modulus-scaling dispatch
  (`core.scaling.apply_scaling_mode`) shared by the mesh builders and sphere distortion; a
  shared planar input-resolver (`core.field.resolve_plane_inputs`) used by both the 2D and
  3D renderers; and a shared PyVista show/export tail
- Tooling and CI: ruff lint/format gates plus a GitHub Actions test matrix
  (ubuntu/windows × Python 3.11–3.13)
- Examples reworked onto the 3.0 surface: new `examples/{notebooks,scripts,gallery}`
  layout; registry-driven `examples/showcase.py` produces the committed (and exactly
  regenerable) gallery images and generated docs; the two legacy hand-rolled gallery
  generators were removed
- All four tutorial notebooks modernized (static PyVista Jupyter backend, no
  `HAS_PYVISTA` guards) and verified to execute top-to-bottom via the opt-in
  `pytest --nbmake examples/notebooks/`

## [2.0.0] - 2025-10-19

**Published to PyPI on 2025-10-19**, and the latest release until 3.0 — so it is the version
`pip install complexplorer` provided, and the one the 3.0 upgrade notes are written against. It
introduced the perceptual colormap families, renamed `n_phi` to `phase_sectors`, and began the
deprecation of the matplotlib 3D paths that 3.0 completes.

2.0.0 also changed `Rectangle` membership, which matters if you are coming from 1.x:
`contains()` tests the rectangle's actual `re_length`/`im_length` about its `center`, rather than
the square-padded viewing window that 1.x tested. A non-square `Rectangle` (with the default
`square=True`) no longer reports the padded strips as inside, so masking and STL/relief output
differ from 1.x. 3.0 keeps the 2.0.0 behaviour unchanged.

The 2.1 through 2.4 version numbers appear in development history as internal milestones on the
way to 3.0. None of them was released.

## [1.0.0] - 2025-07-27

### Major Refactoring
- Complete restructuring of the codebase with modular architecture
- Migrated from flat structure to organized submodules (`core`, `plotting`, `export`)
- Introduced abstract base classes for extensibility
- Standardized API with consistent parameter naming (`n` → `resolution`)

### New Features
- **Riemann Relief Maps**:
  - Riemann Relief Maps (or Mathematical Ornaments) with many different types of modulus scaling functions
  
- **PyVista Integration**: High-performance 3D visualizations with 15-30x speed improvement
  - `plot_landscape_pv()`, `pair_plot_landscape_pv()`, `riemann_pv()` functions
  - Interactive rotation, zooming, and navigation
  - HTML export capability for sharing visualizations
  
- **STL Export**: Generate 3D-printable mathematical ornaments
  - Export Riemann sphere and analytic landscape visualizations
  - Create mathematical art pieces and educational models
  
- **Enhanced Phase Portraits**: Auto-scaling for optimal visualization
  - Automatic square cell sizing with `auto_scale_r=True`
  - Improved modulus scaling options for all 3D plots
  
- **Domain Operations**: Set operations for complex domains
  - Union, intersection, and difference operations
  - Composite domains with automatic viewing window calculation

### Improvements
- Added PyQt6 backend support for interactive matplotlib plots
- Fixed Riemann sphere grid visualization and phase coloring symmetry
- Improved singularity handling in all plot types
- Added comprehensive unit test suite (341 tests with full coverage)
- Enhanced numerical stability and warning management

### API Changes
- Minimum Python version raised to 3.11
- Added optional dependencies: `[pyvista]`, `[qt]`, `[all]`
- Deprecated `sawtooth_legacy` function (removed)
- Consistent parameter naming across all functions

### Documentation
- New tutorial notebooks with clear examples
- Interactive CLI demo for optimal PyVista experience
- CLAUDE.md project guide for AI-assisted development

### Breaking Changes
- Removed backward compatibility with pre-0.1.2 versions
- Changed module structure (imports may need updating)
- Some function signatures updated for consistency

## [0.1.2] - Previous Release
- Initial public release with basic functionality

[Unreleased]: https://github.com/kuvychko/complexplorer/compare/v3.0.0...HEAD
[3.0.0]: https://github.com/kuvychko/complexplorer/compare/v2.0.0...v3.0.0
[2.0.0]: https://github.com/kuvychko/complexplorer/compare/v1.0.1...v2.0.0
[1.0.1]: https://github.com/kuvychko/complexplorer/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/kuvychko/complexplorer/compare/v0.1.2...v1.0.0
[0.1.2]: https://github.com/kuvychko/complexplorer/releases/tag/v0.1.2
