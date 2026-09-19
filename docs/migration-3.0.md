# Migrating from 2.0.0 to 3.0

Complexplorer 2.0.0 was published to PyPI on 2025-10-19 and was the latest release until 3.0, so
it is what `pip install complexplorer` gave you before this release. This page covers upgrading
from it. There is a [short appendix](#appendix-coming-from-1x) for 1.x.

## If you only do three things

1. **Rename your 3D calls.** `plot_landscape` → `plot_landscape_pv`, `pair_plot_landscape` →
   `pair_plot_landscape_pv`, and the 3D `riemann` → `riemann_pv`. The matplotlib 3D backend is
   gone; PyVista is now a required dependency and the sole 3D renderer.
2. **Replace `show(...)` with `quick_plot(...)`,** and the `*_preset()` functions with
   `PlotPresets` methods.
3. **Catch `ComplexplorerError` or `ValidationError`.** The twelve specific exception classes are
   gone; the base classes cover them.

Everything else below is a lookup table for when something breaks.

## Removed names and their replacements

Every public name 2.0.0 exported that 3.0 does not. This table is checked against the 2.0.0 tag by
`tests/unit/test_v2_surface_accounted.py`, so it cannot fall behind.

| 2.0.0 name | 3.0 replacement |
|---|---|
| `ComputationError` | `ComplexplorerError` — the base class covers computation failures |
| `DependencyError` | removed — PyVista is a required dependency, so the condition cannot arise |
| `DomainError` | `ValidationError` |
| `ensure_interactive_plots` | removed from the public API — use matplotlib directly (below) |
| `ExportError` | `ComplexplorerError` |
| `FunctionEvaluationError` | `ComplexplorerError` |
| `HAS_PYVISTA` | removed — PyVista is always installed |
| `HAS_STL_EXPORT` | removed — STL export is always available |
| `ImageExportError` | `ComplexplorerError` |
| `MeshGenerationError` | `ComplexplorerError` |
| `OptionalDependencyError` | removed — PyVista is a required dependency |
| `PyVistaNotAvailableError` | removed — PyVista is a required dependency |
| `ResolutionError` | `ValidationError` |
| `setup_matplotlib_backend` | removed from the public API — use matplotlib directly (below) |
| `STLExportError` | `ComplexplorerError` |
| `disable_logging` | removed — use the standard `logging` module |
| `enable_debug_logging` | removed — use `logging.getLogger('complexplorer').setLevel(logging.DEBUG)` |
| `get_logger` | removed — use `logging.getLogger(__name__)` |
| `high_contrast_preset` | `PlotPresets.high_contrast()` |
| `interactive_preset` | `PlotPresets.interactive()` |
| `pair_plot_landscape` | `pair_plot_landscape_pv()` |
| `plot_landscape` | `plot_landscape_pv()` |
| `publication_preset` | `PlotPresets.publication_ready()` |
| `riemann` | `riemann_pv()` (3D) or `riemann_chart()` / `riemann_hemispheres()` (2D charts) |
| `setup_logging` | removed — configure the standard library `logging` module directly |
| `show` | `quick_plot()` |

### Renamed: `Presets` is now `PlotPresets`

If you used an unreleased 3.0 build, the render-settings class was briefly called `Presets`. It is
`PlotPresets` in the release, because `Presets` and `catalog` were too easy to confuse:

> `PlotPresets` configures a render; `catalog` supplies a function.

2.0.0 had neither name — it had `publication_preset()` and friends — so if you are upgrading from
the published release, use the table above.

### The backend helpers

These were thin wrappers over matplotlib, and choosing a backend is matplotlib's business:

```python
# 2.0.0
cp.setup_matplotlib_backend()
cp.ensure_interactive_plots()

# 3.0
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("QtAgg")   # or whichever GUI backend you have
plt.ion()
```

Nothing else changes: the plotting functions behave the same way once a backend is active.

## Renamed: `n_phi` is now `phase_sectors`

2.0.0 renamed the phase-sector count from `n_phi` to `phase_sectors`; 3.0 keeps that name. Code
written against 2.0.0 needs no change. Code written against 1.x does:

```python
cp.Phase(n_phi=6, auto_scale_r=True)          # 1.x
cp.Phase(phase_sectors=6, auto_scale_r=True)  # 2.0.0 and 3.0
```

Passing `n_phi` raises `ValidationError` naming the replacement. The same applies to
`PolarChessboard` and every perceptual colormap, to preset `cmap_spec` records, and to the CLI
`--cmap phase:N` shorthand.

> Unrelated: the PyVista renderers also accept `resolution`, which replaced a *different*
> `n_phi` — the 2.x sphere-mesh argument. That mapping is unchanged.

## Stricter colormap validation

3.0 validates colormap configuration at construction. Values that 2.0.0 accepted and rendered as
nonsense now raise `ValidationError` immediately:

| Configuration | 2.0.0 | 3.0 |
|---|---|---|
| `Phase(phase_sectors=0)` | `ZeroDivisionError` from inside the constructor | `ValidationError` naming the value |
| `Phase(phase_sectors=-1)` or `2.5` | accepted; meaningless sector count | `ValidationError` |
| `Phase(r_log_base=1)` | accepted; **every pixel rendered as NaN** | `ValidationError` |
| `Phase(r_linear_step<=0)`, `scale_radius<=0` | accepted; degenerate output | `ValidationError` |

If one of these starts raising, the previous output was not what you wanted.

## Gallery manifest: `schema_version` 2 → 3

`index.json` records carry `cmap_spec`, whose sector-count key moved from `n_phi` to
`phase_sectors`. Consumers of the manifest should read the new key; the version bump marks the
change.

The manifest is byte-identical for the same selection and library version **on any platform** in
3.0 — derived coordinates are quantized, so a manifest generated on Windows and one generated on
Linux are the same bytes. Portrait PNGs remain reproducible only best-effort.

## A complete example

```python
# ---- 2.0.0 ----
import complexplorer as cp

if cp.HAS_PYVISTA:
    cp.setup_matplotlib_backend()
    f = lambda z: (z**2 - 1) / (z**2 + 1)
    cp.plot_landscape(cp.Rectangle(4, 4), f, **cp.publication_preset())
    cp.show(f)
```

```python
# ---- 3.0 ----
import complexplorer as cp

f = lambda z: (z**2 - 1) / (z**2 + 1)

# PyVista is always available, so there is no flag to check.
cp.plot_landscape_pv(cp.Rectangle(4, 4), f, **cp.PlotPresets.publication_ready())
cp.quick_plot(f)
```

## Troubleshooting

**`ImportError: cannot import name 'plot_landscape'`**
The matplotlib 3D backend was removed. Use `plot_landscape_pv`. See the table above.

**`AttributeError: module 'complexplorer' has no attribute 'HAS_PYVISTA'`**
PyVista is a required dependency now, so the flag has no meaning. Delete the check.

**`ValidationError: Phase no longer accepts 'n_phi'`**
Rename the argument to `phase_sectors`. The message names the replacement.

**`ValidationError: phase_sectors must be a positive integer`**
3.0 validates this at construction. A zero, negative or fractional sector count never produced a
meaningful portrait.

**`ImportError` for one of the twelve exception classes**
Catch `ComplexplorerError` for anything the library raises, or `ValidationError` for bad
arguments. `ColormapError` is a `ValidationError`, so one clause covers both.

**A 3D window does not open, or rendering fails on a server**
Pass `interactive=False` with `filename=` to render off-screen. On Linux without a display, set
`PYVISTA_OFF_SCREEN=true` and run under a virtual framebuffer. See
[3D landscapes and the Riemann sphere](guide/three-dimensions.md).

## Appendix: coming from 1.x

1.x is two breaking releases behind, so read the 2.0 notes in the
[changelog](https://github.com/kuvychko/complexplorer/blob/main/CHANGELOG.md) as well. The parts
that matter most:

- `n_phi` became `phase_sectors` in 2.0 (above).
- The perceptual colormap families — `OklabPhase`, `CubehelixPhase` and the rest — arrived in 2.0.
- Everything in the table above applies, since those names existed in 1.x too.

### Non-square rectangles mask differently than they did in 1.x

`Rectangle` takes `square=True` by default, which expands the *viewing window* to a square. In
1.x, `contains()` tested against that expanded window, so a non-square rectangle had no
out-of-domain region at all. From 2.0 onwards it tests against the dimensions you asked for:

```python
r = cp.Rectangle(re_length=8, im_length=4)   # window is 8x8; the domain is 8x4
r.contains(np.array([0 + 3j]))               # 1.x: True    2.0 and 3.0: False
```

The visible effect is that portraits, masks and exported STLs of non-square rectangles now show
the out-of-domain colour above and below the rectangle, where 1.x rendered function values. Pass
`square=False` if you want the window to match the domain.
