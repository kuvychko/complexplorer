# Migrating from 2.0.0 to 3.0

> **Status: skeleton.** The inventory below is generated and enforced by
> `tests/unit/test_v2_surface_accounted.py`. The task-oriented prose (what to do first, a full
> before/after example, troubleshooting) is written by the `prepare-3-0-release-notes` change.

Complexplorer 2.0.0 was published on 2025-10-19. This page covers upgrading from it to 3.0.

## Renamed: `n_phi` is now `phase_sectors`

2.0.0 renamed the phase-sector count from `n_phi` to `phase_sectors`; 3.0 keeps that name. Code
written against 2.0.0 needs no change. Code written against 1.x (or an unreleased 3.0 build)
does:

```python
cp.Phase(n_phi=6, auto_scale_r=True)          # 1.x
cp.Phase(phase_sectors=6, auto_scale_r=True)  # 2.0.0 and 3.0
```

Passing `n_phi` raises `ValidationError` naming the replacement. The same applies to
`PolarChessboard` and every perceptual colormap, to preset `cmap_spec` records, and to the CLI
`--cmap phase:N` shorthand.

> Unrelated: the PyVista renderers also accept `resolution`, which replaced a *different*
> `n_phi` — the 2.x sphere-mesh argument. That mapping is unchanged.

## Gallery manifest: `schema_version` 2 → 3

`index.json` records carry `cmap_spec`, whose sector-count key moved from `n_phi` to
`phase_sectors`. Consumers of the manifest should read the new key; the version
bump marks the change. The manifest remains byte-stable across runs of the same version.

## Removed names and their replacements

Every public name that 2.0.0 exported and 3.0 does not carry:

| 2.0.0 name | 3.0 replacement |
|---|---|
| `ComputationError` | `ComplexplorerError` — the base class covers computation failures |
| `DependencyError` | removed — PyVista is a required dependency, so the condition cannot arise |
| `DomainError` | `ValidationError` |
| `ExportError` | `ComplexplorerError` |
| `FunctionEvaluationError` | `ComplexplorerError` |
| `HAS_PYVISTA` | removed — PyVista is always installed |
| `HAS_STL_EXPORT` | removed — STL export is always available |
| `ImageExportError` | `ComplexplorerError` |
| `MeshGenerationError` | `ComplexplorerError` |
| `OptionalDependencyError` | removed — PyVista is a required dependency |
| `PyVistaNotAvailableError` | removed — PyVista is a required dependency |
| `ResolutionError` | `ValidationError` |
| `STLExportError` | `ComplexplorerError` |
| `disable_logging` | removed — use the standard `logging` module |
| `enable_debug_logging` | removed — use `logging.getLogger('complexplorer').setLevel(logging.DEBUG)` |
| `get_logger` | removed — use `logging.getLogger(__name__)` |
| `high_contrast_preset` | `Presets.high_contrast()` |
| `interactive_preset` | `Presets.interactive()` |
| `pair_plot_landscape` | `pair_plot_landscape_pv()` |
| `plot_landscape` | `plot_landscape_pv()` |
| `publication_preset` | `Presets.publication_ready()` |
| `riemann` | `riemann_pv()` (3D) or `riemann_chart()` / `riemann_hemispheres()` (2D charts) |
| `setup_logging` | removed — configure the standard library `logging` module directly |
| `show` | `quick_plot()` |

## Still present

Every other 2.0.0 name — including all 13 colormaps, the domains, `ModulusScaling`,
`quick_plot`, and the STL export surface — is unchanged in 3.0.
