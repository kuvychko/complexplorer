## Context

The rev3 line (`docs/openspec-baseline`) forked from `main` at `c892bdb` on 2025-08-22. `main` then
produced the published **2.0.0** (tag `v2.0.0`, PyPI 2025-10-19). Comparing the two public
surfaces (`__all__` at `v2.0.0` against the current package) gives:

- **34 names exist only in 2.0.0:**
  - 9 colormaps
  - 12 extra exception classes
  - the logging API (4 functions)
  - `show` and the `publication`/`interactive`/`high_contrast` presets
  - the matplotlib 3D functions
  - `HAS_PYVISTA` / `HAS_STL_EXPORT`
- **7 names are new in 3.0:** `FunctionPreset`, `Presets`, `catalog`, `ee`, `generate_gallery`,
  `quick_plot`, and `riemann_surface_pv`.
- **One published rename is reverted:** 2.0 renamed `n_phi` → `phase_sectors`, and rev3 still
  uses `n_phi`.

The main `colormaps` spec already describes the 2.0 behavior: `phase_sectors`, the perceptual
family, and `ColormapError`. The rev3 code is therefore out of conformance with its own spec.

Relevant facts about the 2.0 source (`git show v2.0.0:<path>`):

- **`complexplorer/core/colormap.py`** (1775 lines). `BasePhasePortrait` (lines 120–289) holds
  the modulation logic and a `_compute_colors` hook. `Phase` adds unit-circle emphasis. The nine
  families subclass the base.
- **Imports.** The families import `sigmoid` and `circular_interpolate` from
  `core/functions.py`, which rev3 lacks. They also import `oklch_to_srgb`, `hsl_to_rgb`,
  `cubehelix`, `interpolate_hue`, and `clip_to_gamut` from `core/color_utils.py` (404 lines, not
  in rev3), plus a few values from `core/constants.py`.
- **`Phase._compute_colors` matches rev3's math.** With emphasis off, it computes
  `H = φ/2π`, `S = 1`, and `V = (V_φ + V_r)(1 − v_base)/2 + v_base`, which is exactly rev3's
  `Phase.hsv_tuple`.
- **`tests/unit/core/test_colormap.py`** has 45 tests, covering every family.

The rev3 colormap contract that the ports must honor is in `Colormap.hsv()`. It substitutes
non-finite `z` before `hsv_tuple` and repaints non-finite and out-of-domain points with
`out_of_domain_hsv`. That contract comes from `fix-colormap-nonfinite`. The phase-wheel legend
renders any colormap through `rgb()`.

## Goals / Non-Goals

**Goals:**
- Every colormap published in 2.0.0 exists in 3.0 with the same name, parameters, and visual
  intent, on rev3's contract.
- `phase_sectors` is the single canonical sector-count name everywhere, and `n_phi` fails loudly
  with guidance.
- `Phase` output, and therefore every committed gallery portrait, is pixel-identical.
- Every other 2.0.0 name 3.0 doesn't carry is accounted for in a migration inventory that CI
  enforces.
- CVD guidance ships in the tutorials.

**Non-Goals:**
- Porting the 2.0 logging API, the other 11 extra exception classes, `show`/`*_preset`,
  `utils/color.py`, or the plotting refactors. These are documented as removed or replaced
  instead.
- Porting `complexplorer/special.py`. It is optional and handled by the notebook triage below.
- Migration-guide prose and changelog edits (`prepare-3-0-release-notes`).
- Docs-site build and deploy (`publish-rev3-docs-site`).
- Joining the git histories. That is a release-runbook step, and the owner confirms it at
  release time.

## Decisions

### D1. Port the 2.0 colormap code, adapting it to the rev3 contract instead of rewriting it

Take `BasePhasePortrait`, the nine families, and `color_utils.py` verbatim from `v2.0.0`. Then
apply mechanical adaptation only:

- modern typing (`X | None`)
- rev3 imports (`complexplorer.exceptions`)
- ruff compliance

Each family implements `_compute_colors`, so `Colormap.hsv()` still wraps it and the non-finite
contract applies automatically.

- *Alternative:* re-derive the families from scratch. Rejected: it would change the published
  visuals 2.0 users rely on, for no benefit.

### D2. Rebase `Phase` on `BasePhasePortrait`, guarded by a golden fixture

Before touching `Phase`, record a golden `.npz` of `Phase(...).rgb(z)` on a fixed grid. The grid
includes zeros, poles, NaN, and inf. The configuration matrix covers basic, sectors only, linear
rings, log rings, both ring types, auto-scale, `v_base`, and `scale_radius`.

The rebased `Phase` must reproduce the fixture exactly (`np.array_equal`). This makes "committed
portraits don't change" a test, not a hope.

### D3. `phase_sectors` everywhere, with the old keyword rejected

Constructors take `phase_sectors` in the position `n_phi` occupied, so positional callers
(`Phase(6)`) are unaffected. A keyword-only `n_phi` sentinel parameter on the enhanced base,
`PolarChessboard`, and the families raises
`ValidationError("n_phi was renamed to phase_sectors; use Phase(phase_sectors=6)")`. The same
check guards the spec path: `cmap_from_spec` passes keys through, so an `n_phi` key hits the
same error.

Every call site is renamed:
- `core/presets.py` `cmap_spec`
- the CLI `--cmap phase:N` shorthand
- `api.py` `Presets`
- the renderer default colormaps
- the STL generator
- tests, notebooks, and docs

The PyVista renderers' `_REMOVED_KWARGS` entry `n_phi → resolution` refers to the removed 2.x
**mesh** argument and stays as it is. The design notes this so nobody "fixes" it.

- *Alternative:* keep `n_phi` as a deprecated alias. Rejected: the project has no deprecation
  policy, and both names would linger past a major version. The `n_theta` precedent from
  `harden-3-0-release` uses the same loud-failure pattern.

### D4. Add `ColormapError(ValidationError)`, and port only that exception

The `colormaps` spec already requires `ColormapError` for invalid pattern parameters. Porting it
as a `ValidationError` subclass keeps every existing handler working. It also restores
`Chessboard`/`PolarChessboard`/`LogRings` validation, which 2.0 had and rev3 lost.

The other 11 extra 2.0 exception classes are not ported: nothing in 3.0 raises them. The
inventory (D6) maps each to `ComplexplorerError`/`ValidationError`.

### D5. Port the helpers the families need, and keep them off the top level

`sigmoid` and `circular_interpolate` go into `core/functions.py`, and `color_utils.py` becomes
`core/color_utils.py`. None is added to the top-level `__all__`, since none was top-level in
2.0.0 either.

The ported classes reference no `constants.*` values (verified against the `v2.0.0` source), so
`core/constants.py` is not ported. Its unused `import` is dropped.

### D6. A migration inventory that CI enforces

Add `docs/migration-3.0.md` as a skeleton containing a machine-readable table: one row per
2.0.0-only public name, with its replacement or "removed — no replacement".
`prepare-3-0-release-notes` later wraps prose around it.

`tests/unit/test_v2_surface_accounted.py` works against a frozen `tests/data/v2_0_0_all.txt`,
the 2.0.0 `__all__`, so the test needs no git access. For every frozen name, it asserts that the
name is either in `complexplorer.__all__` or has a row in the inventory table. After this change
the table holds 24 rows:

- 11 exceptions
- 4 logging functions
- 3 `*_preset` helpers and `show`
- 3 matplotlib 3D functions
- 2 `HAS_*` flags

### D7. The gallery manifest schema version bumps

Renaming a key inside `cmap_spec` changes the shape of `index.json` records, so the bundle's
`schema_version` goes from 2 to 3 (it reached 2 in `enrich-answer-key-stats`; `showcase.json`
carries its own, separate version and is unaffected). That tells downstream consumers (web, Godot) the key moved.
The manifest stays byte-stable across runs.

Only `index.json` and the `card.json` files are committed from the regeneration. Portrait PNG
churn from environment sensitivity is reverted, because pixels are unchanged by D2.

### D8. Extend the colormap gallery in this change; it gets reviewed later

`examples/showcase.py` `_colormap_family()` gains one entry per ported family, each with an
explicit `phase_sectors` snippet. A minimal `--only colormaps` flag regenerates just that
section, so C1 doesn't re-render (and churn) every 3D screenshot.
`tests/unit/test_showcase_bundle.py` gains the "every exported colormap appears" check.

The new colormap PNGs still go through the visual review loop (round R3 of
`curate-rev3-visual-tour`), which may re-render them. `curate-rev3-visual-tour` extends `--only`
with `--out` and the other sections.

### D9. Notebook triage

| 2.0 notebook | Outcome |
|---|---|
| `05_accessibility_cvd` + `04_colormaps_comprehensive` | Ported and merged into a new `examples/notebooks/color_and_accessibility.ipynb`: the colormap tour, CVD simulation, and recommendations. |
| `02_domains_advanced` | Its composite-domain material is folded into `advanced_features.ipynb`, where it isn't already covered. |
| `01`, `03`, `06`, `07`, `08` | Not ported. They overlap rev3's four tutorials, and `03`/`06`/`07` call removed APIs (`HAS_PYVISTA`, 3D `riemann`, `plot_landscape`). |
| `app_01`–`app_04` (applications) | Ported mechanically to `examples/notebooks/applications/` only if each passes `pytest --nbmake` on the 3.0 surface. Each failure is recorded as a 3.1 backlog item. |

`app_02` needs `complexplorer.special` only if the probe below finds that import. If it does,
`app_02` is deferred rather than porting a 420-line module.

All notebooks keep the static PyVista backend and are committed with executed output, per the
`examples` spec.

### D10. Recover the MkDocs scaffold as inert files

Restore `mkdocs.yml`, `docs/javascripts/mathjax.js`, and `.github/workflows/docs.yml` from
`v2.0.0`. Change the workflow trigger to `workflow_dispatch` only, so merging this line can never
redeploy over the live v2 site. `publish-rev3-docs-site` owns the content, nav, strict build,
and tag-triggered deploy.

## Risks / Trade-offs

- **About 1,400 lines of ported colormap code arrive with only 45 family tests.**
  → Add a parametrized contract test over *every* exported colormap: shape, gamut, HSV/RGB
  consistency, non-finite/out-of-domain color, determinism, and legend rendering. It runs on top
  of the ported unit tests.
- **The `Phase` rebase silently changes pixels.** → The D2 golden fixture fails on any
  difference.
- **Downstream consumers of `index.json` read `cmap_spec.n_phi`.** → The schema-version bump
  (D7), plus the inventory and changelog entry.
- **A missed `n_phi` call site ships.** → The constructor rejection makes a missed site fail
  loudly in tests. A repo grep for colormap `n_phi=` (excluding the mesh mapping and the
  rejection tests) must return zero.
- **Application notebooks balloon the scope or slow CI.** → Nbmake is the gate; anything that
  fails is deferred, never hand-fixed at length. Runtime is noted for
  `gate-release-artifacts-and-ci`.
- **New colormap PNGs are committed before visual review.** → Accepted. They are 2D matplotlib
  renders, and R3 re-renders and reviews every colormap.

## Migration Plan

This change lands as ordinary commits on the rev3 line; there is no runtime migration. A 1.x or
rev3-build user renames `n_phi` → `phase_sectors`, and the error message says so. A 2.0.0 user
needs no colormap changes. The inventory feeds the 2.0.0 → 3.0 migration guide. Rollback is a
revert: the golden fixture and inventory test are self-contained.

## Open Questions

_None outstanding._ The one pre-implementation probe is resolved: `app_02_special_functions` uses
`scipy.special`, not `complexplorer.special`, so it goes through the normal nbmake gate in D9
and `special.py` stays out of scope.
