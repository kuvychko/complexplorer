## Why

Complexplorer **2.0.0 was published to PyPI on 2025-10-19** from tag `v2.0.0` on `main`, but the
rev3 line forked from `main` at `c892bdb` (2025-08-22, just after 1.0.1) and never contained it.
Shipping 3.0 as it stands would silently take away features that 2.0.0 users have installed today:

- nine colormaps
- the `phase_sectors` parameter name
- the color-vision-deficiency guidance

The `colormaps` spec already describes that 2.0 behavior, which the rev3 code no longer has. The
gallery, docs, and migration guide will all describe the 3.0 surface, so that surface has to be
reconciled with the published release first.

## What Changes

**Colormap machinery**
- Port the 2.0 enhanced-phase base `BasePhasePortrait`, which provides the shared phase-sector /
  modulus-ring modulation and auto-scaled square cells. Rebase `Phase` on it. For every existing
  parameter combination, `Phase` output stays pixel-identical.
- Port `Phase`'s unit-circle emphasis options. This is additive and off by default.

**Nine colormaps**
- Port the colormap families 2.0.0 shipped and rev3 lacks: `OklabPhase`, `PerceptualPastel`,
  `AnalogousWedge`, `DivergingWarmCool`, `Isoluminant`, `CubehelixPhase`, `InkPaper`,
  `EarthTopographic`, and `FourQuadrant`.
- Port their OKLab/OkLCh, HSL, and cubehelix color utilities, including gamut clipping.
- Each family is ported onto rev3's colormap contract: non-finite values and out-of-domain points
  get the neutral color, and the families work with the phase-wheel legend.

**`phase_sectors` is canonical** (**BREAKING** relative to 1.x and the unreleased rev3 line;
non-breaking relative to 2.0.0)
- `phase_sectors` replaces `n_phi` as the phase-sector count on `Phase`, `PolarChessboard`, and
  every ported family. It also replaces it in:
  - preset `cmap_spec` records and `cmap_from_spec`
  - the CLI `--cmap` shorthand
  - the high-level `Presets` bundles
  - renderer default colormaps
- Passing `n_phi=` to a colormap raises `ValidationError` naming `phase_sectors`.
- Out of scope: the PyVista renderers' existing mapping of the removed 2.x *mesh* argument
  `n_phi` → `resolution` stays as it is.

**Colormap validation**
- A non-positive spacing, sector count, or ring base raises `ColormapError`, a new
  `ValidationError` subclass that the `colormaps` spec already requires.

**Gallery and examples**
- Regenerate `examples/gallery/index.json`, because each `cmap_spec` key changes. The manifest
  stays byte-stable across runs; this is a one-time change across versions.
- The colormap gallery covers every public colormap, including the ported families.
- Port the 2.0 color-vision-deficiency (CVD) accessibility notebook.
- Fold the 2.0 advanced-domains and comprehensive-colormaps material into the 3.0 notebook set.
- Port a 2.0 application notebook only if it executes under `nbmake`. Otherwise it goes to the
  backlog.

**2.0.0 surface accounting**
- Every other 2.0.0 public name that 3.0 does not carry is recorded, with its replacement, in a
  migration inventory: `docs/migration-3.0.md`, a skeleton whose prose is written later by
  `prepare-3-0-release-notes`. Examples: the logging API, the extra exception classes,
  `show`/`*_preset`, the matplotlib 3D functions, and the `HAS_*` flags.
- A test compares the frozen 2.0.0 `__all__` against 3.0's, so no removal goes undocumented.

**Docs scaffold**
- Recover the 2.0 MkDocs scaffold (`mkdocs.yml`, `docs/javascripts/mathjax.js`, and
  `.github/workflows/docs.yml`) for `publish-rev3-docs-site`. The workflow stays manual-trigger
  only, so nothing on this line can overwrite the live v2 site.

**Out of scope**
- Joining the git histories (`git merge -s ours origin/main`): a release-time, owner-confirmed
  step in the release runbook.
- Docs-site content.
- Migration-guide prose.

## Capabilities

### New Capabilities

_None._

### Modified Capabilities

- `colormaps`:
  - `phase_sectors` is the only phase-sector parameter; `n_phi` is rejected with the replacement
    named.
  - Enhanced phase colormaps share the `BasePhasePortrait` modulation contract, and `Phase` gains
    optional unit-circle emphasis.
  - The perceptual family honors the non-finite / out-of-domain contract.
  - Invalid pattern parameters raise `ColormapError`.
- `exceptions`: adds `ColormapError`, a `ValidationError` subclass.
- `examples`:
  - The colormap gallery and notebooks cover the full public colormap set, including the
    perceptual families; the "non-existent perceptual family" wording is dropped.
  - The tutorials include color-vision-deficiency guidance.

## Impact

- **Code:**
  - `complexplorer/core/colormap.py`: the base class, the nine families, and the rename.
  - New `complexplorer/core/color_utils.py`.
  - `complexplorer/core/functions.py`, if `sigmoid`/`circular_interpolate` are needed.
  - `complexplorer/exceptions.py`.
  - Package `__init__` re-exports.
  - Every `n_phi` call site: `core/presets.py`, `api.py`, `cli/main.py`, the 2D and PyVista
    plotting modules, `mesh/`, and `export/stl/ornament_generator.py`.
- **Tests:**
  - Port the 2.0 colormap tests (45 in `tests/unit/core/test_colormap.py`) to the rev3 contract.
  - Update the ~70 test references to `n_phi`.
  - Add the `n_phi` rejection, pixel-identity, non-finite, and 2.0-surface-accounting tests.
- **Data:** `examples/gallery/index.json` and the `card.json` files are regenerated, and the
  showcase colormap gallery gains the ported families.
- **Docs and examples:**
  - Notebooks: `phase_sectors`, the new CVD notebook, and the folded material.
  - `README.md`, `CLAUDE.md`, and `docs/*` snippets.
  - New `docs/migration-3.0.md` inventory.
  - The recovered MkDocs scaffold.
- **Compatibility:** 2.0.0 users keep `phase_sectors` and every colormap. Users coming from 1.x
  or rev3 builds must rename `n_phi` → `phase_sectors`.
