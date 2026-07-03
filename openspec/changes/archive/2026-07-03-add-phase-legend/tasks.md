# Tasks — add-phase-legend

## 1. Implementation

- [x] 1.1 Add `_draw_phase_legend(ax, cmap)` helper to `plotting/matplotlib/plot_2d.py`:
      identity grid over the unit disk → `cmap.rgb()` → RGBA with transparent exterior →
      `ax.inset_axes` upper-right with circular border
- [x] 1.2 Add `legend: bool = False` to `plot()` (both the `ax=None` and explicit-`ax` paths,
      drawn before `savefig`)
- [x] 1.3 Add `legend: bool = False` to `pair_plot()`, forwarded to the codomain panel only

## 2. Tests and docs

- [x] 2.1 Unit tests: legend off by default (no inset axes); `legend=True` adds an inset with
      an RGBA image whose corner pixels are transparent; works for all four colormaps;
      pair_plot puts the inset on the codomain panel only
- [x] 2.2 `CHANGELOG.md` Added entry
- [x] 2.3 Off-screen visual check: render a legend'd `plot()` to PNG and inspect

## 3. Verification

- [x] 3.1 `pytest tests/` green; ruff clean; `openspec validate --specs` passes
