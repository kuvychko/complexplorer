# Add Phase-Wheel Legend to 2D Plots

## Why

No renderer in the library offers a legend: a reader of a phase portrait has no in-figure key
for decoding hue → phase or shading → modulus, which weakens publication and teaching use — the
library's core 2D audience. A phase color wheel rendered through the *active* colormap's own
pipeline is the standard remedy (cf. Wegert's phase-portrait legends) and is a small, additive
feature worth shipping with 3.0.0.

## What Changes

- `plot()` and `pair_plot()` in `plotting/matplotlib/plot_2d.py` gain a `legend: bool = False`
  keyword. When true, a small inset in the upper-right corner of the (codomain) axes shows the
  unit disk colored by the same `Colormap` instance used for the portrait (identity map
  `w = r·e^{iθ}`, `r ∈ (0, 1]`), clipped to a circle with a thin border; the area outside the
  disk is transparent.
- Because the legend is rendered via the colormap's own `rgb()` pipeline, it is automatically
  faithful for every colormap: enhanced `Phase` shows its modulus rings/sectors, `Chessboard`,
  `PolarChessboard`, and `LogRings` show their own patterns.
- `pair_plot(legend=True)` draws the legend on the codomain panel only (the domain panel is an
  identity portrait — it already is its own legend).
- Default remains `legend=False` — existing figures are unchanged; the gallery's byte-stable
  output is unaffected.

## Capabilities

### New Capabilities

_None._

### Modified Capabilities

- `plotting-2d`: new requirement — an opt-in phase-legend inset on the domain-coloring plot and
  the pair plot, rendered through the active colormap.

## Impact

- `complexplorer/plotting/matplotlib/plot_2d.py`: new private helper `_draw_phase_legend(ax,
  cmap)` + `legend` kwarg on `plot`/`pair_plot`.
- Tests: `tests/unit/plotting/matplotlib/test_plot_2d.py` gains legend cases (inset axes
  created, image is RGBA with transparent corners, off by default).
- `CHANGELOG.md` Added entry. No dependency changes; no effect on PyVista paths or gallery
  determinism.
