# Design — add-phase-legend

## Context

Phase portraits encode phase as hue and modulus as shading; without an in-figure key readers
must know the convention. The legend must work for *all* four colormaps, not just `Phase`.

## Goals / Non-Goals

**Goals**: opt-in, colormap-faithful legend on `plot`/`pair_plot`; zero change to existing
output when off; no new dependencies.

**Non-Goals**: legends for the PyVista 3D renderers or `riemann_chart` (can be added later,
additively); a separate modulus colorbar (the wheel's radial shading already carries it for
enhanced portraits); tick/angle labels (kept minimal; refinement is additive).

## Decisions

1. **Render the legend through `Colormap.rgb()` on an identity grid** (`w = x + iy`,
   `|w| ≤ 1`) rather than drawing an HSV wheel by hand — this makes the legend exact for
   enhanced `Phase` (rings/sectors), `Chessboard`, `PolarChessboard`, and `LogRings` with no
   per-colormap code. The disk mask is applied as an alpha channel (RGBA), not via the
   colormap's out-of-domain color, so the legend floats cleanly over the portrait.
2. **Inset inside the axes (`ax.inset_axes`, upper-right, ~26% of the axes)** rather than a
   dedicated figure column: works identically whether the caller passed their own `ax`,
   composes with `pair_plot`'s two panels, and cannot be clipped by `tight_layout`/`savefig`
   (insets outside the axes bounding box can be). Occluding a corner is the standard map-legend
   trade-off; the feature is opt-in.
3. **Private helper `_draw_phase_legend`** — the public surface grows by exactly one keyword
   per function. Promoting the helper to a public API later is additive if demand appears.
4. **Unit disk, radius 1** — the natural normalization: hue at radius shows phase; for
   log-periodic enhanced portraits the ring pattern densifying toward the center reads
   correctly as "rings track modulus".

## Risks / Trade-offs

- [Legend occludes data in the corner] → opt-in feature; a thin white border visually
  separates it from the portrait beneath.
- [Auto-scaled colormaps (`auto_scale_r`) compute cell size from plotted data, so legend and
  portrait could disagree] → the legend uses the *same colormap instance* after the portrait
  has been rendered, so any portrait-derived state is already set.

## Migration Plan

Single commit; feature is opt-in and additive. Rollback = revert.

## Open Questions

_None._
