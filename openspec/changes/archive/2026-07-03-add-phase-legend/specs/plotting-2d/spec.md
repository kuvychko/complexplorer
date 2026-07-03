# 2D Plotting — Delta for add-phase-legend

## ADDED Requirements

### Requirement: Phase legend inset

The 2D plotting entry points SHALL support an opt-in legend: when `legend=True`, `plot` and
`pair_plot` draw a small inset showing the unit disk colored by the same colormap instance used
for the portrait (the identity map over the disk), so the mapping from hue to phase and from
shading to modulus can be read off the figure. The legend SHALL be rendered through the
colormap's own RGB pipeline (faithful for any colormap), SHALL be clipped to a circle with the
outside transparent, and SHALL default to off.

#### Scenario: Legend reflects the active colormap

- **WHEN** `plot` is called with `legend=True` and any colormap
- **THEN** an inset axes is added to the plot showing the unit disk colored by that same colormap applied to the identity map, with pixels outside the disk fully transparent

#### Scenario: Legend is off by default

- **WHEN** `plot` or `pair_plot` is called without the `legend` argument
- **THEN** no inset axes is added and the rendered figure is unchanged from previous releases

#### Scenario: Pair plot legend sits on the codomain panel

- **WHEN** `pair_plot` is called with `legend=True`
- **THEN** the legend inset appears on the codomain panel only
