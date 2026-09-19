# 3D Plotting (Matplotlib)

## ADDED Requirements

### Requirement: Matplotlib 3D paths are deprecated toward PyVista-only

The matplotlib 3D entry points SHALL be deprecated in favor of their PyVista equivalents
and signal their removal at 3.0. This applies to the 3D surface functions only; the 2D
stereographic charts (`riemann_chart`, `riemann_hemispheres`) are unaffected.

#### Scenario: Deprecation warning on matplotlib 3D landscape

- **WHEN** `plot_landscape` or `pair_plot_landscape` is called
- **THEN** a `DeprecationWarning` is emitted naming the PyVista replacement
  (`plot_landscape_pv` / `pair_plot_landscape_pv`) and stating removal at 3.0, and the
  plot is still produced

#### Scenario: Deprecation warning on matplotlib 3D Riemann sphere

- **WHEN** the 3D `riemann()` surface function is called
- **THEN** a `DeprecationWarning` is emitted naming `riemann_pv` as the replacement and
  stating removal at 3.0, and the plot is still produced

#### Scenario: 2D stereographic charts are not deprecated

- **WHEN** `riemann_chart` or `riemann_hemispheres` is called
- **THEN** no backend-deprecation warning is emitted, because these are matplotlib 2D
  features that remain supported
