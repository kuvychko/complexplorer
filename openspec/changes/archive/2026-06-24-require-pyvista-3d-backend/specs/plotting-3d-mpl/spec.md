# 3D Plotting (matplotlib)

The matplotlib 3D backend is removed in 3.0 (deprecated since 2.1). 3D landscapes and the 3D
Riemann surface are PyVista-only; the 2D stereographic charts (`riemann_chart`,
`riemann_hemispheres`) are unaffected and remain in the 2D matplotlib capability. This
capability is retired in full.

## REMOVED Requirements

### Requirement: Analytic landscape surface

**Reason:** matplotlib 3D rendering removed in 3.0; use `plot_landscape_pv` (PyVista).

### Requirement: Paired landscape

**Reason:** matplotlib 3D rendering removed in 3.0; use `pair_plot_landscape_pv` (PyVista).

### Requirement: Riemann sphere surface

**Reason:** matplotlib 3D rendering removed in 3.0; use `riemann_pv` (PyVista).

### Requirement: Matplotlib 3D paths are deprecated toward PyVista-only

**Reason:** The deprecation is now complete — the matplotlib 3D paths are removed, so the
deprecation requirement itself no longer applies.
