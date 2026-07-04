## ADDED Requirements

### Requirement: Keyword arguments are validated, not silently forwarded

The PyVista landscape entry points (`plot_landscape_pv`, `pair_plot_landscape_pv`) SHALL accept
only their documented parameters and SHALL NOT forward arbitrary keyword arguments into
`pyvista.Plotter`. An unrecognized keyword argument SHALL raise a `ValidationError`; for keyword
arguments removed in the 3.0 API migration, the error message SHALL name the current replacement
(e.g. `n_theta`/`n_phi` → `resolution`, `show` → `interactive`).

#### Scenario: Unknown keyword argument is rejected

- **WHEN** a landscape function is called with a keyword argument that is not part of its documented signature (e.g. `n_theta=200`)
- **THEN** a `ValidationError` is raised naming the offending argument (and its replacement when it is a known-removed 2.x name), rather than a raw `TypeError` from `pyvista.Plotter` or a silent no-op

## MODIFIED Requirements

### Requirement: Paired interactive landscape

The library SHALL render domain and codomain landscapes in two linked PyVista viewports that share
camera movement when interactive. A supplied `title` SHALL be applied as a figure-level title over
the paired scene and SHALL NOT replace the codomain panel's label.

#### Scenario: Linked dual viewports

- **WHEN** `pair_plot_landscape_pv` is called interactively
- **THEN** the identity-over-domain and `f(z)` surfaces appear in two viewports whose cameras are linked

#### Scenario: Title is a figure title, not a panel label

- **WHEN** `pair_plot_landscape_pv` is called with a `title`
- **THEN** the title is shown for the overall figure and the codomain panel keeps its own label (e.g. `Codomain f(z)`)
