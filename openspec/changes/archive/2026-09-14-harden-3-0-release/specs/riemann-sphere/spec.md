## ADDED Requirements

### Requirement: Riemann sphere renderer validates its keyword arguments

`riemann_pv` SHALL accept only its documented parameters and SHALL NOT forward arbitrary keyword
arguments into `pyvista.Plotter`. An unrecognized keyword argument SHALL raise a `ValidationError`;
for keyword arguments removed in the 3.0 API migration the message SHALL name the current
replacement (e.g. `n_theta`/`n_phi` → `resolution`, `show` → `interactive`). A request for no
interactive window SHALL be honored rather than silently dropped.

#### Scenario: Unknown keyword argument is rejected

- **WHEN** `riemann_pv` is called with a keyword argument outside its documented signature (e.g. `n_theta=200`)
- **THEN** a `ValidationError` is raised naming the offending argument (and its replacement when it is a known-removed 2.x name), rather than a raw `TypeError` or a silent no-op

#### Scenario: Suppressing the interactive window is honored

- **WHEN** `riemann_pv` is asked not to open an interactive window (via the documented `interactive` parameter)
- **THEN** no interactive window is shown
