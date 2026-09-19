## ADDED Requirements

### Requirement: Notebooks execute top-to-bottom on the 3.0 surface

Every tutorial notebook under `examples/notebooks/` SHALL execute top-to-bottom without error on
the 3.0 API surface. No notebook SHALL call a symbol removed at 3.0 (`plot_landscape`,
`pair_plot_landscape`, the 3D `riemann`, `HAS_PYVISTA`, `HAS_STL_EXPORT`); 3D cells SHALL use the
PyVista `*_pv` functions. Execution SHALL be reproducible via a documented command.

#### Scenario: Each notebook runs without a cell error

- **WHEN** `examples/notebooks/*.ipynb` are executed (via the verification harness)
- **THEN** every notebook runs to completion with no cell raising an error

#### Scenario: No notebook references a removed symbol

- **WHEN** the notebook sources are scanned
- **THEN** none call `cp.plot_landscape`, `cp.pair_plot_landscape`, the 3D `cp.riemann`, or
  reference `HAS_PYVISTA` / `HAS_STL_EXPORT`

### Requirement: Notebooks render via the static PyVista backend with committed output

Each notebook SHALL select the static PyVista backend (`pv.set_jupyter_backend('static')`) in a
setup cell so its `*_pv` calls render as embedded static images and execute headlessly. Notebooks
SHALL be committed with their executed output (the static images), so they render on GitHub /
nbviewer; the committed output SHALL be reproducible by re-executing the notebook. A note SHALL
point readers to the interactive / high-quality path (`notebook=False` or the terminal scripts).

#### Scenario: A setup cell selects the static backend

- **WHEN** a notebook is opened
- **THEN** an early cell sets `pv.set_jupyter_backend('static')` and a note explains how to switch
  to an interactive backend for higher-quality exploration

#### Scenario: Committed notebooks carry executed image output

- **WHEN** a committed notebook is viewed without running it
- **THEN** its 2D and 3D cells show embedded image output (no stale matplotlib-3D renders remain)

### Requirement: Notebooks cover the 3.0 feature surface

The tutorials SHALL cover the headline 3.0 additions: a Riemann-**surface** example
(`riemann_surface_pv`) and the preset registry / gallery workflow (`cp.catalog`, and a pointer to
the gallery producer). Colormap material SHALL reference only implemented colormaps
(`Phase`, `Chessboard`, `PolarChessboard`, `LogRings`) — not the non-existent perceptual family.

#### Scenario: Riemann surfaces and the registry are demonstrated

- **WHEN** the tutorial set is read
- **THEN** at least one notebook demonstrates `riemann_surface_pv` and at least one demonstrates
  the preset registry (`cp.catalog`) with a pointer to the gallery producer

### Requirement: A notebook execution harness verifies the tutorials

The project SHALL provide a documented, repeatable way to verify notebook execution using
`nbmake` (`pytest --nbmake examples/notebooks/`). The notebook tooling (`nbmake`, `nbconvert`,
`ipykernel`) SHALL be declared as installable dependencies (an `[examples]` extra). The harness
SHALL be opt-in — it SHALL NOT be collected by the default `pytest` run and SHALL NOT be required
in CI.

#### Scenario: The harness verifies all notebooks on demand

- **WHEN** `pytest --nbmake examples/notebooks/` is run in an environment with the `[examples]`
  extra installed
- **THEN** every notebook is executed and the run passes only if all notebooks complete without a
  cell error

#### Scenario: The default test run does not execute notebooks

- **WHEN** the default `pytest` suite is collected
- **THEN** it does not execute the notebooks (the nbmake harness is opt-in, keeping the default
  suite fast and CI free of PyVista-heavy notebook execution)
