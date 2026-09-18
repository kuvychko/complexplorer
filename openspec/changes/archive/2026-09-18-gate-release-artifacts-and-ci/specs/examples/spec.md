## MODIFIED Requirements

### Requirement: A notebook execution harness verifies the tutorials

The project SHALL provide a documented, repeatable way to verify notebook execution using `nbmake`
(`pytest --nbmake examples/notebooks/`). The notebook tooling (`nbmake`, `nbconvert`, `ipykernel`)
SHALL be declared as installable dependencies (an `[examples]` extra). The harness SHALL NOT be
collected by the default `pytest` run and SHALL NOT run on every push, because the notebooks drive
PyVista and take minutes. It SHALL, however, run automatically on a schedule and before a release,
so a tutorial that stopped working cannot ship.

#### Scenario: The harness verifies all notebooks on demand

- **WHEN** `pytest --nbmake examples/notebooks/` is run in an environment with the `[examples]`
  extra installed
- **THEN** every notebook is executed and the run passes only if all notebooks complete without a
  cell error

#### Scenario: The default test run does not execute notebooks

- **WHEN** the default `pytest` suite is collected
- **THEN** it does not execute the notebooks, keeping the per-push suite fast

#### Scenario: Notebooks are verified before a release

- **WHEN** a release tag is pushed, or the scheduled notebook run fires
- **THEN** the notebooks are executed headlessly and a cell error fails that run
