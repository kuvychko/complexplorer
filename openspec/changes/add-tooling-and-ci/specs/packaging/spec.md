# Packaging

## ADDED Requirements

### Requirement: Continuous integration across backend configurations

The project SHALL run automated CI on push and pull request that executes the test suite in
both a base configuration (without PyVista) and a configuration with PyVista installed and
offscreen rendering enabled, across the supported Python versions.

#### Scenario: Base configuration excludes the 3D backend

- **WHEN** CI runs the base configuration
- **THEN** the test suite passes without PyVista installed, confirming the 2D/core paths do
  not require the 3D backend

#### Scenario: PyVista configuration runs 3D tests headlessly

- **WHEN** CI runs the PyVista configuration
- **THEN** PyVista is installed, offscreen rendering is enabled, and the 3D / mesh / STL
  tests execute and pass

### Requirement: Linting and formatting enforced

The project SHALL enforce a consistent code style via `ruff` (lint and format), checked in
CI.

#### Scenario: CI rejects unformatted or linting-violating code

- **WHEN** code that fails `ruff check` or is not `ruff format`-clean is submitted
- **THEN** the CI lint job fails
