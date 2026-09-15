# Packaging

## Purpose

The packaging capability governs the distribution metadata of the library: how its
version is defined and exposed, and how its license terms are declared. It ensures the
version string has a single source of truth shared by runtime and distribution metadata,
and that the declared license metadata is consistent with the repository's license files,
including the separate terms governing generated artistic artifacts.

## Requirements

### Requirement: Single canonical version

The package SHALL expose exactly one canonical version string, defined in
`complexplorer/_version.py`, and all other version-bearing metadata SHALL derive from it
rather than restate it.

#### Scenario: Runtime and distribution versions agree

- **WHEN** the package is built and installed
- **THEN** `complexplorer.__version__` and the installed distribution metadata version
  report the same value, equal to `__version__` in `complexplorer/_version.py`

#### Scenario: No hardcoded duplicate version

- **WHEN** `pyproject.toml` is inspected
- **THEN** it declares the version dynamically (sourced from
  `complexplorer._version.__version__`) and does not hardcode a separate literal version

### Requirement: License metadata consistency

The package's declared license metadata SHALL agree with the repository license files,
and the dual-licensing of generated artistic artifacts SHALL be documented. The distribution
SHALL declare its code license via a modern SPDX `license` expression in `[project]`, naming
the same license as the `LICENSE` file, and SHALL NOT also carry a deprecated
`License :: ...` trove classifier for the same license.

#### Scenario: SPDX license key matches the LICENSE file

- **WHEN** the `[project].license` SPDX expression in `pyproject.toml` is compared with the `LICENSE` file
- **THEN** they name the same license (`MIT`), the `[build-system].requires` pins a setuptools new enough to accept the SPDX expression, and no `License :: OSI Approved :: ...` classifier remains

#### Scenario: License ships in the distribution metadata

- **WHEN** the built wheel's METADATA is inspected
- **THEN** it carries the MIT license and both `LICENSE` and `LICENSE.art` are included in the distribution

#### Scenario: Artistic outputs licensing is documented

- **WHEN** a user looks for the terms governing generated artistic/STL artifacts
- **THEN** the documentation states these are covered by `LICENSE.art` (CC-BY-NC 4.0),
  distinct from the code license

### Requirement: Typed, PyPI-ready distribution

The distribution SHALL advertise its inline type information per PEP 561 by shipping a
`complexplorer/py.typed` marker in the wheel. The user-facing "everything" extra SHALL install
only runtime features, not development tooling. The PyPI long description (`README.md`) SHALL
reference images and links by absolute URL so the project page renders outside the repository.

#### Scenario: py.typed ships in the wheel

- **WHEN** the built wheel is inspected
- **THEN** it contains `complexplorer/py.typed`, so downstream type checkers honor the package's annotations

#### Scenario: The user extra excludes dev tooling

- **WHEN** a user installs the `all` extra
- **THEN** only user-facing optional features (e.g. interactive Qt support) are installed, and test/lint/build tooling is confined to a separate development extra

#### Scenario: README renders on PyPI

- **WHEN** the README is rendered as the PyPI project description
- **THEN** its images and documentation links resolve via absolute URLs (no repository-relative paths), and no placeholder project URL (such as `github.com/user/...`) remains

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

### Requirement: 3D backend dependency strategy

The package SHALL provide PyVista as the 3D backend through an optional extra in the 2.x
line, exposed under both `pyvista` and a `3d` alias, and SHALL document that PyVista
becomes a required dependency at 3.0.

#### Scenario: PyVista installable via either extra name

- **WHEN** a user installs `complexplorer[3d]` or `complexplorer[pyvista]`
- **THEN** PyVista is installed and the same set of 3D features is enabled

#### Scenario: 3D backend policy is discoverable

- **WHEN** a user reads the installation/backend documentation
- **THEN** it states that matplotlib serves 2D and PyVista serves 3D, that new 3D features
  are PyVista-only, and that PyVista will be required starting at 3.0
