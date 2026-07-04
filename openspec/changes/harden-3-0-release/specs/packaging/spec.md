## MODIFIED Requirements

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

## ADDED Requirements

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
