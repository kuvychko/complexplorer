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
and the dual-licensing of generated artistic artifacts SHALL be documented.

#### Scenario: Classifier matches the LICENSE file

- **WHEN** the `pyproject.toml` license classifier is compared with the `LICENSE` file
- **THEN** they name the same license

#### Scenario: Artistic outputs licensing is documented

- **WHEN** a user looks for the terms governing generated artistic/STL artifacts
- **THEN** the documentation states these are covered by `LICENSE.art` (CC-BY-NC 4.0),
  distinct from the code license
