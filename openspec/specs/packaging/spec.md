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

### Requirement: Linting and formatting enforced

The project SHALL enforce a consistent code style via `ruff` (lint and format), checked in CI,
across the package, the tests and the executable example scripts. The ruff version SHALL be pinned
to a known minor range wherever it is installed, so a formatting change in a new release cannot
fail the gate without a deliberate upgrade. Jupyter notebooks are exempt: their cell structure
legitimately conflicts with import-order and unused-expression rules.

#### Scenario: CI rejects unformatted or linting-violating code

- **WHEN** code that fails `ruff check` or is not `ruff format`-clean is submitted
- **THEN** the CI lint job fails

#### Scenario: Example scripts are held to the same standard

- **WHEN** a Python file under `examples/` is submitted
- **THEN** it is linted and format-checked like the package, with documented per-file exceptions
  where a script must configure a backend before importing it

#### Scenario: The lint tool cannot change underneath the project

- **WHEN** ruff publishes a new minor release
- **THEN** CI continues to use the pinned range, and adopting the new version is a deliberate change

### Requirement: Built distributions are verified before release

The project SHALL verify the distributions it publishes, not only the source tree. CI SHALL build
the wheel and the sdist, validate their metadata, inspect their contents, and install the **wheel**
into a fresh environment where it SHALL import, expose its console entry point, render in 2D,
render in 3D off-screen, and export an STL. The sdist SHALL be proven buildable by producing a
wheel from it.

#### Scenario: The wheel is installed and exercised in a clean environment

- **WHEN** the artifact job runs
- **THEN** it builds the wheel and sdist, and in an environment containing neither the repository
  checkout nor its development dependencies it installs the wheel and successfully runs: an import
  reporting `complexplorer.__version__`, the `complexplorer list` command, a 2D render written to a
  file, an off-screen PyVista render written to a file, and a small STL export

#### Scenario: Distribution contents are inspected

- **WHEN** the built wheel and sdist are inspected
- **THEN** the wheel contains `complexplorer/py.typed`, both license files, and the expected
  subpackages, and neither distribution contains tests, examples, gallery images or build debris

#### Scenario: Distribution metadata is inspected

- **WHEN** the wheel's metadata is inspected
- **THEN** it declares the version from `complexplorer/_version.py`, the supported
  `Requires-Python`, the runtime dependencies, the SPDX license expression, the project URLs, and
  the `complexplorer` console entry point, and `twine check` passes

#### Scenario: The sdist can build a wheel

- **WHEN** a wheel is built from the sdist in a clean environment
- **THEN** the build succeeds and the resulting wheel imports

### Requirement: CI covers the claimed platforms and dependency floors

CI SHALL run the test suite on every Python version the package claims to support, on Linux,
Windows and macOS, with no claimed version excluded from failing the build. CI SHALL additionally
run one lane that installs the **lowest** version of each declared direct dependency, so the
declared floors are exercised rather than assumed. A version that is not yet supported MAY run as
an explicitly non-blocking lane, and SHALL NOT be advertised in the package classifiers until that
lane passes.

#### Scenario: Every claimed version can fail the build

- **WHEN** the test suite fails on a Python version listed in the package classifiers
- **THEN** the CI run fails

#### Scenario: The declared dependency floors are exercised

- **WHEN** the minimum-dependency lane runs
- **THEN** each declared direct dependency is installed at its lowest allowed version and the test
  suite passes against them

#### Scenario: An unsupported version is visible but not blocking

- **WHEN** a lane for a Python version the package does not yet claim fails
- **THEN** the CI run still succeeds, and that version is absent from the classifiers

#### Scenario: Upstream breakage surfaces on a schedule

- **WHEN** the scheduled run installs the newest compatible dependencies
- **THEN** a failure is reported against the schedule, before the next release sprint

### Requirement: The public typing surface is gated in CI

CI SHALL type-check the public surface and a small downstream program that imports complexplorer
the way a user would. The gate SHALL cover what users can see rather than the whole package: a
whole-package gate is dominated by numpy scalar-versus-array unions and third-party stub
imprecision, which produces failures that are not defects and teaches contributors to add ignores.

#### Scenario: A wrong public annotation fails the build

- **WHEN** a public entry point is annotated in a way that rejects correct user code, or a public
  callable loses its return type
- **THEN** the typing job fails

### Requirement: The distribution points at the project's own resources

The distribution metadata SHALL declare, in addition to the homepage and issue tracker, the
documentation site, the changelog and the release notes, so that the package page leads a reader
to the material the project publishes rather than only to its source.

The repository SHALL carry citation metadata in a machine-readable form, and any citation example
in the documentation SHALL be derived from it rather than restating a hard-coded year.

#### Scenario: The package page links the documentation

- **WHEN** the built distribution's metadata is inspected
- **THEN** it declares project URLs for the documentation site, the changelog and the release
  notes, each resolving to a published location

#### Scenario: Citation metadata exists and agrees with the documentation

- **WHEN** the repository is inspected for citation metadata
- **THEN** a machine-readable citation file is present, and the version and year it states match
  the release being documented

### Requirement: The changelog states the release history truthfully

The changelog SHALL describe what was actually released. A version that was published SHALL NOT be
described as unpublished, and versions that were only internal milestones SHALL be marked as such.
The upgrade notes for a release SHALL be written against the version users are most likely to be
upgrading from, which is the latest published one.

The changelog SHALL carry comparison links for the unreleased range and for each released version.

#### Scenario: The published baseline is described correctly

- **WHEN** the changelog describes the release preceding this one
- **THEN** its publication state and date match the package index's record of it

### Requirement: Issue reporting collects what a rendering bug needs

The repository SHALL provide issue templates, including one for visual and rendering problems that
asks for the operating system, the Python version, the PyVista and VTK versions, the GPU or display
situation, and a minimal example. Rendering defects are not reproducible without them.

#### Scenario: A rendering report asks for the environment

- **WHEN** a contributor opens a visual or rendering issue
- **THEN** the template requests the operating system, Python version, PyVista and VTK versions,
  display or GPU situation, and a minimal reproducing example

