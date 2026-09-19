## ADDED Requirements

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

## MODIFIED Requirements

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

## REMOVED Requirements

### Requirement: Continuous integration across backend configurations

**Reason**: This required CI to run a configuration "without PyVista". PyVista became a required
core dependency at 3.0, so that configuration does not exist: there is nothing to install without,
and the 2D/core paths cannot be imported in isolation. The replacement requirement above states
what CI must actually cover — every claimed Python version on three operating systems, plus the
declared dependency floors.

**Migration**: None. Installing `complexplorer` always installs PyVista, and the headless 3D, mesh
and STL tests run on every lane.

### Requirement: 3D backend dependency strategy

**Reason**: This described the 2.x line, where PyVista was an optional extra under `pyvista`/`3d`
and "will be required starting at 3.0". 3.0 has arrived: PyVista is a core dependency and those
extras are empty no-op aliases kept only so existing install commands keep working. The decision to
keep PyVista required — with the measured install footprint and cold import time behind it — is
recorded in the backend policy.

**Migration**: `pip install complexplorer` installs PyVista. `complexplorer[pyvista]` and
`complexplorer[3d]` still resolve, and install nothing extra.
