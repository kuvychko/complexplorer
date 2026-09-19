## ADDED Requirements

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
