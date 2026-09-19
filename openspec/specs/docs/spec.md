# docs Specification

## Purpose
TBD - created by archiving change publish-rev3-docs-site. Update Purpose after archive.
## Requirements
### Requirement: The documentation site builds strictly from this repository

The project SHALL provide an MkDocs site that builds from the checked-out tree with
`mkdocs build --strict`, so that a navigation entry pointing at a missing page, or a broken
internal link, fails the build rather than reaching a reader. Every page listed in the navigation
SHALL exist. The build SHALL run in CI on every push. The site's dependencies SHALL be declared as
an installable `docs` extra, so the build does not depend on what a particular machine happens to
have installed.

#### Scenario: A missing page fails the build

- **WHEN** `mkdocs.yml` lists a page that does not exist, or a page links to a document that does
  not exist
- **THEN** `mkdocs build --strict` exits non-zero and CI fails

#### Scenario: The site builds from the declared extra

- **WHEN** the `docs` extra is installed into a clean environment and `mkdocs build --strict` runs
- **THEN** the build succeeds without further packages being installed by hand

### Requirement: The site covers the documented surface of the release

The site SHALL contain, at minimum: an overview and visual tour; installation and a first portrait;
how to read a phase portrait, including the phase-wheel legend and colour-vision-deficiency
guidance; domains and colormaps, including composite domains; 3D landscapes and the Riemann sphere
and relief, covering headless use and a reproducible camera; Riemann surfaces; engineering mode;
the distinction between the function catalog and the plot presets; the CLI; the STL and physical
workflow; an API reference; a migration guide; and contribution and release instructions.

Documentation SHALL describe the 3.0 surface. It SHALL NOT document removed entry points, the
`n_phi` argument, or the capability flags, except where a page is explicitly describing the
migration from an earlier version.

#### Scenario: The CLI page matches the CLI

- **WHEN** the CLI documentation is compared against `complexplorer --help`
- **THEN** every subcommand the program offers is documented, and no documented subcommand or
  option is absent from the program

#### Scenario: Claims about the project are checkable

- **WHEN** the documentation states a countable fact about the project, such as the size of the
  test suite
- **THEN** that number matches the repository at the time of the release

### Requirement: The API reference is generated from the source

The API reference SHALL be generated from docstrings rather than written by hand, and SHALL cover
every name exported in `complexplorer.__all__` and the `cp.ee` namespace, so that the reference
cannot drift from the code it documents.

#### Scenario: A public name reaches the reference

- **WHEN** a name is exported from `complexplorer.__all__` or `cp.ee`
- **THEN** it appears in the generated API reference with its docstring

### Requirement: Gallery assets reach the site without a second copy in the repository

The generated gallery page references images under `examples/gallery/`, which lies outside the
documentation source directory. Those assets SHALL be copied into the built site at build time.
The repository SHALL NOT carry a duplicate copy of the gallery images under `docs/`.

#### Scenario: Gallery images resolve in the built site

- **WHEN** the site is built and the gallery page is served
- **THEN** every image it references resolves, and no gallery image is committed twice

### Requirement: Deployment cannot overwrite the published 2.x site

The documentation deploy SHALL trigger only on a `v3.*` release tag, or by explicit manual
dispatch. It SHALL NOT trigger on a push to a branch. Until 3.0 is tagged, the published site
continues to document the released 2.x version.

#### Scenario: A branch push does not deploy

- **WHEN** a commit is pushed to any branch
- **THEN** the documentation site is built and checked, but not deployed

#### Scenario: A release tag deploys

- **WHEN** a `v3.*` tag is pushed
- **THEN** the site is built and published

### Requirement: The site states which version it documents

Every page SHALL make the documented version visible, taken from the installed
`complexplorer.__version__`. A site built from an untagged commit SHALL be labelled as a
development build, so a reader can tell released documentation from unreleased.

#### Scenario: An untagged build is labelled

- **WHEN** the site is built from a commit that is not a release tag
- **THEN** the version indicator marks it as a development build

### Requirement: Internal working notes are excluded from the site

Design histories, fix write-ups and planning documents SHALL be kept out of the published
navigation. They MAY remain in the repository under an internal directory, which the site excludes.

#### Scenario: A planning document is not published

- **WHEN** the site is built
- **THEN** documents under the internal directory appear in neither the navigation nor the built
  site

