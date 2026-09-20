## ADDED Requirements

### Requirement: Releases are published from a tag

The distribution SHALL be published to PyPI only in response to a release **tag**, and SHALL NOT be
published in response to a push or merge to a branch.

A branch accumulates commits that are not releases — documentation, chores, specification
archives — so a branch-triggered publish depends on a condition that suppresses it for those
commits, and the cost of that condition being wrong is an upload that cannot be withdrawn, only
yanked. A tag states that a specific commit is a release, and omitting it publishes nothing.

#### Scenario: A merge to the default branch publishes nothing

- **WHEN** a pull request is merged to the default branch without a release tag
- **THEN** no distribution is uploaded to any package index

#### Scenario: A release tag triggers publication

- **WHEN** a release tag is pushed
- **THEN** the release workflow runs and, subject to the gates below, uploads the distribution

### Requirement: Publication uses a short-lived credential

Publication SHALL authenticate through PyPI's Trusted Publishing, using a short-lived identity
issued to the workflow run and verified by the index against the repository, workflow and
environment. The project SHALL NOT store a long-lived API token in repository secrets or require
one to be held by a maintainer for routine releases.

The job that uploads SHALL request identity-token permission; jobs that do not upload SHALL NOT.

#### Scenario: No upload token is stored

- **WHEN** the repository's configuration is inspected
- **THEN** no PyPI API token is present as a repository or organization secret, and the release
  workflow does not read one

#### Scenario: Only the publishing job can assume the identity

- **WHEN** the release workflow is inspected
- **THEN** identity-token permission is granted on the publishing jobs and not on the build job

### Requirement: The published artifact is the one that passed the gate

The release workflow SHALL build the distributions once, run the project's release artifact checks
against those files, and upload those same files. It SHALL NOT rebuild the distribution between
validation and upload.

The checks SHALL be the ones the project already documents as its release gate: metadata
validation, distribution inspection, installation of the wheel into an environment containing
neither the checkout nor the development dependencies, a smoke run executed from outside the
checkout, and a build from the source distribution.

#### Scenario: Validation and upload share one build

- **WHEN** a release runs
- **THEN** the artifacts uploaded to the index are byte-identical to those the gate checked, having
  been produced by a single build step in the same run

#### Scenario: A failed gate blocks the upload

- **WHEN** any release artifact check fails
- **THEN** no upload occurs to either index

### Requirement: The release tag agrees with the canonical version

Before building, the release workflow SHALL verify that the version named by the release tag equals
the canonical version exposed by the package, and SHALL fail the release when they differ.

The package already guarantees a single canonical version shared by runtime and distribution
metadata. A tag is a second, hand-written assertion of the same fact, and a disagreement can upload
a distribution under a version number nobody intended — which, being immutable, cannot be corrected
afterwards.

#### Scenario: A mismatched tag stops the release

- **WHEN** a release tag names a version that differs from the package's canonical version
- **THEN** the release fails before anything is built or uploaded, and the message names both
  versions

#### Scenario: A matching tag proceeds

- **WHEN** the tag and the canonical version agree
- **THEN** the release proceeds to the artifact gate

### Requirement: Rehearsal and approval precede an irreversible upload

The release workflow SHALL upload to a test index before uploading to PyPI, and the upload to PyPI
SHALL require an explicit human approval recorded against the run.

The two guard different failures and neither replaces the other. The test index surfaces
configuration faults — a publisher entry that does not match, metadata an index rejects that local
validation accepts — against an index where a version number can be spent harmlessly. The approval
surfaces intent faults, such as a tag pushed at the wrong commit or ahead of a pending fix, which
produce a perfectly valid artifact that no automated check can distinguish from a correct release.

#### Scenario: The test index is exercised first

- **WHEN** a release runs
- **THEN** the distribution is uploaded to the test index before any upload to PyPI is attempted

#### Scenario: PyPI upload waits for a person

- **WHEN** the release reaches the PyPI upload
- **THEN** it does not proceed until a reviewer approves the run

### Requirement: Re-releasing an existing version fails loudly

An attempt to publish a version already present on the index SHALL fail the release rather than be
skipped silently.

A re-run happens because something went wrong, and the question being asked is whether the fix
shipped. A run that reports success while uploading nothing answers that question wrongly and can
leave a partially completed release looking complete. Because a published version cannot be
replaced, the correct recovery is a new version rather than a retry.

#### Scenario: A duplicate version is refused

- **WHEN** a release is run for a version that already exists on PyPI
- **THEN** the release fails and reports that the version is already published
