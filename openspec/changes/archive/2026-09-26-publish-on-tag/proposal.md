## Why

Releases are published by hand: `uv build`, then `twine upload` with an API token pasted in. That
works, and it has shipped 2.0.0 and 3.0.0 correctly. Two things are worth improving, and neither is
"it takes too long".

**The token is the weak part.** A PyPI API token with upload scope is a long-lived credential that
lives in a shell history, a password manager, or both, and that authorizes publishing under this
project's name from anywhere. PyPI's Trusted Publishing removes it: the workflow authenticates with
a short-lived OIDC identity that PyPI verifies against the repository, workflow file and
environment, and no secret is stored in the repository or on a laptop. The credential stops
existing rather than being better hidden.

**The manual gate and the automated gate have drifted apart in principle.**
`docs/development/contributing.md` documents a pre-tag checklist — build, `twine check`,
`check_distribution.py`, install the wheel into a throwaway environment, smoke it from another
directory, build from the sdist. CI's `artifact` job already runs exactly that on every push. But
nothing connects either of them to the act of publishing, so what reaches PyPI is whatever was on
the machine that ran `twine`. Publishing from the same run that passed the gate closes that gap.

A third motivation is negative and worth stating: **this must not publish on merge to `main`.**
`main` is not a release branch. Of the last fifteen commits, one is `docs: date the 3.0.0 release`,
one is `chore: join the 2.0 history into the rev3 line`, and three are OpenSpec archive commits.
Every one of those would need either a version bump or a skip-if-unchanged branch to maintain, and
PyPI is immutable — a mistaken publish can be yanked but never replaced or corrected. The
repository already made this decision once for documentation: `docs.yml` deploys only on a `v3.*`
tag, with a comment saying deployment is deliberately hard to trigger. The same reasoning applies
with more force to an irreversible upload.

## What Changes

- A release workflow triggered by a `v*` **tag push** only. No branch trigger, no publish on merge.
- The workflow builds the distributions once and runs the documented release gate against them:
  `twine check`, `scripts/check_distribution.py`, a clean-environment wheel install, the
  `scripts/smoke_wheel.py` run from outside the checkout, and a build from the sdist. The artifacts
  that are published are the ones that passed, from the same run — not a rebuild.
- A gate asserting that the tag and `complexplorer._version.__version__` agree. The packaging
  capability already requires a single canonical version; this extends that guarantee to the
  release tag, so `v3.1.0` cannot ship a package that calls itself something else.
- Publication to **TestPyPI first**, then to PyPI, both through Trusted Publishing with
  `id-token: write` and no stored token.
- The PyPI step runs in a protected GitHub Environment with a required reviewer, so an irreversible
  upload takes a deliberate human approval even once the tag is pushed.
- Re-running a release for a version already on PyPI fails loudly rather than being skipped
  silently, so a partially completed release cannot look successful.
- `docs/development/contributing.md` gains the release procedure: bump `_version.py`, land it, tag,
  approve. The manual commands stay documented — they are how a person reproduces the gate locally
  — but they stop being the publication path.

**Not in scope.** Automating the version bump, the changelog entry, or GitHub Release notes. The
bump stays a deliberate commit; the tag gate checks it rather than writing it. Signing and
attestation beyond what Trusted Publishing provides is a separate question.

## Capabilities

### Modified Capabilities

- `packaging`: publication is defined as part of the capability — released from a tag, through
  Trusted Publishing rather than a stored credential, gated on the artifact checks and on tag/
  version agreement, and requiring human approval.

## Impact

- **New:** `.github/workflows/release.yml`.
- **Config, outside the repository:** a PyPI and a TestPyPI Trusted Publisher entry for this
  repository and workflow; a `pypi` GitHub Environment with a required reviewer. These cannot be
  created by a commit and must be done in the PyPI and GitHub UIs before the first tag.
- **Docs:** `docs/development/contributing.md` release section.
- **Risk:** the first run is the one that matters, because a Trusted Publisher misconfiguration
  surfaces only at upload. The TestPyPI step exists to surface it against a throwaway index first,
  and the change is deliberately landed and rehearsed ahead of 3.1.0 rather than during it.
- **No library code changes**, so this cannot affect the released package's behaviour.
