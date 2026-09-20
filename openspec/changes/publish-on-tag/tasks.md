## 0. Prerequisites (outside the repository)

> These cannot be done by a commit, and the workflow publishes nothing until they exist. Do them
> first, or the first tag fails at the upload step with an authentication error that looks like a
> workflow bug.

- [ ] 0.1 Create a Trusted Publisher on **PyPI** for `complexplorer`: this repository, workflow
  `release.yml`, environment `pypi`.
- [ ] 0.2 Create a Trusted Publisher on **TestPyPI** for the same, environment `testpypi`.
- [ ] 0.3 Create the `pypi` GitHub Environment with a required reviewer. Create `testpypi` without
  one — the rehearsal should not need an approval to be useful.
- [ ] 0.4 Confirm no PyPI API token remains as a repository or organization secret; remove it if
  one is there. Keep the personal token until the first real release succeeds, then revoke it.

## 1. The release workflow

- [ ] 1.1 Add `.github/workflows/release.yml`, triggered on `push` of tags matching `v*` and on
  `workflow_dispatch`. No branch trigger.
- [ ] 1.2 Set `permissions: contents: read` at workflow level; grant `id-token: write` on the two
  publish jobs only, not on the build job.
- [ ] 1.3 Add a `concurrency` group so two tags pushed together cannot race, and do **not** cancel
  in progress — an interrupted upload is worse than a queued one.

## 2. Tag/version gate

- [ ] 2.1 As the first step of the build job, compare the tag (stripped of its leading `v`) against
  `complexplorer._version.__version__` read from the checkout, and fail on a mismatch with a message
  naming both.
- [ ] 2.2 Read the version without installing the package — `_version.py` has no imports, so
  parsing it directly avoids needing a built environment before the gate runs.

## 3. The artifact gate

> These steps already exist in `ci.yml`'s `artifact` job. Duplicating them is deliberate — see
> `design.md`. Keep them textually close so a future extraction into a composite action is easy.

- [ ] 3.1 `uv build`, once. Everything downstream consumes this `dist/`.
- [ ] 3.2 `twine check dist/*`.
- [ ] 3.3 `python scripts/check_distribution.py dist`.
- [ ] 3.4 Install the wheel into a throwaway environment with neither the checkout nor the dev
  dependencies, and run `scripts/smoke_wheel.py` from outside the checkout. Needs
  `PYVISTA_OFF_SCREEN` and the headless display action, as in CI.
- [ ] 3.5 Build from the sdist and import the result.
- [ ] 3.6 Upload `dist/` as a workflow artifact; the publish jobs download it rather than
  rebuilding.

## 4. Publish

- [ ] 4.1 A `testpypi` job: `needs: build`, environment `testpypi`, downloads the artifact, publishes
  with `pypa/gh-action-pypi-publish` pointed at the TestPyPI repository URL.
- [ ] 4.2 A `pypi` job: `needs: testpypi`, environment `pypi` (the one with the required reviewer),
  same artifact, default index.
- [ ] 4.3 Do **not** pass `skip-existing` on the PyPI job — a duplicate version must fail. It is
  acceptable on TestPyPI, where a rehearsal may legitimately repeat a version.
- [ ] 4.4 Pin the publish action to a commit SHA rather than a tag: it handles the upload identity,
  so it is the one action here worth pinning immutably.

## 5. Docs

- [ ] 5.1 Add the release procedure to `docs/development/contributing.md`: bump `_version.py`, land
  it, add the changelog entry, tag `vX.Y.Z`, push the tag, approve the `pypi` environment.
- [ ] 5.2 Keep the existing manual gate commands, reframed as "how to reproduce the gate locally"
  rather than as the release path.
- [ ] 5.3 State that the version bump, changelog and release notes stay manual, and why.
- [ ] 5.4 Note that a bad release is recovered with a new version, never a retry, because PyPI
  versions are immutable.

## 6. Verification

- [ ] 6.1 `openspec validate publish-on-tag` and `openspec validate --specs`.
- [ ] 6.2 Actionlint (or equivalent) over the new workflow.
- [ ] 6.3 **Rehearse before 3.1.0, not during it.** Tag a prerelease — `v3.1.0rc1` against a tree
  whose `_version.py` says `3.1.0rc1` — and confirm: the gate passes, TestPyPI receives the upload,
  the `pypi` job waits for approval, and approving it publishes. A prerelease is the cheapest
  honest end-to-end test, and PyPI hides prereleases from the default install.
- [ ] 6.4 Verify the tag gate fires: push a deliberately mismatched tag to a scratch branch and
  confirm the run fails before building.
- [ ] 6.5 Verify the duplicate-version guard by re-running the release for the already-published
  prerelease and confirming it fails rather than reporting success.
- [ ] 6.6 After the first successful real release, revoke the personal PyPI API token.
