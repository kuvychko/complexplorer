## 0. Prerequisites (outside the repository)

> These cannot be done by a commit, and the workflow publishes nothing until they exist. Do them
> first, or the first tag fails at the upload step with an authentication error that looks like a
> workflow bug.

- [x] 0.1 Create a Trusted Publisher on **PyPI** for `complexplorer`: this repository, workflow
  `release.yml`, environment `pypi`.
- [x] 0.2 Create a Trusted Publisher on **TestPyPI** for the same, environment `testpypi`.
- [ ] 0.3 Create the `pypi` GitHub Environment with a required reviewer. Create `testpypi` without
  one — the rehearsal should not need an approval to be useful.
  **Not actually in place.** The `v3.1.0rc1` rehearsal published to PyPI without waiting: the
  environment exists but carries no protection rules (`protection_rules: []`, `updated_at` equal to
  `created_at`, so it was never edited after creation). Add the reviewer under Settings →
  Environments → pypi → Required reviewers, then verify the **rule**, not the environment:
  `GET /repos/kuvychko/complexplorer/environments` must show a `required_reviewers` entry for
  `pypi`. Checking that the environment exists proves nothing — GitHub silently creates a missing
  environment, without protections, the first time a workflow names one.
  Consider clearing `can_admins_bypass` too; it is `true` by default, which lets the approval be
  skipped by the person most likely to be in a hurry.
- [x] 0.4 Confirm no PyPI API token remains as a repository or organization secret; remove it if
  one is there. Keep the personal token until the first real release succeeds, then revoke it.

## 1. The release workflow

- [x] 1.1 Add `.github/workflows/release.yml`, triggered on `push` of tags matching `v*` and on
  `workflow_dispatch`. No branch trigger.
- [x] 1.2 Set `permissions: contents: read` at workflow level; grant `id-token: write` on the two
  publish jobs only, not on the build job.
- [x] 1.3 Add a `concurrency` group so two tags pushed together cannot race, and do **not** cancel
  in progress — an interrupted upload is worse than a queued one.

## 2. Tag/version gate

- [x] 2.1 As the first step of the build job, compare the tag (stripped of its leading `v`) against
  `complexplorer._version.__version__` read from the checkout, and fail on a mismatch with a message
  naming both.
- [x] 2.2 Read the version without installing the package — `_version.py` has no imports, so
  parsing it directly avoids needing a built environment before the gate runs.

## 3. The artifact gate

> These steps already exist in `ci.yml`'s `artifact` job. Duplicating them is deliberate — see
> `design.md`. Keep them textually close so a future extraction into a composite action is easy.

- [x] 3.1 `uv build`, once. Everything downstream consumes this `dist/`.
- [x] 3.2 `twine check dist/*`.
- [x] 3.3 `python scripts/check_distribution.py dist`.
- [x] 3.4 Install the wheel into a throwaway environment with neither the checkout nor the dev
  dependencies, and run `scripts/smoke_wheel.py` from outside the checkout. Needs
  `PYVISTA_OFF_SCREEN` and the headless display action, as in CI.
- [x] 3.5 Build from the sdist and import the result.
- [x] 3.6 Upload `dist/` as a workflow artifact; the publish jobs download it rather than
  rebuilding.

## 4. Publish

- [x] 4.1 A `testpypi` job: `needs: build`, environment `testpypi`, downloads the artifact, publishes
  with `pypa/gh-action-pypi-publish` pointed at the TestPyPI repository URL.
- [x] 4.2 A `pypi` job: `needs: testpypi`, environment `pypi` (the one with the required reviewer),
  same artifact, default index.
- [x] 4.3 Do **not** pass `skip-existing` on the PyPI job — a duplicate version must fail. It is
  acceptable on TestPyPI, where a rehearsal may legitimately repeat a version.
- [x] 4.4 Pin the publish action to a commit SHA rather than a tag: it handles the upload identity,
  so it is the one action here worth pinning immutably.

## 5. Docs

- [x] 5.1 Add the release procedure to `docs/development/contributing.md`: bump `_version.py`, land
  it, add the changelog entry, tag `vX.Y.Z`, push the tag, approve the `pypi` environment.
- [x] 5.2 Keep the existing manual gate commands, reframed as "how to reproduce the gate locally"
  rather than as the release path.
- [x] 5.3 State that the version bump, changelog and release notes stay manual, and why.
- [x] 5.4 Note that a bad release is recovered with a new version, never a retry, because PyPI
  versions are immutable.

## 6. Verification

- [x] 6.1 `openspec validate publish-on-tag` and `openspec validate --specs`.
- [x] 6.2 Actionlint (or equivalent) over the new workflow. (actionlint 1.7.12, clean over all
  four workflows.)
- [ ] 6.3 **Rehearse before 3.1.0, not during it.** Tag a prerelease — `v3.1.0rc1` against a tree
  whose `_version.py` says `3.1.0rc1` — and confirm: the gate passes, TestPyPI receives the upload,
  the `pypi` job waits for approval, and approving it publishes. A prerelease is the cheapest
  honest end-to-end test, and PyPI hides prereleases from the default install.
  Note that `v3.1.0rc1` also matches `docs.yml`'s `v3.*` trigger and `notebooks.yml`'s `v*`, so the
  rehearsal deploys the site from the release candidate too; either accept that or rehearse under a
  tag those two do not match.
  Found while rehearsing, on `chore/release-rehearsal`: a version bump alone turns the suite red,
  because `examples/gallery/index.json` stamps `complexplorer_version` and `test_gallery` asserts
  the committed manifest reproduces byte for byte. `CITATION.cff` is coupled the same way, through
  `test_release_metadata`. The runbook regenerated the gallery *before* bumping, which leaves both
  stale; the order is now reversed. Neither would have blocked the release itself — `release.yml`
  runs the artifact gate, not the suite — so CI was the only thing standing between a green tag and
  a release commit that fails its own tests.
  **Rehearsal result (`v3.1.0rc1`, Release run 1):** the tag gate, all twelve artifact-gate steps,
  the TestPyPI upload and the PyPI upload all passed, through Trusted Publishing with no token — and
  the PyPI job did **not** wait for an approval, because the environment had no reviewer (see 0.3).
  So this task is not yet satisfied: everything but the approval requirement is verified. Re-verify
  by dispatching the workflow against the existing `v3.1.0rc1` tag once the reviewer is configured;
  that run doubles as 6.5, since PyPI already holds the version.
- [ ] 6.4 Verify the tag gate fires: push a deliberately mismatched tag to a scratch branch and
  confirm the run fails before building. (The gate script itself was exercised locally against a
  matching tag, a mismatched tag and a branch ref; what remains is confirming it in a real run.)
- [ ] 6.5 Verify the duplicate-version guard by re-running the release for the already-published
  prerelease and confirming it fails rather than reporting success.
- [ ] 6.6 After the first successful real release, revoke the personal PyPI API token.
