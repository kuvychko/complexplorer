# Design

## Why a tag and not a merge

The question that prompted this change was whether to publish on merge to `main`. The repository
answers it itself.

`main` receives commits that are not releases, routinely. From the last fifteen:

```
f953167  Merge pull request #10 from kuvychko/docs/openspec-baseline
cfb5e1c  docs: date the 3.0.0 release
69e1fc5  chore: join the 2.0 history into the rev3 line
1f89a6f  docs(openspec): archive the last two changes and close out rev3
```

Publishing on merge means every one of those either carries a version bump or is filtered out by a
skip-if-version-unchanged condition. That condition is the whole design, it has to be right every
time, and when it is wrong the failure mode is an irreversible upload — PyPI allows a version to be
yanked but never replaced. A tag is an explicit, cheap, reviewable statement that *this commit is a
release*, and it fails safe: forgetting to tag publishes nothing.

There is also a consistency argument. `docs.yml` already restricts deployment to `v3.*` tags, with
a comment explaining that publication should be deliberately hard to trigger. Documentation can be
redeployed freely; a PyPI upload cannot. Whatever threshold is right for the docs is a floor for
this.

## Trusted Publishing is the point, the trigger is secondary

The stated pain is handling an API key. Trusted Publishing removes the key rather than relocating
it: the workflow requests a short-lived OIDC token, and PyPI validates it against the repository,
the workflow filename and the environment name configured on the project. Nothing is stored in
repository secrets, nothing sits in a password manager, and a token cannot leak from a laptop
because none exists.

This requires `permissions: id-token: write` on the publishing job. That permission is scoped to
the job, not the workflow, so the build job runs without it.

The configuration lives on PyPI and in GitHub's environment settings, not in the repository. That
is worth noting explicitly because it means this change is not self-contained: the workflow file
alone will not publish anything until the Trusted Publisher entries exist. The tasks call that out
as a prerequisite rather than a follow-up.

## Publish the artifact that passed

The release gate in `docs/development/contributing.md` is not ceremonial — it caught two Windows
console crashes that the full test suite could not see. CI's `artifact` job already runs it on
every push.

The gap is that neither is connected to publishing. Today what reaches PyPI is whatever `uv build`
produced on the machine that ran `twine`, which is *probably* identical to what CI validated, but
nothing enforces it.

So the release workflow builds once and passes the `dist/` directory to the publish jobs as a
workflow artifact. The build job is the gate; the publish jobs upload its output. Rebuilding in the
publish job would reintroduce exactly the gap being closed, even though the rebuild would almost
certainly be identical.

Reusing CI's existing `artifact` job across workflows was considered and rejected: fetching another
workflow's artifact requires resolving its run for the tagged commit, which is more moving parts
than rerunning the gate, and it couples the release to CI's scheduling. The gate steps are
duplicated between `ci.yml` and `release.yml` deliberately — a composite action could share them
later, but that indirection is not worth adding while there are two call sites.

## The tag must agree with the version

`pyproject.toml` derives its version from `complexplorer._version.__version__`, and the packaging
capability already requires that single source of truth. A tag is a second, independent assertion
of the same fact, written by hand, at the moment attention is lowest.

Tagging `v3.1.0` on a tree that still says `3.0.0` would publish a package named 3.0.0 — which
PyPI rejects, since 3.0.0 exists, so this particular mistake fails safely today. The dangerous
version is the reverse or the near-miss: `v3.1.0` against a tree already bumped to `3.1.1`, which
uploads successfully under the wrong name and cannot be undone.

Comparing the two before building costs one step and removes the class.

## Why TestPyPI first, and why an approval gate as well

They catch different failures.

TestPyPI catches **configuration** failure — a Trusted Publisher entry that does not match, a
malformed metadata field that `twine check` passes but the index rejects. This is most likely on
the very first run, which is precisely why the change should land and be rehearsed before 3.1.0
rather than during it.

The approval gate catches **intent** failure — a tag pushed early, pointing at the wrong commit, or
pushed while a fix is still in flight. No automated check can distinguish that from a correct
release, because the artifact is perfectly valid. A human looking at the tag and the run can.

TestPyPI accepts re-uploads under a version that already exists far more readily than PyPI does, so
a rehearsal does not burn the real version number.

## Failing on an already-published version

`twine upload --skip-existing` and the equivalent publish-action flag make a re-run succeed without
uploading. That is the wrong default here. A re-run happens when something went wrong, and the
question it needs answered is "did my fix ship?" — a green run that silently uploaded nothing
answers that incorrectly.

Failing loudly means a partially completed release cannot be mistaken for a complete one. The
correct recovery from a bad publish is a new version, not a retry, because PyPI will not let a
version be replaced.

## What is deliberately left manual

The version bump, the changelog entry, and the GitHub Release notes. Automating the bump means
inferring intent from commit messages, which is a guess about semantics that this project makes
deliberately — 3.0 removed the matplotlib 3D backend, and no commit-message convention would have
decided that was a major. The bump stays a commit a person writes; the tag gate verifies it rather
than producing it.
