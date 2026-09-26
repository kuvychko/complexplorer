# Contributing to complexplorer

Thanks for taking an interest. This file covers how to set the project up, what the checks are,
how changes are specified, and how a release is cut.

The user-facing version of the first half lives on the documentation site under
[Contributing](https://kuvychko.github.io/complexplorer/development/contributing/); this file is
the one that also carries the release runbook.

## Setting up

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
```

`[dev]` brings the test and lint tooling and the `[examples]` extra. Add `[docs]` when working on
the site.

## The checks

```bash
pytest tests/ -q
ruff check complexplorer/ tests/ examples/ scripts/ docs/hooks/
ruff format --check complexplorer/ tests/ examples/ scripts/ docs/hooks/
openspec validate --specs
mkdocs build --strict
```

Ruff is pinned to a minor version on purpose: a formatter release once failed this gate with no
change to the repository, and a gate that breaks on its own tooling teaches people to ignore it.

`ruff format --check` reports failure in its output, but its exit status can be masked by a shell
pipeline — read the output, not just the status.

## Spec-driven changes

Behaviour is specified before it is written. The baseline specs live in `openspec/specs/`,
organised into capabilities. A change starts as a proposal in `openspec/changes/<name>/` with
`proposal.md`, `design.md`, `tasks.md` and delta specs:

```bash
openspec new change my-change          # scaffold
openspec validate my-change            # check it
openspec archive my-change             # apply the deltas to the baseline specs
```

For anything non-trivial, write the change first. The delta specs are what get merged into the
baseline on archive, so they are the lasting artifact.

## Notebooks

```bash
pytest --nbmake examples/notebooks/
```

Not part of the default suite — they drive PyVista and take minutes. They run on a schedule, on
demand, and on every release tag.

## Regenerating the gallery

```bash
python examples/showcase.py
```

Every image in the repository and on the site comes from that command, via the recipes in
`examples/gallery/showcase.json`. Assets are promoted by regenerating, never by copying a file in
by hand.

Manifests (`index.json`, `card.json`) are byte-identical for a given library version on any
platform, so a regeneration that changes them is telling you something real. Portrait PNGs are
reproducible only best-effort and differ across machines.

"For a given library version" is literal: `index.json` and `showcase.json` record
`complexplorer_version`, so bumping the version invalidates the committed manifest until the gallery
is regenerated. That is why the release runbook below bumps before it regenerates.

## The release artifact gate

The suite runs against the source tree, so it cannot see what a user installs; these checks look at
the artifact itself. CI runs them on every push, and the release workflow runs them again against
the exact files it then publishes — nothing is rebuilt between the gate and the upload. To reproduce
the gate locally:

```bash
# 1. Build, and check the metadata the way PyPI will.
uv build
uv run --no-project --with twine twine check dist/*

# 2. Inspect the distributions: contents, licences, py.typed, metadata, entry point.
python scripts/check_distribution.py dist

# 3. Install the WHEEL into a throwaway environment - not the checkout.
uv venv /tmp/smoke --python 3.12
uv pip install --python /tmp/smoke/bin/python dist/*.whl

# 4. Smoke it from somewhere else: import, CLI, 2D render, off-screen 3D render, STL export.
cd /tmp && /tmp/smoke/bin/python "$OLDPWD/scripts/smoke_wheel.py"
```

On Windows use `/tmp/smoke/Scripts/python.exe`. The smoke script refuses to run against a source
tree, because that is the mistake it exists to catch.

This gate is not ceremonial. On its first run it found two crashes the test suite could not see,
both on Windows consoles using a legacy code page: `complexplorer list` and STL export each ended
in `UnicodeEncodeError`. The tests cover the library; the gate covers the product.

## Releasing

A release is a **tag**. Pushing `vX.Y.Z` runs `.github/workflows/release.yml`, which checks the tag
against the packaged version, runs the artifact gate against the distributions it will publish,
uploads them to TestPyPI, and then waits for an approval before uploading to PyPI. Merging to `main`
publishes nothing, on purpose: `main` carries documentation, chores and specification archives, and
a condition that suppressed a publish for those would have to be right every time against a mistake
PyPI can yank but never undo.

1. **Join the histories.** On the release branch, `git merge -s ours origin/main` records main as
   merged while keeping this tree, so the PR into `main` is a fast-forward and the older history
   stays reachable.
2. **Bump the version, first.** `__version__` in `complexplorer/_version.py` is the single source
   of truth, and the release workflow fails before building if the tag disagrees with it.
   `CITATION.cff` must state the same version — a test asserts the two agree.
3. **Regenerate the gallery** — `python examples/showcase.py`, with the diff reviewed. This follows
   the bump rather than preceding it: `examples/gallery/index.json` and `showcase.json` stamp
   `complexplorer_version`, and a test asserts the committed `index.json` reproduces byte for byte,
   so regenerating before the bump leaves a stale manifest and a release commit whose suite is red.
4. **Set the changelog date.** `## [3.0.0] - Unreleased` becomes the release date, and the compare
   links at the bottom are updated.
5. **Run the full gate on a clean checkout:** the checks above, plus
   `pytest --nbmake examples/notebooks/`, and the artifact gate.
6. **Tag** `vX.Y.Z` and push it. This triggers the release workflow, the documentation deploy and
   the notebook run; none of them can be triggered by a branch push.
7. **Approve the `pypi` environment** on the release run. TestPyPI publishes without an approval;
   PyPI waits for a person, because a tag at the wrong commit — or ahead of a fix still in flight —
   produces a perfectly valid artifact that no automated check can distinguish from a correct
   release.
8. **Cut the GitHub release**, with notes drawn from the changelog and a link to the migration
   guide.
9. **Verify in a clean browser session:** the PyPI page (hero image, badges, links, code fences,
   licence, version) and the documentation site (version banner showing the release rather than
   "development build").

Both uploads authenticate through PyPI's Trusted Publishing: the run requests a short-lived identity
that the index verifies against this repository, the workflow filename and the environment name.
There is no API token in repository secrets or in anyone's password manager, and the configuration
lives on PyPI and in the GitHub environment settings rather than in this repository.

The version bump, the changelog entry and the release notes stay manual. Inferring a bump from
commit messages is a guess about semantics this project makes deliberately — 3.0 removed the
matplotlib 3D backend, and no commit-message convention would have decided that was a major. The
tag gate verifies the bump rather than writing it.

A bad release is recovered with a **new version**, never a retry. A published version is immutable —
PyPI allows it to be yanked, never replaced — so the workflow fails on a version already on the
index rather than skipping it: a green run that uploaded nothing would answer "did my fix ship?"
incorrectly.

## Reporting problems

Issue templates are in `.github/ISSUE_TEMPLATE/`. For anything visual, the rendering template asks
for your OS, Python version, PyVista and VTK versions, and GPU or display situation — rendering
bugs are not reproducible without them.
