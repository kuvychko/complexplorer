# Contributing and release checks

## Setting up

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
```

`[dev]` pulls in the test and lint tooling and the `[examples]` extra. Add `[docs]` if you are
working on this site.

## The checks

```bash
pytest tests/ -q
ruff check complexplorer/ tests/ examples/ scripts/
ruff format --check complexplorer/ tests/ examples/ scripts/
openspec validate --specs
mkdocs build --strict
```

Ruff is pinned to a minor version deliberately: a formatter release once failed this gate with no
change to the repository, and a gate that breaks on its own tooling teaches people to ignore it.

Note that `ruff format --check` reports failure in its output but its exit code can be masked by a
shell pipeline — read the output, not just the status.

## Spec-driven changes

Behaviour is specified before it is written. Specs live in `openspec/specs/`, organised into
capabilities; a change starts as a proposal in `openspec/changes/<name>/` with `proposal.md`,
`design.md`, `tasks.md` and delta specs, and is archived into the baseline when it lands.

For anything non-trivial, write the change first. `openspec validate <change>` checks it, and
`openspec archive <change>` applies the deltas to the baseline specs.

## The notebooks

```bash
pytest --nbmake examples/notebooks/
```

Not part of the default suite — the notebooks drive PyVista and take minutes. They run on a
schedule, on demand, and on every release tag.

## Regenerating the gallery

```bash
python examples/showcase.py
```

Every image on this site and in the repository comes from that command, via the recipes in
`examples/gallery/showcase.json`. Assets are promoted by regenerating, never by copying a file in
by hand — that is what keeps "every committed asset reproduces from a documented command" true.

Manifests (`index.json`, `card.json`) are byte-stable for a given library version on any platform,
so a regeneration that changes them is telling you something real. Portrait PNGs are reproducible
only best-effort and will differ across machines.

"For a given library version" is literal: `index.json` and `showcase.json` record
`complexplorer_version`, so a version bump invalidates the committed manifest until the gallery is
regenerated. Bump first, then regenerate — the other order leaves a release commit whose suite is
red.

## Releasing

A release is a **tag**. Pushing `vX.Y.Z` runs `.github/workflows/release.yml`, which checks the tag
against the packaged version, runs the artifact gate below, uploads to TestPyPI, and then waits for
an approval before uploading to PyPI. Merging to `main` publishes nothing: `main` carries
documentation, chores and specification archives, and a condition that suppressed a publish for
those would have to be right every time against a mistake PyPI can yank but never undo.

1. Bump `__version__` in `complexplorer/_version.py`, and `CITATION.cff` to match. The workflow
   compares the tag against `__version__` and fails before building if they disagree.
2. Regenerate the gallery, after the bump, for the reason given above.
3. Add the changelog entry and date it.
4. Land all of that, then tag the landed commit and push the tag:
   `git tag vX.Y.Z && git push origin vX.Y.Z`.
5. Approve the `pypi` environment when the run asks. TestPyPI publishes without an approval; PyPI
   waits for a person.

Neither guard replaces the other. TestPyPI surfaces configuration faults — a publisher entry that
does not match, metadata an index rejects that `twine check` accepts — against an index where a
version number can be spent harmlessly. The approval surfaces intent faults, such as a tag at the
wrong commit or ahead of a pending fix, which produce a perfectly valid artifact that no automated
check can distinguish from a correct release.

Both uploads authenticate through PyPI's Trusted Publishing, so there is no API token in repository
secrets or in anyone's password manager.

The version bump, the changelog entry and the GitHub Release notes stay manual. Inferring a bump
from commit messages is a guess about semantics this project makes deliberately — 3.0 removed the
matplotlib 3D backend, and no commit-message convention would have decided that was a major. The
tag gate verifies the bump rather than writing it.

A bad release is recovered with a **new version**, never a retry. A published version is immutable,
so the workflow fails on a version already on the index rather than skipping it: a green run that
uploaded nothing would answer "did my fix ship?" incorrectly.

The wider runbook — joining the histories, the notebook run, the gallery regeneration,
`CITATION.cff`, the GitHub release — is in
[`CONTRIBUTING.md`](https://github.com/kuvychko/complexplorer/blob/main/CONTRIBUTING.md) in the
repository.

## The release artifact gate

The test suite runs against the source tree, so it cannot see what a user installs; these checks
look at the artifact itself. CI runs them on every push, and the release workflow runs them again
against the exact files it then publishes — nothing is rebuilt between the gate and the upload. To
reproduce the gate locally:

```bash
# Build, and check the metadata the way PyPI will.
uv build
uv run --no-project --with twine twine check dist/*

# Inspect the distributions: contents, licences, py.typed, metadata, entry point.
python scripts/check_distribution.py dist

# Install the WHEEL into a throwaway environment - not the checkout.
uv venv /tmp/smoke --python 3.12
uv pip install --python /tmp/smoke/bin/python dist/*.whl

# Smoke it from somewhere else: import, CLI, 2D render, off-screen 3D render, STL export.
cd /tmp && /tmp/smoke/bin/python "$OLDPWD/scripts/smoke_wheel.py"
```

On Windows use `/tmp/smoke/Scripts/python.exe`. The smoke script refuses to run against a source
tree, because that is the mistake it exists to catch.

This gate is not ceremonial. On its first run it found two crashes that the 695-test suite could
not see, both on Windows consoles using a legacy code page: `complexplorer list` and STL export
each ended in `UnicodeEncodeError`. The tests cover the library; the gate covers the product.

## CI

Every push runs lint, the suite on Linux and Windows across the supported Python versions plus
macOS, a lane that installs the declared minimum dependency versions, a non-blocking lane on the
next Python, the artifact gate, and a strict documentation build. A weekly run repeats the suite
against the newest dependencies so an upstream change surfaces away from a release.
