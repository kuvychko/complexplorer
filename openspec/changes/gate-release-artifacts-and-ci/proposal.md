## Why

CI installs the source tree with `uv pip install -e ".[dev]"` and runs the tests. Nothing ever
checks the thing users actually receive: the wheel. A release could ship with `py.typed` missing,
a license file absent, the console entry point broken, tests or example images swept into the
distribution, or a dependency floor that does not resolve — and every test would still be green.

Three more gaps make the gate weaker than it looks:

- **The lint job is unpinned** (`ruff>=0.6`). Ruff 0.16 formats Markdown code blocks, which is
  already enough to fail `ruff format --check` on this repository the moment a runner picks up a
  newer release. A gate that breaks on its own tooling teaches people to ignore it.
- **Python 3.13 cannot fail the build** (`continue-on-error`), while the package claims to support
  it. It has in fact been passing on both Linux and Windows for some time.
- **The declared dependency floors are never exercised.** `numpy>=1.26`, `matplotlib>=3.8`,
  `scipy>=1.11`, `asteval>=1.0` and `pyvista>=0.47` are a promise no job tests.

Two `packaging` requirements also still describe a world that ended at 3.0: a CI configuration
"without PyVista" and PyVista as an optional 2.x extra.

## What Changes

**An artifact gate**
- Build the wheel and sdist, and validate their metadata with `twine check`.
- Install the **wheel** into a fresh environment — not the working tree — and smoke-test it:
  `import complexplorer` and its version, `complexplorer list`, one 2D file render, one real
  off-screen PyVista render, and one small STL export.
- Inspect the built distributions: `py.typed` present, both license files included, the expected
  subpackages shipped, and no tests, examples, gallery images or build debris.
- Inspect the wheel metadata: version, `Requires-Python`, dependencies, the SPDX license
  expression, the console entry point and the project URLs.
- Build a wheel **from the sdist** once, so the source distribution is proven to be buildable.

**A matrix that matches the claims**
- Pin ruff to a known minor in both CI and the `dev` extra, so the lint gate cannot break
  underneath the project.
- Remove `continue-on-error` from Python 3.13; a claimed version blocks the build.
- Add a macOS lane, which matters because VTK wheels are platform-specific.
- Add an Ubuntu/3.11 **minimum-dependency** lane that installs the declared floors.
- Add a scheduled run against the newest dependencies, so an upstream VTK or PyVista change
  surfaces before a release sprint rather than during one.
- Add a Python 3.14 lane that is allowed to fail, so the next version's readiness is visible
  without blocking. The classifier is added only once that lane is green.

**Coverage the repository already earns but does not check**
- Lint the executable examples (`examples/**/*.py`), which today are excluded wholesale — the
  gallery producer, the tour recipes and the interactive showcase included.
- Run the notebooks (`pytest --nbmake`) as a scheduled and pre-release gate rather than on every
  push, because they are PyVista-heavy and slow.
- Assert that a freshly generated `index.json` is byte-identical to the committed one, which is
  the manifest contract stated as a test rather than a habit.

**Decisions recorded**
- Measure the installed footprint and cold `import complexplorer` time, and record the
  mandatory-PyVista decision in the backend policy with those numbers, closing the question the
  closeout asks to settle once.

## Capabilities

### New Capabilities

_None._

### Modified Capabilities

- `packaging`:
  - **Removed**: the requirement for a CI configuration without PyVista, and the 2.x
    optional-extra dependency strategy. Both describe a world that ended at 3.0.
  - **Added**: built distributions are verified before release (contents, metadata, and a
    fresh-environment install of the wheel that renders and exports).
  - **Added**: CI covers every claimed Python version as a blocking lane, on Linux, Windows and
    macOS, plus the declared minimum dependency versions.
  - **Modified**: linting covers the executable examples, and the tool version is pinned.
- `examples`: the notebook execution harness is required before a release (scheduled and
  pre-release), while staying out of the per-push suite.

## Impact

- **CI:** `.github/workflows/ci.yml` gains artifact, macOS, minimum-dependency, 3.14 and
  scheduled lanes; a new workflow (or job) runs the notebooks.
- **Packaging:** `pyproject.toml` — pinned ruff, ruff's `extend-exclude` narrowed so example
  scripts are linted.
- **Code:** whatever lint findings the newly covered `examples/**/*.py` surface.
- **Tests:** a committed-vs-fresh `index.json` comparison; a small script that inspects a built
  distribution, reusable locally and in CI.
- **Docs:** `docs/development/backend-policy.md` records the mandatory-PyVista decision with
  measured numbers.
- **Not here:** trusted publishing and TestPyPI dry runs (3.1+ backlog), and the release runbook
  itself (`prepare-3-0-release-notes`).
