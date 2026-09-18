## Context

`.github/workflows/ci.yml` has two jobs: a lint job that installs `ruff>=0.6` and checks
`complexplorer/` and `tests/`, and a test matrix over {ubuntu, windows} × {3.11, 3.12, 3.13} that
installs the source editable (`uv pip install -e ".[dev]"`), sets `PYVISTA_OFF_SCREEN`, brings up a
headless display on Linux, and runs pytest. Python 3.13 carries `continue-on-error` from a time
when VTK wheels lagged; both 3.13 lanes pass today.

`harden-3-0-release` ran a one-time wheel smoke test by hand and recorded it; nothing enforces it.
The `packaging` capability still carries two requirements written for the 2.x line: CI in "both a
base configuration (without PyVista) and a configuration with PyVista", and PyVista offered through
`[pyvista]`/`[3d]` extras. Since 3.0 PyVista is a core dependency and those extras are empty
aliases.

Facts that shape the work:

- `uv` supports `--resolution lowest-direct`, so the declared floors can be installed without
  hand-maintaining a constraints file.
- Ruff is pinned nowhere; 0.16 formats Markdown code blocks, which already required a fix in this
  repository.
- `examples/` is excluded wholesale by `tool.ruff.extend-exclude`.
- The notebooks take about four minutes locally and drive PyVista.
- `cp.gallery` is deterministic and PyVista-free, so a byte comparison of `index.json` is cheap and
  safe to run everywhere.

## Goals / Non-Goals

**Goals:**
- What users install is tested before a release, not just what contributors check out.
- Every claimed Python version and dependency floor is exercised by a lane that can fail the build.
- The lint gate cannot break because of an unpinned tool.
- The notebooks are verified before a release without slowing every push.
- The mandatory-PyVista decision is recorded with measurements.

**Non-Goals:**
- Publishing: trusted publishing, TestPyPI dry runs and the release workflow stay in the 3.1+
  backlog.
- The release runbook narrative (`prepare-3-0-release-notes`).
- Docs-site building and deployment (`publish-rev3-docs-site`).
- Making the notebooks fast enough for per-push CI.

## Decisions

### D1. The artifact gate is a script the developer can run, wrapped by a job

`scripts/check_distribution.py` takes a built wheel and sdist and asserts the contents and metadata
rules. CI calls it; a developer can call it before tagging. Putting the logic in a script rather
than in YAML keeps it reviewable, testable and runnable locally — the same reason the gallery
producer is a script.

The job then does what the script cannot: create a throwaway environment, install the **wheel**
from `dist/`, and run the smoke commands against the installed package with the repository checkout
off `sys.path` (run from a different working directory) so nothing passes by accident.

- *Alternative:* assert everything inside pytest. Rejected: the test suite runs against the source
  tree, and the point is to test the artifact.

### D2. Smoke tests exercise the paths that actually break

Import and version; `complexplorer list` (the console entry point); a 2D file render (matplotlib,
Agg); a real off-screen PyVista render (VTK, the dependency most likely to be broken by
packaging); and a small STL export (mesh repair plus file output). These are the five things a
user does in the first five minutes.

### D3. Ruff is pinned to a minor range in both places

`pyproject.toml`'s dev extra and the CI install use the same `ruff>=0.16.7,<0.17`. A range rather
than an exact pin keeps patch fixes flowing; the minor bound is what stops a formatting change from
breaking the gate unannounced. Upgrading becomes a deliberate commit.

### D4. Lint the example scripts, not the notebooks

`extend-exclude` drops `examples` and gains `examples/**/*.ipynb`. The scripts — the gallery
producer, the tour recipes, the showcases — are executable code the project ships and should hold to
the same standard. Notebook cells legitimately break lint rules (imports after code, unused display
expressions) and linting them fights the format instead of helping it.

The one exception the scripts need is `E402`: `examples/showcase.py` and `examples/tour.py` must
call `matplotlib.use("Agg")` before importing `pyplot`, so their imports cannot all sit at the top.
That is recorded as a per-file ignore with the reason.

### D5. The matrix grows by lane, not by product

Adding macOS to the existing `os` list would multiply into three more lanes. macOS runners are
slower and VTK's behaviour there is the thing under test, so macOS is added as one explicit lane on
the current Python (3.12), with the full {ubuntu, windows} × {3.11, 3.12, 3.13} grid unchanged and
3.13 now blocking. Python 3.14 joins as its own allowed-to-fail lane. Total lanes: 6 + 1 + 1 + 1
minimum-dependency + 1 artifact.

### D6. The minimum-dependency lane installs the floors, and is allowed to be strict

`uv pip install --resolution lowest-direct -e ".[dev]"` resolves every declared direct dependency to
its lowest allowed version. It runs on Ubuntu/3.11 (the oldest supported interpreter), because a
floor that only works on the newest runtime is not a floor. If an old release genuinely cannot work
with 3.11, the fix is to raise the floor in `pyproject.toml` — which is the point of the lane.

### D7. Notebooks run on a schedule and before a release, not on every push

A separate workflow (`notebooks.yml`) runs `pytest --nbmake examples/notebooks/` on a weekly
schedule, on `workflow_dispatch`, and on `v*` tags. It keeps the per-push suite fast while making
the tutorials a release gate. The `examples` capability is amended to say exactly that, since it
currently forbids requiring them in CI.

### D8. The gallery contract becomes a test, not a habit

`tests/unit/test_gallery.py` gains a check that regenerating the catalog into a temporary directory
reproduces the committed `index.json` byte-for-byte. That is stronger than the existing
"two runs agree" test: it also catches a manifest committed from a modified tree. Images are
excluded, as the capability already says.

### D9. The scheduled newest-dependency run reuses the test job

The weekly run installs without constraints (the same `uv pip install -e ".[dev]"`, which already
resolves to the newest compatible releases) on one lane, so an upstream VTK/PyVista change appears
as a red scheduled run rather than a surprise during release week.

### D10. The PyVista decision is recorded with numbers, in the backend policy

The artifact job prints the installed footprint and a cold `import complexplorer` time; those
numbers go into `docs/development/backend-policy.md` alongside the decision to keep PyVista
required. The residual optional-backend language in that document goes at the same time.

## Risks / Trade-offs

- **The minimum-dependency lane may fail immediately**, since those floors have never been tested.
  → That is information, not breakage: either a floor rises in `pyproject.toml` or a genuine
  incompatibility is documented. It is a blocking lane precisely so the answer gets written down.
- **A real off-screen render inside the artifact job can be flaky on a headless runner.** → It
  already runs in the test matrix on the same runners with the same headless display action, so the
  risk is known rather than new.
- **macOS runner minutes.** → One lane on one Python version.
- **The 3.14 lane will fail until VTK ships wheels.** → It is allowed to fail and carries no
  classifier until it is green.
- **Pinning ruff freezes lint findings until someone upgrades.** → Accepted; the upgrade is then a
  deliberate commit with its own diff, which is the behaviour that was missing.

## Migration Plan

No runtime behaviour changes. The gate can only fail the build; if a lane proves unusable it is
removed in its own commit with the reason. Rollback is a revert of the workflow and `pyproject.toml`
changes.

## Open Questions

_None._ Whether Python 3.14 earns its classifier is answered by the allowed-to-fail lane itself.
