## 1. Pin the tooling and widen lint coverage

- [ ] 1.1 Pin ruff to `>=0.16.7,<0.17` in the `dev` extra and in the CI lint job, so the gate
  cannot break on a new formatter release.
- [ ] 1.2 Narrow `tool.ruff.extend-exclude` so `examples/**/*.py` is linted while notebooks stay
  exempt, and add the per-file `E402` exception for the scripts that must call
  `matplotlib.use("Agg")` before importing `pyplot`.
- [ ] 1.3 Fix the findings this surfaces in `examples/`, and extend the CI lint job to cover
  `examples/`.

## 2. The distribution inspector

- [ ] 2.1 Add `scripts/check_distribution.py`, which takes the built wheel and sdist and asserts:
  - the wheel contains `complexplorer/py.typed`, both license files and the expected subpackages
  - neither distribution carries tests, examples, gallery images or build debris
  - the metadata declares the version, `Requires-Python`, runtime dependencies, SPDX license,
    project URLs and the console entry point
- [ ] 2.2 Make it runnable locally (`python scripts/check_distribution.py dist/`) and give it a
  clear failure report, since it is the tool someone runs before tagging.

## 3. The artifact gate in CI

- [ ] 3.1 Add an `artifact` job: `uv build`, then `twine check dist/*`, then
  `scripts/check_distribution.py`.
- [ ] 3.2 In the same job, create a throwaway environment, install the **wheel** from `dist/`, and
  run the smoke set from a directory outside the checkout:
  - `import complexplorer; print(complexplorer.__version__)`
  - `complexplorer list`
  - a 2D render written to a file
  - a real off-screen PyVista render written to a file (headless display on Linux)
  - a small STL export
- [ ] 3.3 Build a wheel from the sdist in a clean environment and import it.
- [ ] 3.4 Print the installed footprint and a cold `import complexplorer` time for task 6.1.

## 4. Matrix

- [ ] 4.1 Remove `continue-on-error` from Python 3.13; both 3.13 lanes already pass.
- [ ] 4.2 Add a macOS lane on the current Python, as one explicit lane rather than a third
  dimension of the grid.
- [ ] 4.3 Add an Ubuntu/3.11 minimum-dependency lane using
  `uv pip install --resolution lowest-direct -e ".[dev]"`. If a declared floor cannot work, raise it
  in `pyproject.toml` and say why in the commit.
- [ ] 4.4 Add a Python 3.14 lane that is allowed to fail, and leave the classifier off until it
  passes.
- [ ] 4.5 Add a weekly scheduled run against the newest compatible dependencies.

## 5. Notebooks and the gallery contract

- [ ] 5.1 Add a notebooks workflow running `pytest --nbmake examples/notebooks/` on a weekly
  schedule, on `workflow_dispatch` and on `v*` tags, with a headless display and a timeout.
- [ ] 5.2 Add a test asserting that a freshly generated `index.json` is byte-identical to the
  committed one (images excluded, as the capability says).

## 6. Decisions and documentation

- [ ] 6.1 Record the mandatory-PyVista decision in `docs/development/backend-policy.md` with the
  measured footprint and cold import time, and remove the residual optional-backend language there.
- [ ] 6.2 Note in `CONTRIBUTING.md`'s absence — or in the closeout tracker until that file exists —
  how to run the artifact gate locally before tagging.

## 7. Verification

- [ ] 7.1 Run the gate locally: `pytest`, `ruff check`/`format` over package, tests and examples,
  `openspec validate --specs`, `openspec validate gate-release-artifacts-and-ci`, `uv build`,
  `twine check`, and the distribution inspector.
- [ ] 7.2 Push and confirm on CI: every lane green, the artifact job green, 3.13 blocking, the
  minimum-dependency lane resolved and passing, and the 3.14 lane reporting without blocking.
- [ ] 7.3 Flip C4's status in `openspec/ROADMAP.md` and record the outcome in
  `openspec/REV3_CLOSEOUT.md`.
