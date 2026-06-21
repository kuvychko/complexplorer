# Tasks — reconcile-versioning-and-license

## 1. Canonical version
- [x] 1.1 Canonical version is **`2.0.0`** — confirmed as the latest PyPI release (the
      `v2.0.0` tag matches). The `1.0.0`/`1.0.1` values in the repo are stale.

## 2. Single source of truth for version
- [x] 2.1 Set `complexplorer/_version.py` `__version__` to `2.0.0`.
- [x] 2.2 In `pyproject.toml`, remove the hardcoded `version = "1.0.1"` and add
      `dynamic = ["version"]` to `[project]`.
- [x] 2.3 Add `[tool.setuptools.dynamic]` with
      `version = {attr = "complexplorer._version.__version__"}`.
- [x] 2.4 Build (`uv build` or `python -m build`) and confirm the wheel/sdist metadata
      version matches `_version.py`. → built `complexplorer-2.0.0.tar.gz`.
- [x] 2.5 Confirm `python -c "import complexplorer; print(complexplorer.__version__)"`
      reports the same value. → reports `2.0.0`.

## 3. Resolve the license contradiction
- [x] 3.1 Decide MIT vs BSD (recommend MIT, matching the `LICENSE` file). → **MIT**.
- [x] 3.2 Update the `pyproject.toml` classifier to match the decision
      (`License :: OSI Approved :: MIT License`).
- [x] 3.3 Document the dual-licensing arrangement: code under the primary license;
      generated artistic/STL artifacts under `LICENSE.art` (CC-BY-NC 4.0). → already
      documented in `README.md` §License (MIT code / CC BY-NC 4.0 outputs).
- [x] 3.4 Verify `LICENSE`, `LICENSE.art`, `pyproject.toml`, and `README.md` agree.

## 4. Close out
- [x] 4.1 Run the test suite (`pytest tests/`) to confirm nothing depends on the old
      version string. → 339 passed, 2 skipped.
- [x] 4.2 Update `openspec/ROADMAP.md`: flip this change's STATUS to `in-progress`, then
      `archived` when synced.
