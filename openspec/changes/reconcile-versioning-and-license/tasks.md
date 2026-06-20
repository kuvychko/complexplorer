# Tasks — reconcile-versioning-and-license

## 1. Canonical version
- [x] 1.1 Canonical version is **`2.0.0`** — confirmed as the latest PyPI release (the
      `v2.0.0` tag matches). The `1.0.0`/`1.0.1` values in the repo are stale.

## 2. Single source of truth for version
- [ ] 2.1 Set `complexplorer/_version.py` `__version__` to `2.0.0`.
- [ ] 2.2 In `pyproject.toml`, remove the hardcoded `version = "1.0.1"` and add
      `dynamic = ["version"]` to `[project]`.
- [ ] 2.3 Add `[tool.setuptools.dynamic]` with
      `version = {attr = "complexplorer._version.__version__"}`.
- [ ] 2.4 Build (`uv build` or `python -m build`) and confirm the wheel/sdist metadata
      version matches `_version.py`.
- [ ] 2.5 Confirm `python -c "import complexplorer; print(complexplorer.__version__)"`
      reports the same value.

## 3. Resolve the license contradiction
- [ ] 3.1 Decide MIT vs BSD (recommend MIT, matching the `LICENSE` file).
- [ ] 3.2 Update the `pyproject.toml` classifier to match the decision
      (`License :: OSI Approved :: MIT License`).
- [ ] 3.3 Document the dual-licensing arrangement: code under the primary license;
      generated artistic/STL artifacts under `LICENSE.art` (CC-BY-NC 4.0). Add a short
      "License" note to `README.md` if not already clear.
- [ ] 3.4 Verify `LICENSE`, `LICENSE.art`, `pyproject.toml`, and `README.md` agree.

## 4. Close out
- [ ] 4.1 Run the test suite (`pytest tests/`) to confirm nothing depends on the old
      version string.
- [ ] 4.2 Update `openspec/ROADMAP.md`: flip this change's STATUS to `in-progress`, then
      `archived` when synced.
