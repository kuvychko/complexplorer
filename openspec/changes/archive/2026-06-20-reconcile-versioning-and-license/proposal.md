# Reconcile versioning and license

## Why

The repository carries three different "current versions" and a license contradiction.
This is repo hygiene that must be fixed before any roadmap framing toward 3.0 is credible
(see `openspec/ROADMAP.md`, Phase 0).

Observed state:

| Source | Value |
|---|---|
| Latest git tag | `v2.0.0` |
| `complexplorer/_version.py` (`__version__`, the runtime version) | `1.0.0` |
| `pyproject.toml` `[project].version` (the build/dist version) | `1.0.1` |
| `LICENSE` file | MIT |
| `pyproject.toml` classifier | `License :: OSI Approved :: BSD License` |
| `LICENSE.art` | CC-BY-NC 4.0 (covers artistic/STL outputs) |

`pyproject.toml` does not read the version from `_version.py` — it hardcodes a separate,
divergent value, so the two will keep drifting.

## What changes

This is a **metadata / packaging** change only. It does not alter any behavioral contract,
so there is **no delta under `specs/`** — the work lives entirely in `tasks.md`.

1. **Single source of truth for version.** Make `pyproject.toml` derive the version
   dynamically from `complexplorer/_version.py` (`dynamic = ["version"]` +
   `[tool.setuptools.dynamic]`), and set `_version.py` to the true current version.
2. **Canonical version is `2.0.0`.** The latest release published to PyPI is `2.0.0`
   (and the `v2.0.0` tag matches it). Reconcile all in-repo sources to `2.0.0`; the
   `1.0.0`/`1.0.1` values in `_version.py`/`pyproject.toml` are stale and wrong.
3. **Resolve the license contradiction.** Decide MIT vs BSD (the `LICENSE` file is MIT;
   recommend standardizing on **MIT** and correcting the classifier), and document the
   dual-licensing arrangement where `LICENSE.art` (CC-BY-NC 4.0) governs generated
   artistic/STL artifacts.

## Non-goals

- Backend policy / PyVista-first declaration — that is a separate Phase 0 change
  (`establish-backend-and-release-policy`).
- Tooling (ruff, CI matrix) — separate change.
- Any code behavior change.

## Impact

- Affected files: `pyproject.toml`, `complexplorer/_version.py`, `LICENSE`/`LICENSE.art`
  references, `README.md` (license badge/section if present).
- Affected specs: none (metadata only).
- Risk: low. Verify `import complexplorer; complexplorer.__version__` and a build
  (`python -m build` / `uv build`) both report the reconciled version.
