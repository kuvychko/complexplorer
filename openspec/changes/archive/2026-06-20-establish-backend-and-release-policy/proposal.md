# Establish backend and release policy

## Why

Complexplorer's growth path is fundamentally 3D and mesh-oriented (Riemann relief,
Riemann surfaces, STL export), and matplotlib's 3D rendering is low quality and a
maintenance drag. The roadmap (`openspec/ROADMAP.md`) commits to a **sharp capability
line — matplotlib for 2D, PyVista for 3D — with no backend parity**, and to **removing
the matplotlib 3D paths entirely at 3.0** when PyVista becomes a required dependency.

That removal is breaking, so it cannot land in a 2.x minor. This change does the
**non-breaking groundwork** in 2.1: it declares the policy, signals the deprecation to
users in code and docs, and adds the dependency ergonomics — so 3.0 is a clean,
well-telegraphed switch rather than a surprise.

## What changes

Non-breaking, in 2.1. Nothing is removed yet.

1. **Backend policy document.** Add `docs/development/backend-policy.md` stating the
   2D=matplotlib / 3D=PyVista division, why it exists, and the 3.0 migration plan.
2. **Deprecate the matplotlib 3D paths.** `plot_landscape`, `pair_plot_landscape`, and
   the 3D `riemann()` surface SHALL emit a `DeprecationWarning` pointing to their PyVista
   equivalents (`plot_landscape_pv`, `pair_plot_landscape_pv`, `riemann_pv`) and noting
   removal at 3.0. (The 2D stereographic charts `riemann_chart` / `riemann_hemispheres`
   are **not** affected — they are matplotlib 2D and stay.)
3. **Dependency ergonomics.** Add a `[3d]` extra in `pyproject.toml` aliasing `[pyvista]`,
   and document that new 3D features are PyVista-only. PyVista stays optional in 2.x and
   becomes required at 3.0 (the actual switch is the separate `require-pyvista-3d-backend`
   change).

## Non-goals

- **Removing** any matplotlib 3D code — that is the breaking `require-pyvista-3d-backend`
  change at 3.0.
- Making PyVista a required dependency — also deferred to 3.0.
- Version/license reconciliation — handled by `reconcile-versioning-and-license`.
- Tooling (ruff, CI matrix) — separate change.

## Impact

- Affected code: `complexplorer/plotting/matplotlib/plot_3d.py` (add warnings; follow the
  existing `DeprecationWarning` pattern in `complexplorer/utils/validation.py`).
- Affected packaging: `pyproject.toml` (`[3d]` extra).
- Affected docs: new `docs/development/backend-policy.md`; README/3D docs note.
- Affected specs (additive): `plotting-3d-mpl` (deprecation requirement), `packaging`
  (backend dependency strategy).
- Risk: low. Warnings are advisory; existing calls keep working. Verify the warning fires
  once per deprecated entry point and that the test suite still passes (tests that call
  these functions may need `pytest.warns` / filter adjustments).
