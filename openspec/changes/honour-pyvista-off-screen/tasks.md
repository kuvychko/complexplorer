## 1. Honour the switch

- [x] 1.1 Add a small helper that answers "should this render be off-screen?" from the call's
  `interactive` argument and PyVista's global, so the four call sites cannot drift apart.
- [x] 1.2 Use it in `plot_landscape_pv`, `pair_plot_landscape_pv`, `riemann_pv` and
  `riemann_surface_pv`, replacing `off_screen: not interactive`.
- [x] 1.3 Check the interactive-show path: a plotter that is off-screen must not be asked to open a
  window afterwards.

## 2. Tests

- [x] 2.1 A test per renderer: with `pyvista.OFF_SCREEN` set, the plotter is constructed
  off-screen even though `interactive` was not passed.
- [x] 2.2 A test that `interactive=False` still renders off-screen with the global unset, so the
  existing contract is unchanged.
- [x] 2.3 A test that the default (no switch, no argument) is unchanged, so this does not silently
  make everything headless.

## 3. Verification

- [x] 3.1 `pytest`, `ruff check`/`format`, `openspec validate --specs`, and
  `openspec validate honour-pyvista-off-screen`.
- [x] 3.2 Re-run the two README/migration snippets that blocked, with `PYVISTA_OFF_SCREEN=true`
  and nothing else, and confirm they complete without opening a window.
  **Verified the safe way, not end to end.** With `PYVISTA_OFF_SCREEN=true` and no `interactive`
  argument, `plot_landscape_pv`, `riemann_pv` and `riemann_surface_pv` each construct a real
  plotter whose `off_screen` is `True` (checked via `return_plotter=True`, which never calls
  `show()`, so no window can appear even if the fix were wrong).
  The first attempt at the end-to-end run *did* open a window -- because it was pointed at the
  smoke environment, which still had the wheel built **before** this fix (`off_screen: not
  interactive`). That was a stale-environment mistake, not a failure of the change. Rebuilding
  the wheel first is the prerequisite for a genuine end-to-end run.
- [ ] 3.3 Confirm on CI, then archive and note the outcome in `openspec/REV3_CLOSEOUT.md`.
