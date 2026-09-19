## Why

PyVista's documented way to go headless is `pv.OFF_SCREEN = True`, or the `PYVISTA_OFF_SCREEN`
environment variable that sets it. Complexplorer ignores both: every renderer computes
`off_screen = not interactive`, which overrides the global rather than deferring to it. Since
`interactive` defaults to `True`, a user who has done exactly what PyVista tells them to do still
gets a window — and on a machine that cannot open one, still fails.

This was found while running the README's own snippets against the built wheel: they blocked on an
interactive window despite both switches being set, twice, on a desktop. The same would happen to
anyone scripting the library on a headless box who reached for the switch PyVista documents.

Every renderer already honours its own `interactive=False` parameter. The gap is only that a
global instruction, set once for a whole session or a whole CI job, loses to a per-call default the
user never chose.

## What Changes

- The four places that construct a plotter — `plot_landscape_pv`, `pair_plot_landscape_pv`,
  `riemann_pv` and `riemann_surface_pv` — render off-screen when **either** the call asks for it
  (`interactive=False`, or a `filename`) **or** PyVista's global says the session is off-screen.
- An explicit `interactive=False` keeps working exactly as before; nothing gains a new parameter.
- The behaviour is stated in the `plotting-3d-pyvista` capability, which owns PyVista rendering,
  and applies to every PyVista renderer the library exposes.

## Capabilities

### New Capabilities

_None._

### Modified Capabilities

- `plotting-3d-pyvista`: a session-level off-screen instruction is honoured by every PyVista
  renderer, not overridden by the per-call `interactive` default.

## Impact

- **Code:** `complexplorer/plotting/pyvista/plot_3d.py` (two sites), `riemann.py`,
  `riemann_surface.py`.
- **Behaviour:** strictly fewer windows. A caller who sets neither switch sees no change; a caller
  who set the global and was ignored now gets what they asked for.
- **Not changed:** `mesh/surface.py`, which already passes `off_screen=True` unconditionally for
  its export path.
