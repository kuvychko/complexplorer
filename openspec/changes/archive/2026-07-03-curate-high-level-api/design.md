# Design — curate-high-level-api

## Context

3.0.0 is about to ship from the `docs/openspec-baseline` branch. `complexplorer/api.py`
contains three unimplemented or half-implemented public callables and two aliases; a dead
plotter base-class layer (`plotting/base.py`) is star-exported but used by no concrete
plotter. All of this is removable *now* at zero semver cost because 3.0 is already the
breaking release; after 3.0 each removal would require a 4.0.

## Goals / Non-Goals

**Goals**
- 3.0.0 exports only implemented functionality from the high-level API.
- One quick-exploration entry point: `quick_plot`.
- Spec, code, and CHANGELOG agree.

**Non-Goals**
- Implementing animation, comparison grids, or zero/pole detection (candidates for 3.1
  as additive features — see openspec/ROADMAP.md).
- Touching `export/base.py` (live: `BaseExporter` is real infrastructure).
- Renaming `api.Presets` (docstring clarification only; renaming is gratuitous churn).

## Decisions

1. **Delete rather than deprecate.** 3.0 is the breaking boundary; shipping
   `DeprecationWarning` stubs for functions that never worked would preserve broken
   promises for another major cycle. Alternative considered: implement minimal versions —
   rejected by owner decision 2026-07-03 (defer to 3.1 as additive).
2. **`analyze_function` goes entirely** rather than losing only `show_zeros`/`show_poles`:
   without detection it reduces to `quick_plot` with a different colormap default; keeping
   it would preserve a misleading name ("analyze" implies computation).
3. **Keep `quick_plot`, drop aliases.** `quick_plot` is the name used by tests, docs, and
   the archived `fix-quick-plot-backend` change; `visualize`/`explore` are verified unused.
4. **Delete `plotting/base.py` wholesale** (with its test module) instead of wiring the
   concrete plotters onto it. The concrete plotters have shipped for two majors without it;
   an abstraction with zero implementors is speculative weight.
5. **Spec true-up rides this change**: the never-implemented `show` requirement is REMOVED
   in the delta spec instead of being "fixed", because specs are behavioral contracts of
   real behavior.

## Risks / Trade-offs

- [Users of 2.x `visualize`/`explore`/`analyze_function` break] → Already a breaking
  release; CHANGELOG "Removed" section names each symbol and its replacement.
- [Hidden internal usage of `plotting.base`] → Mitigation: repo-wide grep in tasks before
  deletion; test suite must stay green.

## Migration Plan

Single commit on the release branch; no data or deployment concerns. Rollback = revert.

## Open Questions

_None._
