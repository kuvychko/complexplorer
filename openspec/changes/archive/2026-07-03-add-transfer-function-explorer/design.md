# Design — add-transfer-function-explorer

## Context

Phase-5 EE feature pulled into 3.0.0. The roadmap's boundary discipline applies: complexplorer
exports *math*; audience-specific apps stay out. The full EE suite (filters, resonators, QCM,
RF) remains 3.1+ backlog — this change is only the transfer-function kernel and its canonical
views.

## Goals / Non-Goals

**Goals**: a serializable-friendly rational-function object that behaves as a plain complex
callable; the four canonical views; `s` and `z` systems; zero new dependencies.

**Non-Goals**: filter-family constructors, group delay, time-domain responses, Smith charts,
impedance/RF objects (3.1+); a `scipy.signal`-backed implementation (numpy `roots`/`polyval`
suffice; scipy stays a test oracle); catalog/gallery integration (presets carry plain-dict
specs — a `tf_spec` factory can be added later, additively).

## Decisions

1. **`TransferFunction.__call__` is the integration surface.** Rather than a parallel
   `plot_transfer_domain` renderer, the object is a complex callable, so `cp.plot(domain, tf)`,
   `cp.plot_landscape_pv`, `cp.riemann_pv`, `cp.quick_plot`, and `OrnamentGenerator` all work
   today. `transfer_portrait` is a thin composition over `plot()` that adds the EE annotations
   (markers + boundary), not a new rendering path.
2. **`cp.ee` namespace, flat inside.** The vision plan reserved `complexplorer/ee/`; keeping
   EE names out of the top-level package keeps the core surface math-generic (the audience-mode
   boundary), while `from complexplorer import ee` gives `cp.ee.TransferFunction` ergonomics.
   Only the module `ee` is importable from the package root.
3. **Coefficients in `numpy.polyval` order (highest degree first)** — consistent with the new
   algebraic-curves `p=` parameter and with scipy/matlab conventions.
4. **Both `s` and `z` systems now.** The contour/boundary machinery (imaginary axis vs unit
   circle) is shared by all four views; supporting `z` at birth costs a few lines and the
   roadmap explicitly framed the feature as "H(s)/H(z)".
5. **Auto frequency range from pole/zero magnitudes** (a decade beyond the extremes,
   log-spaced; fallback `[0.01, 100]` rad/s when no finite nonzero features exist). For `z`
   systems the contour is one period `ω ∈ (0, π]` by default.
6. **Stability is strict** (`Re p < 0`, `|p| < 1`); marginal poles count as unstable — the
   conservative engineering convention.

## Risks / Trade-offs

- [Pole/zero cancellation: common roots inflate both lists] → documented behavior (`poles`/
  `zeros` are the raw polynomial roots; no symbolic cancellation). Numerically robust
  cancellation is a rabbit hole deliberately avoided.
- [Nyquist plots of functions with poles on the contour blow up] → values are plotted as
  computed (inf/nan break the locus line naturally); no detour contours in the MVP.

## Migration Plan

Additive subpackage; single commit; rollback = revert.

## Open Questions

_None._
