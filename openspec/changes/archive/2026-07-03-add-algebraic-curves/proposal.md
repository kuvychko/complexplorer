# Add Algebraic-Curve Riemann Surfaces (w² = P(z))

## Why

3.0's headline feature made the Riemann surfaces of `z^(1/n)` and `log` first-class; the
natural next member of that family — the two-sheeted surface of `w² = P(z)` for a polynomial
`P` (hyperelliptic/elliptic curves, e.g. `w² = z³ − z`) — was on the 3.1+ backlog and has been
pulled into 3.0.0 (owner decision 2026-07-03). It extends the same "make branch points, cuts,
and sheets visible" story from a single branch point at the origin to an arbitrary
constellation of branch points at the roots of `P`, and it slots into the existing
`family=`-dispatched builder/renderer seam with no new architecture.

## What Changes

- `build_riemann_surface()` (mesh layer) gains `family="algebraic"` with a new `p` parameter —
  the polynomial coefficients of `P` in `numpy.polyval` order (highest degree first). The
  two sheets are the graphs of `±Re(√P(z))` over a polar disk grid of radius `r_max`
  (each graph is continuous; the sheets intersect exactly along the curves where
  `P(z) ≤ 0`, so cuts and monodromy are emergent, matching the "honest embedding" principle).
  Each sheet is colored by the phase of its value `w = ±√P(z)`; the phase jump across a cut
  *is* the monodromy (crossing continues onto the other sheet).
- Branch points (the roots of `P`, via `numpy.roots`) are recorded in the returned
  `SurfaceMesh.metadata` so downstream consumers can annotate them.
- `riemann_surface_pv()` (renderer) accepts and forwards `family="algebraic"` and `p`,
  labeling the height axis `Re w` as for the power family.
- Input validation: `p` must be a sequence of at least two coefficients with a nonzero
  leading coefficient (degree ≥ 1); requesting `family="algebraic"` without `p` is an error.
- Additive only — the `power` and `log` families are byte-for-byte unchanged.

## Capabilities

### New Capabilities

_None._

### Modified Capabilities

- `riemann-surfaces`: the supported-families requirement is extended with the algebraic
  family `w² = P(z)` (new scenario), and a new requirement covers branch-point metadata and
  `p` validation.

## Impact

- `complexplorer/mesh/riemann_surface.py`: `_algebraic_grid()` helper, `p` parameter,
  family registration, metadata.
- `complexplorer/plotting/pyvista/riemann_surface.py`: `p` kwarg, `_OWN_KWARGS`, docstring,
  axis label.
- Tests: `tests/unit/mesh/test_riemann_surface.py` and
  `tests/unit/plotting/pyvista/test_riemann_surface_pv.py` gain algebraic cases.
- `CHANGELOG.md` Added entry.
- Deliberately out of scope (deferred, additive): a catalog preset + `examples/showcase.py`
  `SURFACE_FAMILY` entry for an elliptic curve — adding a preset would change the byte-stable
  gallery `index.json` contract right before release and belongs in its own change.
