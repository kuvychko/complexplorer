## Context

The 3.0.0 candidate is functionally sound (466 tests green, 87% coverage, clean wheel
install, working CLI) but carries pre-3.0 residue: internal PyVista-optional guards that
contradict the "PyVista required" decision, ~700 lines of dead code about to be frozen into
the public surface, several silent-failure correctness bugs, and packaging gaps that break
the PyPI project page. This change is a single cross-cutting hardening pass rather than one
change per capability, because the items are individually small, share test/spec churn, and
are all gated on the same event: tagging 3.0.0. See proposal.md for the itemized findings.

Constraints: matplotlib is the 2D backend, PyVista (>=0.47) the sole 3D backend and a hard
dependency. The library raises `ComplexplorerError` subclasses (`ValidationError`), never
bare `ValueError`/`RuntimeError`/`ImportError`. The gallery `index.json` manifest is a
byte-stable contract and must not drift.

## Goals / Non-Goals

**Goals:**
- Make the internal code match the shipped 3.0 contract (PyVista unconditionally imported).
- Remove dead/unused/duplicated surface while it is still free to delete.
- Fix silent-failure bugs so documented arguments do what they say.
- Make the distribution correct on PyPI (renders, typed, licensed).
- Add real (non-mocked) render coverage for the flagship PyVista paths.

**Non-Goals:**
- Reordering established public signatures. `plot(domain, func)` stays domain-first;
  `riemann_chart(func, domain=)` stays func-first. Churning the most-used call for
  consistency is high-risk, low-value even at a major bump.
- Unifying the default colormap across backends (2D `auto_scale_r=True` vs 3D `v_base=0.6`).
  These are deliberate per-backend defaults; changing them silently alters everyone's
  output. Documented, not changed.
- New features. This is hardening only; animation/comparison stay out (per roadmap 3.1+).
- Touching the gallery manifest / `cp.gallery` determinism.

## Decisions

**1. Delete the PyVista guards rather than soften the CHANGELOG.** PyVista is required; the
`try/except ImportError → HAS_PYVISTA` pattern and `check_pyvista_available()` are dead
branches (`# pragma: no cover`) that also violate the exception convention. Replace with
`import pyvista as pv` at module top. The `stl-export` spec's "Optional-dependency gating"
requirement is retired in favor of "STL export is always available." Alternative (keep flags,
reword docs) rejected: it preserves dead code and a false capability signal.

**2. `riemann_chart` masks via `domain.contains()`.** The dead `mask_list` branch is
replaced with the same masking `plot()` uses (blank out-of-domain samples), making the
documented `domain` argument real. Alternative (remove the parameter) rejected: masking a
stereographic chart to a sub-domain is a legitimate, documented use.

**3. Unknown-kwarg handling: validate, don't forward.** The PyVista entry points currently
pass `**kwargs` into `pv.Plotter` through hand-maintained blocklists. Most blocklist entries
are named parameters that can never appear in `**kwargs`; the only live effect is silently
swallowing legacy names (`show`) or crashing on others (`n_theta`). Decision: stop
forwarding arbitrary kwargs; accept only the explicit signature, and raise `ValidationError`
naming the 3.0 replacement for the known-removed names (`n_theta`/`n_phi` → `resolution`,
`show` → `interactive`, `project_from_north` → removed). Alternative (extend the blocklist)
rejected: it perpetuates a leaky, drift-prone filter.

**4. `Rectangle.contains` uses actual dimensions.** Membership is computed from
`re_length`/`im_length` about `center`, decoupled from the square-padded viewing window
(which stays as-is for display). This is a behavioral break, acceptable pre-first-3.0-publish
and correct per the `domains` "Region membership testing" requirement.

**5. Dead-code deletion is mechanical but spec-light.** Most removals
(`export/base.py`, `Matplotlib2DPlotter`, `utils.mesh` aliases + `RectangularSphereGenerator`,
`compute_riemann_sphere_distortion`, `warn_deprecated`, unused `validate_*`,
`ensure_consistent_normals`, the `stereographic` alias) touch no stated requirement, so they
live in tasks, not delta specs. Guard against silent breakage by grepping tests/examples for
each symbol before deletion and updating `__all__`/`__init__` exports.

**6. Consolidation preserves behavior.** The shared modulus-scaling dispatch, shared
domain/`z`/`f` input resolution, and `OrnamentGenerator.save_stl → SurfaceMesh.save_stl`
delegation must be byte-for-byte behavior-preserving; the existing regression tests
(surface kernel, ornament generator) are the guardrail. If any output shifts, stop and treat
it as a separate decision.

**7. Packaging.** Adopt SPDX `license = "MIT"` (requires setuptools>=77 in build-system);
add `complexplorer/py.typed` with a `package-data`/`include` entry; split `all` to
user-facing extras only (`qt`, `pyvista` already core) and leave a `dev` extra for tooling;
rewrite README asset URLs to `raw.githubusercontent.com/kuvychko/complexplorer/main/...`.

## Risks / Trade-offs

- **[Non-square Rectangle behavior change silently alters existing users' plots]** → It is a
  bugfix and 3.0 is unpublished; call it out explicitly in CHANGELOG under Breaking Changes.
- **[Deleting `export/base.py` / exported helpers breaks a downstream importer]** → All are
  unexported internals or unused exports (verified by repo-wide grep during review); the only
  importers are their own tests, which are deleted/updated in the same change.
- **[Real off-screen PyVista test is flaky/slow in CI]** → Use `off_screen=True` +
  `pv.OFF_SCREEN`, assert on returned mesh/plotter state rather than pixels; keep it to one
  or two smoke renders, not a screenshot-diff suite.
- **[SPDX license key trips an older setuptools]** → Pin `setuptools>=77` in
  `[build-system].requires`; verify `uv build` + fresh-venv install still succeed.
- **[Kwarg validation rejects a kwarg some example/notebook still passes]** → Grep
  examples/notebooks/docs for the removed names and update them in the same change.

## Migration Plan

1. Land spec deltas + tasks; implement in dependency order (guards → dead code → bugs →
   packaging → consolidation → tests).
2. Run `pytest`, `ruff check`, `openspec validate --specs`, `uv build`, and a fresh-venv
   install smoke test (`complexplorer list`; import `cp`; one real off-screen render).
3. Update CHANGELOG's 3.0.0 Breaking Changes with the `HAS_PYVISTA` removal and the
   Rectangle membership fix.
4. Rollback is a plain `git revert` of the change branch; no data or state migration.

## Open Questions

- Should `riemann_chart(domain=...)` mask by blanking (NaN → out-of-domain color) to match
  `plot()`, or clip the chart extent? Default to blanking for consistency unless review
  prefers clipping.
- Keep `warn_deprecated` as dormant future infrastructure, or delete now? Leaning delete
  (nothing triggers it; re-add when a real deprecation appears).
