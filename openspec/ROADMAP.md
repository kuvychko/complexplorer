# Complexplorer Roadmap — the road to 3.0

**Status:** living document · **Scope:** v2.1 → v3.0 · **Owner:** Igor Kuvychko

This is the umbrella that sequences the individual OpenSpec changes into a coherent
journey. It is intentionally **not** an OpenSpec change (changes get archived; a roadmap
should not). The detailed source for the vision is
`complexplorer_phased_implementation_plan.md`; this file is the operational distillation:
what ships, in what order, as which version, tracked by which change.

---

## North star

> **Complexplorer makes complex-valued structure visible, explorable, and physical.**

Complexplorer is evolving from a domain-coloring package into a toolkit for
**visualizing, exploring, and fabricating** complex-valued mathematical and engineering
structures — serving three audience modes: research/math, engineering, and creative/public.

It should not compete with general plotting libraries. It specializes in the visual
grammar of complex functions: phase, modulus, zeros, poles, branch points, cuts, sheets,
maps, surfaces, and physical relief.

---

## Architecture spine: a sharp 2D / 3D capability line

The single most important structural decision: **draw a clean capability boundary by
dimensionality, and do not maintain backend parity across it.**

| Concern | Backend |
|---|---|
| 2D phase portraits, pair plots, static educational figures | **matplotlib** |
| 2D stereographic charts (`riemann_chart`, `riemann_hemispheres`) | **matplotlib** |
| 3D landscapes, Riemann relief, Riemann sphere, surfaces, mesh export | **PyVista** |

matplotlib's 3D rendering is low quality and a maintenance drag. Rather than deprecate it
slowly, **3.0 removes the matplotlib 3D paths entirely** — `plot_landscape`,
`pair_plot_landscape`, and the 3D `riemann()` surface. PyVista becomes a **required**
dependency at 3.0. This is the breaking change that *defines* the major version bump, and
it is justified by pairing it with the feature that genuinely needs a real 3D backend:
Riemann surfaces.

Everything else on the road to 3.0 is **additive and non-breaking**, shipped continuously
as 2.x minors so momentum and releases never stall behind one big branch ("thin 3.0").

---

## Games boundary: complexplorer is not a game engine

Complexplorer is used for **game prototyping only**. Actual games are built in **Godot**,
which reimplements the math **natively** (mobile platforms are the target). Complexplorer's
role is therefore to **produce, not run**:

- reference imagery (PNG) — what the phase portrait / relief *should* look like,
- machine-readable **answer keys** (the gallery manifest JSON: expression, domain/cmap/
  scaling specs, exact zeros/poles/essential/branch points with orders, plus derived
  geometry) — the *ground truth* Godot is validated against,
- STL ornaments.

**The decisive test (settled 2026-06-24): does it export _math_ or _game_?** complexplorer
exports mathematical/geometric *truth*; everything about *playing* belongs to Godot.

| MATH — complexplorer exports it | GAME — lives in Godot |
|---|---|
| expression, singularities `{type, at, order}` | task verb ("tap the poles") |
| derived geometry (counts, min separation) | scoring rule / partial credit |
| domain / cmap / scaling spec, portrait | position *tolerance* (a game knob) |
| behavior facts (has-∞, has-essential) | difficulty rank/tier, progression, packs, UI |

Consequence — **the gallery manifest already _is_ the Godot interchange.** `index.json`
carries expression + domain/cmap/scaling + the full singularity answer key + portrait, in a
byte-stable contract Godot can consume today. A separate "level-export" profile that added
difficulty/scoring/tolerance/task would import *game design* into a numerical library — the
exact boundary erosion this section exists to prevent — so it is **cut**. The only surviving
"games" work is `enrich-answer-key-stats`: pure derived geometry on the card. Curated SETS
already exist as **tags** (`singularity-detective`, `branch-cut-zoo`, …). No interactive
loops, scoring, difficulty, or UI live in complexplorer.

**Function presets are static — by decision.** The `FunctionPreset` is one callable +
expression + a fixed answer key, and stays that way. A parametrized-family abstraction
(`FunctionFamily`) was considered and **cut** (2026-06-24): its only concrete consumer was
the now-dissolved games layer, the gallery uses static snapshots, and `create_animation`
already takes `f(z, t)` directly — so it was scope without a driver. If a real
visualization need appears (interactive parameter exploration), it gets designed fresh then,
not carried as speculative roadmap weight.

---

## Release map

```
            CHANGE (OpenSpec proposal)                 VER     BREAKING?  STATUS
────────────────────────────────────────────────────────────────────────────────
Phase 0  reconcile-versioning-and-license              2.1     no         archived
         establish-backend-and-release-policy          2.1     no         archived
         add-tooling-and-ci                            2.1     no         archived
Phase 1  add-pyvista-surface-kernel                    2.2     no         archived
Phase 2  add-function-preset-registry                  2.3     no         archived
         add-cli                                        2.3     no         archived
         fix-colormap-nonfinite                         2.3     no         archived
         add-gallery-generator                          2.3     no         archived
         · gallery: 2D portraits + a BYTE-STABLE JSON manifest (the
           deterministic contract + Godot/web interchange); images are
           best-effort. Gated on the colormap fix (clean, deterministic RGB).
Phase 3  enrich-answer-key-stats                        2.4     no         proposed
         · the ONLY surviving "games" work — see "Games boundary" below.
           Adds DERIVED geometry to the catalog card (count_by_type,
           min_separation) so the answer key is a sharper oracle. Pure math.
         · DISSOLVED: add-level-export. The gallery manifest already IS the
           Godot interchange; task/scoring/difficulty are game design and live
           in Godot, not here. Curated SETS already exist as tags.
Phase 5  add-transfer-function-explorer                 2.5     no         planned
         (EE pulled early — additive, not gated on 3.0)
────────────────────────────────────────────────────────────────────────────────
★ 3.0    require-pyvista-3d-backend                     3.0     YES        planned
         · PyVista becomes a required dependency
         · remove matplotlib 3D paths (plot_landscape,
           pair_plot_landscape, 3D riemann)
         · retire the plotting-3d-mpl capability spec
         add-riemann-surfaces                           3.0     no         planned
         · sqrt(z), z^(1/n), branch points/cuts, monodromy
────────────────────────────────────────────────────────────────────────────────
3.1+     OUT OF UMBRELLA SCOPE (future backlog)
         full EE (filters, resonators, QCM, RF bridge),
         w² = P(z) algebraic curves, objects/project-cards,
         special-function atlas, conformal/hyperbolic labs
```

`STATUS` values: `planned` → `proposed` (OpenSpec change exists) → `in-progress` →
`archived` (synced into `specs/`). Update this column as changes move.

---

## Sequencing rationale

1. **Phase 0 first, always.** The repo currently has an incoherent version (tags say
   `v2.0.0`; `_version.py` says `1.0.0`; `pyproject.toml` says `1.0.1`) and a license
   contradiction (`LICENSE` = MIT, `pyproject` classifier = BSD). You cannot credibly
   release 3.0 on top of that. Hygiene is split from policy so the pure metadata fixes
   don't wait on any design discussion.
2. **Kernel before features.** `add-pyvista-surface-kernel` (the `SurfaceMesh`
   abstraction + shared mesh pipeline) underpins relief maps, Riemann surfaces, and STL
   export. Build it once so later phases are cheap. It must reconcile with the existing
   `export/stl/OrnamentGenerator` and `utils/mesh.py` rather than duplicate them.
3. **Presets are leverage — and the Godot interchange.** A metadata-rich function preset
   registry powers gallery pages, CLI rendering, object cards, AND the game level data
   from one source — so it lands before everything that consumes it. Under the games
   boundary above, the preset is the serializable record handed to Godot: it carries a
   `callable` (complexplorer renders with it) + an `expression` string (Godot reimplements
   from it) + plain-dict `domain_spec`/`cmap_spec`/`scaling_spec` (no live objects) +
   hand-authored `singularities` (the exact answer key). Serialization is a design center,
   not an afterthought; `core/presets.py` stays PyVista-free.
4. **3.0 is anchored on the breaking change**, not on feature volume. It contains exactly
   two changes: the dependency/removal change and the headline feature.

---

## PyVista compatibility

PyVista churns its API across minor releases; CI tests against the latest via unpinned
ranges, so breaks surface early. Notes:

- **Floor: `pyvista>=0.47`** (the release that dropped the `box` kwarg from `add_axes`;
  our orientation widget is box-free, validated on 0.47 and 0.48).
- **`extract_surface(algorithm="dataset_surface")` is pinned explicitly** everywhere — 0.48
  warned that the default will change to `None`; pinning locks current mesh behavior and is
  forward-compatible.

---

## Known bugs (backlog — fix as dedicated changes)

_None currently tracked. Fixed bugs become archived changes (e.g.
`fix-colormap-nonfinite`, `fix-quick-plot-backend`)._

---

## Design principles (carried from the plan)

- **Make mathematical structure explicit** — preserve concepts (zero, pole, branch point,
  sheet, cut, path, monodromy, transfer function, resonance), don't just render arrays.
- **Keep 2D and 3D responsibilities separate** — see the capability line above.
- **Prefer presets over one-off examples** — one preset powers gallery + game + STL +
  card + tutorial.
- **Avoid fake generality** — support `sqrt(z)`, `z^(1/n)`, `w² = P(z)` very well rather
  than a grand API that fails on hard cases.
- **Treat objects as first-class outputs** — images and meshes carry reproducible
  metadata (function, domain, scaling, colormap, version, generation script).

---

## How to use this roadmap

- Each non-trivial step is an OpenSpec change under `openspec/changes/`.
  Create them with `/opsx:propose` (or `/opsx:explore` first to think).
- Keep changes **per-capability**, not per-phase — phases are the grouping layer here,
  not the unit of work.
- When a change is created, flip its `STATUS` to `proposed`; when archived, to `archived`.
- Validate specs with `openspec validate --specs`.
