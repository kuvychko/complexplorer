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
- machine-readable **answer keys / level data** (JSON: zeros/poles with orders, branch
  points, expected monodromy, recommended domain) — the *ground truth* Godot is validated
  against,
- STL / object cards.

Consequence: the "games" are **curated preset sets + a JSON level-export profile**, not
Python engine code. They ride on the Phase 2 preset/gallery/export infrastructure. No
interactive loops, scoring, or UI live in complexplorer.

**Parametrized families (Möbius, later Julia / `z^(1/n)` / resonators).** The Phase 2 base
`FunctionPreset` is **static** (one callable + expression + a fixed answer key). A
parametrized "playground" is a Phase-3 extension, `FunctionFamily`: a parameter schema
(names, types, defaults, ranges, constraint), a `make_callable`/expression template, and
`singularities(**params)`. `family.bind(**params)` emits an ordinary static
`FunctionPreset` snapshot — so the base model needs no change, and snapshot answer keys
become the validation oracle for Godot's native math. (A path through a family's parameter
space is exactly `create_animation`'s `f(z, t)`.)

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
         add-cli                                        2.3     no         in-progress
         add-gallery-generator                          2.3     no         planned
Phase 3  add-level-export                               2.4     no         planned
         (game PROTOTYPING assets — see "Games boundary" below)
         curated preset SETS (data, not code): singularity-detective,
         branch-cut-zoo, function-guessr, mobius — tagged collections +
         export profiles, NOT engine code
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

## Known bugs (backlog — fix as dedicated changes)

- **Colormaps emit invalid / non-deterministic RGB at in-domain poles.** `Colormap.rgb()`
  feeds a NaN hue (non-finite `f` on a grid node) to matplotlib's `hsv_to_rgb`, whose
  `(h*6).astype(int)` produces run-varying garbage. The `outmask` path only covers
  out-of-*domain* points, not in-domain singularities. Surfaced 3×: matplotlib 3D raises
  on it (band-aided by clipping facecolors in `plotting/matplotlib/plot_3d.py`), PyVista
  renders garbage, and it makes pole-node pinning tests flaky. Real fix: sanitize
  non-finite `f` in the base colormap (then drop the facecolor band-aid). Own change.

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
