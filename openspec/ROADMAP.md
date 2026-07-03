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
Phase 3  enrich-answer-key-stats                        2.4     no         archived
         · the ONLY surviving "games" work — see "Games boundary" below.
           Adds DERIVED geometry to the catalog card (count_by_type,
           min_separation) so the answer key is a sharper oracle. Pure math.
         · DISSOLVED: add-level-export. The gallery manifest already IS the
           Godot interchange; task/scoring/difficulty are game design and live
           in Godot, not here. Curated SETS already exist as tags.
Phase 5  add-transfer-function-explorer                 2.5     no         planned
         (EE pulled early — additive, not gated on 3.0)
────────────────────────────────────────────────────────────────────────────────
★ 3.0    require-pyvista-3d-backend                     3.0     YES        archived
         · PyVista is now a required core dependency; HAS_PYVISTA /
           HAS_STL_EXPORT flags removed; CI collapsed to one config
         · removed matplotlib 3D paths (plot_landscape,
           pair_plot_landscape, 3D riemann); plotting-3d-mpl retired
         · _version bumped to 3.0.0; PyPI 3.0.0 RELEASE held until
           add-riemann-surfaces lands (bundled 3.0)
         add-riemann-surfaces                           3.0     no         archived
         · sqrt(z), z^(1/n), branch points/cuts, monodromy
         EXAMPLES & DOCS REWORK (migrate-examples-and-docs, phased 3-change sub-block,
         gallery-first M1→M2→M3 — see "Examples & docs rework" section below):
         M1 restructure-examples                        3.0     no         archived
            · new examples/ layout (notebooks/, scripts/, gallery/); retired the
              TWO legacy hand-rolled generators (examples/generate_gallery.py +
              gallery/generate_gallery_images.py); culled archive/+old/; rewrote README
            · new `examples` capability spec (specs 16→17). Seam (D6): docs/gallery
              is 100% M2's. modulus_scaling_showcase deleted (mpl-3D) → M2 rebuilds
         M2 rebuild-gallery-from-registry               3.0     no         archived
            · examples/showcase.py (Option B): cp.catalog → hi-res 2D + PyVista
              3D/sphere/surface/relief screenshots (1560px); tag-driven render set;
              split manifest (showcase.json) + generated docs/gallery; colormap
              gallery. Library cp.gallery + index.json stay deterministic (untouched).
              Hero banner kept. STL not committed. Riemann spheres use domain=None
              (full sphere). examples spec 3→8 reqs
         M3 migrate-and-verify-notebooks                3.0     no         proposed
            · modernize the 4 notebooks onto the 3.0 surface (excise the obsolete
              matplotlib-vs-PyVista narrative; static backend; strip notebook=/show=);
              add Riemann-surface + preset-registry coverage (NOT perceptual — it does
              not exist); DoD = pytest --nbmake (local, opt-in). [examples] extra added
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

## Examples & docs rework

After the 3.0 *library* work landed (`require-pyvista-3d-backend` + `add-riemann-surfaces`),
`examples/` is stale: 2 of 4 notebooks call removed mpl-3D functions, and **two** legacy
hand-rolled gallery generators predate — and now duplicate — the canonical registry-driven
`cp.gallery`. Rather than a mechanical fix, this is a full rework, **phased into 3 sequenced
changes** under the umbrella name `migrate-examples-and-docs`. Run **gallery-first**:

```
 M1 restructure-examples ──► M2 rebuild-gallery-from-registry ──► M3 migrate-and-verify-notebooks
    (teardown + skeleton)      (the producer, Option B)            (the tutorials)
```

- **Single source of truth = the registry.** `cp.catalog` (17 curated presets) is canonical.
  The two hand-rolled generators are retired; `examples/` becomes a *consumer* of the
  registry, never a parallel catalog.
- **Option B — the deterministic/pretty split.** The library `cp.gallery` stays
  **PyVista-free and byte-deterministic** (it is the Godot/web interchange contract and must
  not drift). The new `examples/showcase.py` is the *presentation* layer on top: it reads the
  same registry but adds the nondeterministic high-res 3D / Riemann sphere / Riemann surface
  / STL screenshots the manifest deliberately omits. Two consumers, two jobs.
- **Committed but regenerable.** Gallery images stay committed (PyPI/GitHub render them) yet
  must reproduce exactly from `python examples/showcase.py` — no orphaned hand-named PNGs.
- **DoD for notebooks is execution.** "Migrated" means each notebook runs top-to-bottom via
  `nbconvert --execute` on the 3.0 surface (local bar). Whether notebook execution ever
  becomes a *CI* gate (nbmake/papermill) is a deliberately deferred, separate decision —
  examples are outside CI today, and gating on heavy PyVista-screenshot notebooks is its own
  problem. Keep it local for M3; revisit later.
- **Dependencies.** M2 and M3 both need M1's clean layout; M3 is otherwise independent of M2.
  Gallery-first is chosen because the README/PyPI images are the most public surface.

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
