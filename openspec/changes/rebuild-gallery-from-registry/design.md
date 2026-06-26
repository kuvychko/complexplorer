## Context

After M1, the registry (`cp.catalog`, 17 presets) is the single source of truth and `cp.gallery`
is the canonical generator — but it is **2D-only and PyVista-free by design** (its `index.json`
is a byte-deterministic interchange contract). The *visual* gallery shown on GitHub/PyPI needs
3D landscapes, Riemann spheres, and the new Riemann **surfaces**, which only PyVista produces.
M1 deliberately deferred to M2: the 25 hand-named `examples/gallery/*.png`, the `README.md`
gallery image links, the wholesale rewrite of `docs/gallery/README.md` (it still shows removed
mpl-3D code examples), and the rebuild of the deleted `modulus_scaling_showcase.py`.

Grounding established during exploration:
- All four `_pv` renderers already accept `filename=` + `interactive=False` → off-screen
  screenshot via `handle_export`. No new rendering primitive is needed.
- Off-screen VTK screenshots crash **only** on headless Windows CI (`win32 AND CI==true`);
  locally the real-screenshot tests run and pass. So a local-only producer is safe.
- Tags already encode mathematical character: `branches`={sqrt,cbrt,log}, `ornament`=
  {pole_flower_10,sqrt}, `canonical`=7 hero functions. The render set falls out of the tags.

## Goals / Non-Goals

**Goals:**
- One local command (`python examples/showcase.py`) regenerates every gallery image, the
  presentation manifest, and the docs gallery page.
- Keep `cp.gallery`/`index.json` deterministic and untouched (the Godot/math interchange).
- Retire all hand-named PNGs; every committed image is id-addressed and reproducible.
- A docs gallery whose code snippets are derived (cannot drift, never reference removed symbols).

**Non-Goals:**
- No change to `complexplorer/gallery.py`, the catalog, or any library rendering API.
- No CI gate on rendering (local-only producer). No notebook work (that is M3).
- No committed `.stl` (a future dedicated 3D-printing site hosts downloadable meshes).
- No new render *primitives* — the showcase composes existing `_pv` functions.

## Decisions

**D1 — `showcase.py` wraps `cp.gallery`; it does not extend or modify it.** The showcase calls
`generate_gallery()` to produce the deterministic 2D portraits + `index.json`, then augments each
`<id>/` directory with PyVista screenshots. The library stays PyVista-free and deterministic;
all PyVista use lives in `examples/`. *Alternative:* add a PyVista path to `cp.gallery` —
rejected; it breaks the manifest's byte-stability contract (D3) and pushes rendering into the
library.

**D2 — Tag-driven render policy, expressed in the showcase.** A small mapping in `showcase.py`:
`canonical → {landscape, sphere}`, `branches → {surface}`, `ornament → {ornament}`, all →
`{portrait}`. The render type follows from the math the tag describes — the presentation layer
*reads* the math and chooses, honoring the roadmap's math/game boundary. The multivalued mapping
is explicit: `sqrt→(power, n=2)`, `cbrt→(power, n=3)`, `log→(log)`. *Alternative:* a
`gallery_renders` field on `FunctionPreset` — rejected; it injects presentation data into the
Godot interchange record. *Alternative:* a hand-curated `(id, render)` table — rejected as
redundant with tags, though the tag map is trivially overridable if an editorial exception ever
arises.

**D3 — Split manifests.** `index.json` (deterministic, math/Godot interchange) stays exactly what
`cp.gallery` writes. The showcase writes a *separate* `showcase.json` (presentation: catalog data
+ all render paths). PyVista screenshots are environment-dependent and non-deterministic; keeping
them out of `index.json` preserves its byte-stable guarantee. The two manifests share preset ids
as the join key. *Alternative:* one manifest — rejected; it forfeits determinism for the half
that downstream math consumers rely on.

**D4 — Generated docs gallery, framed by a thin hand-written README (the "middle path").**
`showcase.py` emits `docs/gallery/gallery.generated.md` (one entry per rendered preset: image +
a derived code snippet + `title`/`story`). A short, hand-written `docs/gallery/README.md` provides
the prose framing (intro, the Wegert reference, section headers) and includes or links the
generated page. This gets zero-drift snippets without a wholly mechanical-feeling page.
*Alternative A:* fully generated README — rejected; reads mechanically, no room for editorial
prose. *Alternative B:* hand-written but manifest-checked — rejected; snippets still drift and
must be manually kept symbol-clean.

**D4a — Snippets are registry-driven, not expression-as-lambda.** Review finding: the preset
`expression` strings are math *notation*, not runnable Python — `sqrt(z)`, `exp(z)`, `log(z)`,
`sin(z)`, `tan(z)`, `exp(1 / z)` all reference bare names that need an import, so a naive derived
`lambda z: sqrt(z)` would not run. Therefore each snippet is generated against the **registry**
and matched to its render type, e.g.

```python
import complexplorer as cp
preset = cp.catalog.get("sqrt")                 # f(z) = sqrt(z)
cp.plot(preset.domain(), preset.func, cmap=preset.colormap())          # portrait
cp.riemann_surface_pv("power", n=2)                                    # surface
```

with the `expression` shown as a human-readable comment. This is always correct, always
3.0-symbol-clean, teaches the registry (a real feature), and avoids fragile expression→Python
rewriting and spec→constructor reconstruction. *Alternative:* rewrite each expression into a
namespaced runnable lambda (`np.sqrt`, …) and reconstruct `cp.Rectangle(...)`/`cp.Phase(...)`
from the specs — rejected as fragile (a per-function name table) for a marginal copy-paste gain.

**D5 — Per-id naming `<id>/<render-type>.png`.** Strict superset of `cp.gallery`'s
`<id>/portrait.png`. Predictable, collision-free, and lets `showcase.json` reference renders by a
simple convention. No flat hand-names ever again.

**D6 — STL not committed; the ornament render is a relief sphere.** The gallery represents
`ornament` presets with a relief PNG and links the generation code; downloadable `.stl` belongs
on a dedicated 3D-printing site (future). The relief render is distinct from the plain phase
`sphere.png`: `sphere.png` is `riemann_pv` with phase only (modulus flat), while `ornament.png`
is `riemann_pv` with the preset's `scaling_spec` applied as modulus relief — i.e. "what this
3D-prints as," the same family of image as the kept hero composite. So a preset tagged both
`canonical` and `ornament` (e.g. `pole_flower_10`) gets three non-redundant 3D views: analytic
landscape, phase sphere, relief sphere.

**D8 — Keep the hand-crafted hero composite as a curated banner (per request).** The README
header image `examples/gallery/Riemann_relief_map_20250726.png` is a hand-composed composite that
reads better than any single auto-render; it is kept. It is NOT a registry render and is exempt
from the id-addressed/regenerable invariant — recorded in `showcase.json` under a `banner` key
and documented as curated. All renders live in `<id>/` subdirectories, so the hero (a top-level
file) is structurally distinct from the regenerable bundle and the guard test can allow exactly
the manifests + banner at the top level. *Alternative:* move it to a separate `assets/` dir for
physical separation — deferred; keeping it in place avoids churning the README header link, and
the `<id>/`-subdir convention already separates curated-from-generated.

**D7 — Rebuild `modulus_scaling_showcase.py` as PyVista here.** M1 deleted the mpl-3D version; the
multi-mode composition is cheap once the showcase's screenshot helpers exist. It lands under
`examples/scripts/` and uses `plot_landscape_pv(..., modulus_mode=...)` across the scaling modes.

**D9 — Colormap gallery via a reference function (decided during review).** All 17 presets use
`Phase`, so the registry alone cannot showcase the colormaps. The showcase renders one designated
reference preset's function (e.g. `rational_zeros_poles`, which has zeros at ±1 and poles at ±i —
good colormap contrast) under the library's *implemented* colormaps, into a reserved
`_colormaps/` directory, recorded under a `colormaps` section of `showcase.json`. The reference
function stays registry-anchored (a preset id); the varied axis is the colormap, constructed
explicitly in each snippet. The real colormap set is `Phase` (basic), `Phase` enhanced (`n_phi`),
`Phase` modulus-enhanced (`r_linear_step`), `Phase` phase+modulus (`n_phi`+`auto_scale_r`),
`Chessboard`, `PolarChessboard` (linear), `PolarChessboard` (log), `LogRings` — i.e. exactly the
variants the old hand gallery showed. *Alternative:* add colormap-demo presets to the registry —
rejected; the function registry is about functions + answer keys (a chessboard demo carries no
singularity meaning). *Alternative:* defer colormaps to the M3 notebook — rejected by request; the
gallery is the visual showcase and should display them.

**D10 — The advertised "perceptual colormap family" does not exist; flag, do not fix here.**
Review finding: CLAUDE.md and `README.md` describe `OklabPhase`, `PerceptualPastel`,
`AnalogousWedge`, `DivergingWarmCool`, `Isoluminant`, `CubehelixPhase`, `InkPaper`,
`EarthTopographic`, `FourQuadrant` — none are implemented (`core/colormap.py` defines only `Phase`,
`Chessboard`, `PolarChessboard`, `LogRings`). The colormap gallery (D9) renders only the four real
classes. Whether to *implement* the perceptual family (a feature change) or *correct the docs* (a
doc fix) is a separate decision tracked outside M2; this change must not silently render or
reference colormaps that do not exist.

## Risks / Trade-offs

- **[Off-screen screenshots are non-reproducible across machines/GPUs]** → Mitigation: the
  *manifest* (`showcase.json`, like `index.json`) is the stable contract; images are
  reproducible best-effort, regenerable by anyone running the producer locally. Same stance
  `cp.gallery` already takes for portraits.
- **[`showcase.py` can't run in CI, so image regression isn't gated]** → Mitigation: a non-render
  guard test asserts `showcase.json` is consistent with the catalog and the tag policy; image
  correctness is a local/manual concern by design (examples are outside CI).
- **[The render budget (~36 images) grows as the catalog grows]** → Accepted: tag-driven means it
  scales with curation, not combinatorially; a new `canonical` preset adds 3 images, not N.
- **[Derived snippets may not capture every nuance of a hand-tuned example]** → Mitigation: the
  derived snippet reconstructs domain/cmap/function faithfully; the hand-written README frame
  carries any extra pedagogical prose.
- **[Repointing README/PyPI images changes the social-preview image]** → Accepted and intended;
  M2 is pre-3.0-release, so no published image URL has propagated. The hero banner (D8) is kept,
  so the most prominent header image is unchanged.
- **[Tag-driven renders may occasionally produce an awkward view]** — e.g. `sqrt` is tagged both
  `branches` and `ornament`, so it gets a relief-sphere render of a branch-cut function, which
  may look odd. → Mitigation: the fix is *curation* (adjust the preset's tags in the catalog),
  not special-casing in the producer; the mechanical tag policy stays simple, and any awkward
  render is a one-line tag change surfaced at first local run.

## Migration Plan

1. Build `examples/showcase.py`: wrap `generate_gallery()`; add the tag→render map and the
   multivalued→family map; render screenshots per preset via the `_pv` `filename=` path; write
   `showcase.json`; generate `docs/gallery/gallery.generated.md`.
2. Run it locally to produce `examples/gallery/<id>/*.png` + `showcase.json` + the generated page.
3. Delete the 25 hand-named PNGs; commit the regenerated id-based images.
4. Repoint `README.md`'s 4 gallery image links to id-based renders; rewrite
   `docs/gallery/README.md` as the thin hand frame including the generated page.
5. Rebuild `examples/scripts/modulus_scaling_showcase.py` as a PyVista demo.
6. Add the non-render guard test (`showcase.json` ↔ catalog/tag-policy consistency); run the
   suite; `openspec validate`.

Rollback: revert the commit; images and docs return to the M1 state.

## Open Questions

- Resolution/window-size target for the "high-res" renders (e.g. 1200² vs 1600²) — a tuning knob
  settled during implementation, not a contract; the spec says "high-resolution," not a number.
- Whether `gallery.generated.md` is `include`d (if the docs toolchain supports transclusion) or
  linked — depends on the eventual docs site; M2 can link and revisit when a docs framework lands.
