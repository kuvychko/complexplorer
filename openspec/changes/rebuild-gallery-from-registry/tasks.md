## 1. The showcase producer

- [ ] 1.1 Create `examples/showcase.py` that imports `cp.catalog` and `generate_gallery`, and
  defines the tag→render-set map (`canonical`→{landscape,sphere}, `branches`→{surface},
  `ornament`→{ornament}, all→{portrait}) and the multivalued→family map
  (`sqrt`→power n=2, `cbrt`→power n=3, `log`→log)
- [ ] 1.2 Render the 2D portraits + `index.json` by calling `generate_gallery()` into
  `examples/gallery/` (do not reimplement; reuse the deterministic library path)
- [ ] 1.3 For each preset, render its tag-selected PyVista screenshots to
  `examples/gallery/<id>/<render-type>.png` via the `_pv` functions with
  `interactive=False, filename=...` (landscape→`plot_landscape_pv`, sphere→`riemann_pv`,
  surface→`riemann_surface_pv`, ornament→relief render); use the preset's `scaling_spec`
  for modulus where applicable
- [ ] 1.4 Pick a high-res window size/dpi target and a fixed camera per render type so the
  bundle is visually consistent

## 1b. Colormap gallery (reference function × the real colormaps)

- [ ] 1b.1 Render a designated reference preset's function (e.g. `rational_zeros_poles`) under the
  implemented colormaps — `Phase` (basic / enhanced `n_phi` / modulus-enhanced `r_linear_step` /
  phase+modulus), `Chessboard`, `PolarChessboard` (linear, log), `LogRings` — to
  `examples/gallery/_colormaps/<name>.png`. Do NOT reference the unimplemented perceptual family
  (`OklabPhase`, etc.) — only colormaps that exist in the public API

## 2. Presentation manifest (split from index.json)

- [ ] 2.1 Write `examples/gallery/showcase.json`: catalog data per preset + every render's
  relative path (`portrait`, and any `landscape`/`sphere`/`surface`/`ornament`), a `colormaps`
  section (reference preset id + each colormap render path), plus a `banner` key recording the
  curated hero composite; do NOT modify `index.json`
- [ ] 2.2 Ensure `showcase.json` is deterministic in structure (sorted keys, id-sorted presets,
  LF, trailing newline) even though the referenced images are best-effort

## 3. Generated docs gallery (middle path)

- [ ] 3.1 Have `showcase.py` emit `docs/gallery/gallery.generated.md`: one entry per rendered
  preset with its image(s), a REGISTRY-DRIVEN code snippet (`cp.catalog.get(<id>)` +
  `.func`/`.domain()`/`.colormap()` or the matching `riemann_surface_pv` family call, expression
  as a comment) that runs as shown, and `title`/`story` text; snippets reference only
  3.0-surface APIs (NOT expression-as-lambda — the expression strings are math notation, see D4a)
- [ ] 3.2 Include a Colormaps section in the generated page (the reference function under each
  implemented colormap, with explicit-colormap snippets)
- [ ] 3.3 Rewrite `docs/gallery/README.md` as a thin hand-written frame (intro, Wegert
  reference, section prose) that includes or links `gallery.generated.md`; remove all
  removed-symbol code examples and the dead `plots_example.ipynb` / `domains_cmaps_example.ipynb`
  links

## 4. Retire legacy images and rewire links

- [ ] 4.1 Run `python examples/showcase.py` locally; commit the regenerated
  `examples/gallery/<id>/*.png`, `showcase.json`, and `gallery.generated.md`
- [ ] 4.2 Delete 24 of the 25 hand-named `examples/gallery/*.png`; KEEP the curated hero
  composite `Riemann_relief_map_20250726.png`
- [ ] 4.3 Repoint `README.md`'s 3 gallery thumbnail links (lines ~101–103) to id-based renders;
  LEAVE the header hero image (line ~10) pointing at `Riemann_relief_map_20250726.png`
- [ ] 4.4 Confirm no `.stl` is committed under `examples/gallery/`

## 5. Rebuild the modulus-scaling demo (PyVista)

- [ ] 5.1 Add `examples/scripts/modulus_scaling_showcase.py` as a PyVista demo iterating the
  scaling modes via `plot_landscape_pv(..., modulus_mode=...)`; repoint the README
  modulus-scaling reference to it

## 6. Guard test and verification

- [ ] 6.1 Add a non-render guard test that loads the committed `showcase.json` and asserts: it
  covers exactly the catalog ids, each preset's render paths match the tag policy, and no
  `surface`/`landscape`/etc. appears for a preset lacking the corresponding tag
- [ ] 6.2 Extend the M1 examples-structure guard (or add a check) asserting no `.stl` under
  `examples/gallery/`, and that the only non-id-addressed top-level image is the hero composite
  recorded under `showcase.json`'s `banner` key (no other hand-named legacy PNG remains)
- [ ] 6.3 Run `pytest tests/` — full suite green
- [ ] 6.4 Run `openspec validate rebuild-gallery-from-registry` and `openspec validate --specs`
  — clean
