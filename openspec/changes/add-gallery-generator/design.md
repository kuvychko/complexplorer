# Design — gallery generator

## Shape

```
complexplorer/gallery.py   (matplotlib-only, PyVista-free)

  generate_gallery(out_dir, *, selection=None, dpi=150) -> dict   # out_dir required; returns manifest
    presets = _resolve(selection)                 # ids | tag | None(all)  -> sorted by id
    records = []
    for p in presets:                             # deterministic order
        domain = domain_from_spec(p.domain_spec)
        cmap   = cmap_from_spec(p.cmap_spec)
        _render_portrait(p.func, domain, cmap, out_dir/p.id/"portrait.png", dpi)
        rec = {**p.to_dict(), "files": {"portrait": f"{p.id}/portrait.png"}}
        _write_json(out_dir/p.id/"card.json", rec)
        records.append(rec)
    manifest = {"schema_version": 1,
                "complexplorer_version": __version__,
                "generator": "complexplorer gallery",
                "presets": records}               # already id-sorted
    _write_json(out_dir/"index.json", manifest)
    return manifest

cli/main.py :: cmd_gallery  ->  generate_gallery(...)   # --tag T | --preset ID… | -o DIR
```

## Determinism mechanics (the whole point)

| Threat | Mitigation |
|---|---|
| Timestamps in JSON | Never write one. No `datetime`. `complexplorer_version` is the only varying field, and it's constant within a run. |
| Iteration order | `sorted(presets, key=id)`; `presets` list already ordered. |
| JSON key order | `json.dump(obj, f, sort_keys=True, indent=2, ensure_ascii=False)` + trailing newline. |
| Absolute paths leak in | `files` uses **relative** paths (`<id>/portrait.png`); bundle is relocatable. |
| PNG metadata | `plot_2d` does a bare `plt.savefig` — so the gallery renders to a figure and saves it itself: `fig.savefig(path, dpi=dpi, metadata={"Software": None})`, then `plt.close(fig)`. |

**Figure ownership (review finding).** `plot_2d(ax=None)` draws on the *global* pyplot
current axes (`plt.imshow`/`plt.gca()`) and never creates a fresh figure — so a render loop
would stack images on one accumulating axes. `_render_portrait` therefore **owns the figure**:

```python
fig, ax = plt.subplots(figsize=(4, 4))           # fixed size for reproducibility
plot_2d(domain, func, cmap=cmap, ax=ax)          # explicit ax -> ax.imshow, no global state
fig.savefig(path, dpi=dpi, metadata={"Software": None})
plt.close(fig)
```

(`scaling_spec` is carried in the card as data but unused by the 2D portrait — it's for the
3D/STL consumers.)

## Spike result (run during review)

Rendered `pole_flower_10` twice with the figure-owning path above (Agg backend,
`metadata={"Software": None}`) and diffed the bytes:
- **PNG byte-identical across runs** (115 520 bytes); no `Software` tag, no `tIME` chunk.
- `to_dict()` is JSON-serializable, stable under `sort_keys`, and contains **only builtin
  types** (no numpy floats leaking in) — so the manifest is clean and deterministic.

So portraits are byte-stable **within an environment** (same matplotlib/freetype), and the
run-twice-diff test may assert PNG equality too. The *cross-environment / cross-version*
caveat still stands (the spec keeps "images best-effort"); the **manifest** is the hard
cross-version guarantee.

## Test strategy

- **Determinism (the contract):** call `generate_gallery(tag=…, out_dir=A)` and again into
  `out_dir=B`; assert `index.json`, every `card.json`, **and every `portrait.png`** are
  **byte-identical** A vs B (portraits are byte-stable within one environment — see spike). No
  committed golden fixture (would break each version bump) — compare two live runs.
- **Structure:** every selected preset has `<id>/portrait.png` (non-empty) + `<id>/card.json`;
  `index.json.presets` is id-sorted; `files.portrait` is relative and exists; record keys ==
  `to_dict()` keys + `files`.
- **Round-trip sanity:** `domain_from_spec`/`cmap_from_spec` instantiate for every selected
  preset (already true for the catalog, but the gallery exercises all of them at once).
- **CLI:** `complexplorer gallery --tag <t> -o tmp` exits 0 and writes `index.json`.

## Open micro-decisions (resolve in implementation)

- `index.json` as a wrapper object `{…, presets:[…]}` (chosen) vs a bare array — wrapper wins
  (versions the bundle once).
- Default `dpi` (150) and figure size — fixed constants for reproducibility.
- Whether to also write a per-card `schema_version` (yes, so a card is self-describing
  standalone).
- Empty selection (unknown tag → 0 presets): the CLI SHALL exit non-zero with a clear
  "0 presets matched" message rather than silently writing an empty bundle (a no-match tag is
  almost always a user typo). The library fn may still write an empty, valid manifest.
