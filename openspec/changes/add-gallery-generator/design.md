# Design — gallery generator

## Shape

```
complexplorer/gallery.py   (matplotlib-only, PyVista-free)

  generate_gallery(selection=None, out_dir, *, dpi=150) -> dict   # returns the manifest
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

`_render_portrait` calls `plot_2d(domain, func, cmap=cmap, filename=None)` to get the axes,
then `ax.figure.savefig(...)` with stripped metadata. (`scaling_spec` is carried in the card
as data but unused by the 2D portrait — it's for the 3D/STL consumers.)

## The implementation spike (de-risk before claiming pixel reproducibility)

Render one preset twice to two paths with `metadata={"Software": None}` and **diff the PNG
bytes**. Outcomes:
- Byte-identical → portraits are reproducible within the environment; say so (best-effort).
- Still differ (some backend stamps a date) → add the offending key to `metadata` and retry;
  if still non-deterministic, keep the contract as "manifest stable, pixels not asserted" and
  the test simply doesn't compare PNG bytes.

Either way the **manifest** guarantee is independent and holds.

## Test strategy

- **Determinism (the contract):** call `generate_gallery(tag=…, out_dir=A)` and again into
  `out_dir=B`; assert `index.json` and every `card.json` are **byte-identical** A vs B. No
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
