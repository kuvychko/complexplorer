# Add gallery generator

## Why

The catalog (`cp.catalog`) holds ~17 curated presets with exact answer keys, but nothing
turns it into a **published, reproducible asset bundle**. The only existing gallery code is
`examples/generate_gallery.py` — ad-hoc, hard-coded (not catalog-driven), and
non-deterministic (it stamps `datetime` into output). It cannot serve as the interchange
record for Godot game prototyping, a future docs site, or Phase 3's level-export.

This change adds a **gallery generator**: batch the catalog into per-preset 2D portraits plus
a **byte-stable JSON manifest**. The manifest — not the pixels — is the deterministic
contract and the foundation that Phase 3 (`add-level-export`), Godot, and any future web/docs
layer consume. (There is no docs framework today — `mkdocs.yml`/`site/` are gone — so this
produces an asset bundle, not human-facing pages.)

## What changes

- **New library function** `complexplorer/gallery.py :: generate_gallery(selection, out_dir)`
  (matplotlib-only, PyVista-free). `selection` is a list of preset ids, a tag, or `None`
  (all). Iterates presets **sorted by id** and writes, per preset:
  - `<id>/portrait.png` — the 2D phase portrait (PNG metadata stripped, fixed dpi),
  - `<id>/card.json` — `preset.to_dict()` + `{"files": {"portrait": "<id>/portrait.png"}}`.
- **Top-level `index.json`** — a self-contained manifest:
  `{schema_version, complexplorer_version, generator, presets: [<full record>, …]}` with
  records sorted by id and **relative** file paths (the bundle is relocatable).
- **CLI subcommand** `complexplorer gallery [--tag T | --preset ID…] -o DIR` — the thin
  wrapper for the slot `add-cli` deliberately left open.

## Determinism contract

- **Hard guarantee:** `index.json` and `card.json` are **byte-identical across repeated runs**
  of the same library version — no timestamps anywhere, sorted iteration,
  `json.dump(sort_keys=True, indent=2)`. (`complexplorer_version` makes the manifest change
  across *library* versions; that is expected and provenance-useful — the determinism test
  runs twice within one version.)
- **Best-effort:** `portrait.png` is reproducible within an environment (metadata stripped),
  but pixel bytes are **not** asserted (font/AA/matplotlib-version variance). The contract
  degrades gracefully to "manifest stable, pixels not guaranteed."

## Non-goals

- No STL or 3D relief in v1 (those stay as per-preset `complexplorer stl` / `render`;
  optional `--stl`/`--relief` flags can come later).
- No docs/HTML page generation (no framework to feed; the manifest is the layer a site would
  consume).
- No changes to `plot_2d`/`core` — the gallery grabs the figure and saves it with
  deterministic settings itself.

## Impact

- New capability: `gallery` (library fn + manifest contract + CLI subcommand).
- New module `complexplorer/gallery.py`; one new CLI subcommand in `cli/main.py`.
- Reuses: `core.presets` (`catalog`, `domain_from_spec`, `cmap_from_spec`,
  `FunctionPreset.to_dict`), `plotting.matplotlib.plot_2d`.
- Supersedes `examples/generate_gallery.py` (can be retired later).
- Risk: low. The one empirical unknown — whether stripped-metadata PNGs are byte-reproducible
  — is spiked during implementation; the manifest guarantee does not depend on it.
