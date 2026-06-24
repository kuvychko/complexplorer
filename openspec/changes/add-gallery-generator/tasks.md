# Tasks — add-gallery-generator

## 1. Spike: PNG metadata determinism  (DONE during review — see design.md "Spike result")
- [x] 1.1 Verified: with the figure-owning path + `metadata={"Software": None}` (Agg), two
      renders of `pole_flower_10` are byte-identical (no `Software`/`tIME`); `to_dict()` is
      JSON-stable and all-builtin. Within-env portraits are byte-stable; manifest is the
      cross-version guarantee.

## 2. Gallery module
- [ ] 2.1 New `complexplorer/gallery.py` (matplotlib-only, PyVista-free): `generate_gallery(
      selection=None, out_dir, *, dpi=150) -> dict`. Resolve selection (ids | tag | all),
      sort by id.
- [ ] 2.2 `_render_portrait`: `plot_2d(domain, func, cmap=…, filename=None)` → `ax.figure
      .savefig(path, dpi=dpi, metadata={…})` → `plt.close(fig)`. Instantiate domain/cmap via
      `domain_from_spec`/`cmap_from_spec`.
- [ ] 2.3 Write `<id>/card.json` = `to_dict()` + `{"files": {"portrait": "<id>/portrait.png"}}`
      and the top-level `index.json` wrapper `{schema_version, complexplorer_version,
      generator, presets:[…]}`, all via a `_write_json` helper using
      `json.dump(sort_keys=True, indent=2, ensure_ascii=False)` + trailing newline. Relative
      paths only; no timestamps.
- [ ] 2.4 Export `generate_gallery` from the package public API.

## 3. CLI subcommand
- [ ] 3.1 Add a `gallery` subcommand to `cli/main.py` (`--tag` | `--preset ID…`, `-o/--output`
      required) that calls `generate_gallery`; missing output → exit 2 (reuse the error path);
      a selection matching 0 presets → exit 2 with a "0 presets matched" message.

## 4. Tests
- [ ] 4.1 Determinism: run `generate_gallery` twice into two dirs; assert `index.json`, every
      `card.json`, and every `portrait.png` are byte-identical (within-env).
- [ ] 4.2 Structure: each selected preset has a non-empty `portrait.png` + a `card.json`;
      `index.json.presets` is id-sorted; `files.portrait` is relative and exists; record keys
      == `to_dict()` keys + `files`.
- [ ] 4.3 No timestamp/software tag in a written portrait PNG (per the spike outcome).
- [ ] 4.4 CLI: `gallery --tag <t> -o tmp` exits 0 and writes `index.json`; missing `-o` exits 2;
      an unmatched tag exits 2 with a "0 presets matched" message.

## 5. Close out
- [ ] 5.1 Run `pytest tests/` green; `ruff` clean; `openspec validate --specs`.
- [ ] 5.2 Update `openspec/ROADMAP.md` (add-gallery-generator status; Phase 2 complete).
