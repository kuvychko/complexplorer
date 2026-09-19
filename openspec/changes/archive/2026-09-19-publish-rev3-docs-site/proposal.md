## Why

Complexplorer 3.0 has no documentation a user can read. The live site at
`kuvychko.github.io/complexplorer` documents **2.0**, which is a different library: it describes
matplotlib 3D entry points that no longer exist, `n_phi` rather than `phase_sectors`, and capability
flags that were deleted. `docs/` holds eleven Markdown files, two of which are internal planning
notes, and its `mkdocs.yml` — recovered from v2.0.0 in C1 — still lists a navigation of fourteen
pages, **none of which exist in this tree**. A reader arriving today either finds instructions for a
library they did not install, or a 404.

The closeout calls this the largest single gap before release (§2). It also blocks C6, which needs
stable documentation URLs to link from the README, the changelog and the GitHub release.

## What Changes

**A site that builds from this tree**
- Rewrite `mkdocs.yml`'s navigation around the rev3 page set and make `mkdocs build --strict` pass,
  so a broken internal link fails the build rather than shipping.
- Add `mkdocstrings[python]` for a generated API reference over the 47 names in
  `complexplorer.__all__` and the 5 in `cp.ee`, so the reference cannot drift from the code.
- Add a `docs` extra to `pyproject.toml` (mkdocs-material, mkdocstrings, mkdocs-jupyter), so the
  site builds from a declared dependency set rather than from whatever a machine happens to have.

**The page set the closeout asks for (§2)**
- Overview and visual tour, built on the C2 assets; install and first portrait; reading a phase
  portrait (including the legend and the CVD material from C1); domains and colormaps, including
  composite domains; 3D landscapes and the Riemann sphere/relief, covering headless use and a
  deterministic camera; Riemann surfaces; engineering mode; catalog versus plot presets; the CLI;
  the STL and physical workflow; the API reference; the migration guide; and contributing and
  release checks.

**Stale documentation corrected (§2.1)**
- `docs/cli.md` documents three subcommands; there are four — it never mentions `gallery` — and it
  still tells the reader that some commands "work without the 3D backend", which describes the
  optional-PyVista world that ended at 3.0.
- `docs/README.md` claims "330+ tests"; the suite collects 695.
- `docs/development/surface-kernel.md`, `examples/README.md`, and the module docstrings in
  `core/presets.py` and `gallery.py` are checked against the 3.0 surface.
- `docs/function-presets.md`'s Godot and `FunctionFamily` material moves to internal notes: it
  describes a consumer that does not exist yet.
- `docs/phase_7_physical_texture_plan.md` and `docs/composite_domain_viewing_window_fix.md` (515
  lines of planning history between them) move to `docs/internal/`, excluded from the navigation.

**The gallery, without duplicated binaries**
- The generated gallery page links `../../examples/gallery/…` in 27 places, which is outside
  `docs_dir` and therefore cannot be served. An MkDocs hook copies the gallery assets into the site
  at build time, so the images reach the site without a second copy of every PNG in git.

**Deploy that cannot overwrite the live 2.0 site (§4)**
- The deploy workflow triggers on a `v3.*` tag only. The 2.0 site stays up until 3.0 is published.
- The site displays the installed `__version__`, and labels an untagged build "dev", so a reader can
  tell which version they are reading.

## Capabilities

### New Capabilities

- `docs`: the documentation site — what it must contain, that it builds strictly from this tree,
  how the API reference is generated, how gallery assets reach the site, and when it deploys.

### Modified Capabilities

_None._ The `examples` capability already owns the generated gallery page and `showcase.json`; this
change consumes them rather than changing their contract. No library behavior changes.

## Impact

- **New:** `docs/**` page set, `docs/internal/` (excluded from the nav), an MkDocs hook for the
  gallery assets, and a `docs` extra in `pyproject.toml`.
- **Rewritten:** `mkdocs.yml` (nav, plugins, strict build) and `.github/workflows/docs.yml`
  (tag-triggered deploy).
- **CI:** `mkdocs build --strict` runs on every push; a small external-link check runs on a
  schedule, not per push, because it depends on other people's servers.
- **Moved:** two planning documents into `docs/internal/`.
- **Not here:** the migration guide's *content*, the README rewrite, and softening the "15-30x" and
  "cinema-quality" claims in `docs/pyvista_usage_guide.md` — all of that is C6
  (`prepare-3-0-release-notes`). This change builds the shelf; C6 writes the release copy that sits
  on it. Where a page needs migration content before C6 lands, it links the existing
  `docs/migration-3.0.md` stub.
