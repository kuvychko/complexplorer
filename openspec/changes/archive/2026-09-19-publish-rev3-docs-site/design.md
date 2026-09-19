## Context

The MkDocs Material scaffold was recovered from v2.0.0 in C1 and deliberately left inert: its nav
lists fourteen 2.0 pages that do not exist here, and its deploy workflow has the push trigger
removed so it cannot overwrite the live 2.0 site. This change makes it real.

Two structural facts drive most of the design. First, the material a reader most needs — the
gallery images from C2 and the example notebooks — lives **outside** `docs/`, in `examples/`, and
MkDocs only serves what is inside `docs_dir`. Second, the site must not deploy until 3.0 is tagged,
because the published 2.0 site is what today's users depend on.

## Goals / Non-Goals

**Goals:**
- A site that builds from this tree under `--strict`, with every nav entry present.
- An API reference generated from docstrings, covering the 47 names in `complexplorer.__all__` and
  the 5 in `cp.ee`, so it cannot drift.
- The C2 gallery visible on the site without committing a second copy of every image.
- A deploy that is impossible to trigger by accident before the release tag.

**Non-Goals:**
- Release copy. The migration guide's content, the README rewrite and the claim softening in
  `docs/pyvista_usage_guide.md` belong to C6. This change builds the shelf.
- Versioned documentation (`mike`, a version switcher). 3.0 replaces 2.0 at one URL. Revisit when
  there is a 3.1 worth keeping separate.
- Executing notebooks at build time. The Notebooks workflow already runs them; the site renders
  stored outputs.

## Decisions

### Assets outside `docs_dir` are added as MkDocs `File` objects, not copied into `docs/`

The generated gallery page links `../../examples/gallery/…` in 27 places. Three options:

1. **Commit a copy under `docs/`.** Rejected: it duplicates ~7 MB of images in git and creates two
   sources of truth that drift the moment the gallery is regenerated.
2. **Copy the tree into `site/` after the build** (`on_post_build`). Rejected: MkDocs never sees
   the files, so `--strict` flags every reference as a broken link. The gate we are adding would
   have to be disabled to let the site build.
3. **Add them to the file set during `on_files`** — chosen. A hook constructs `File` objects whose
   source is `examples/gallery/` and whose destination is `examples/gallery/`, so MkDocs treats
   them as part of the site. Strict mode is satisfied because the files genuinely exist in the
   build, and nothing is written into the working tree.

The existing relative links then resolve unchanged: with directory URLs, the gallery page is served
at `/gallery/gallery.generated/`, so `../../examples/gallery/x.png` resolves to
`/examples/gallery/x.png` — exactly where the hook puts it. The repository layout and the site
layout happen to agree, so **the generated page needs no rewriting**, and `examples/showcase.py`
stays the single producer of that page.

The same hook adds `examples/notebooks/*.ipynb` for `mkdocs-jupyter` (`execute: false`).

### The API reference is `mkdocstrings`, organized by workflow rather than by module

A single page of 47 entries is a wall; one page per module leaks internal layout the flat public
API deliberately hides. The reference is therefore grouped the way the library is used — domains,
colormaps, plotting, Riemann, export, engineering mode — with each group's members listed
explicitly. A test asserts every `__all__` name appears in some reference page, so grouping by hand
cannot silently drop a name. (C5 adds the companion check that every name has a docstring.)

### The version banner comes from the installed package, not from git

A hook reads `complexplorer.__version__` and asks git whether `HEAD` is exactly a `v*` tag. The
result goes into `config.extra`, and a small `main.html` override renders it as Material's
announcement bar: the version on a tagged build, and "development build" otherwise. This keeps the
claim honest on every page without a version-switcher plugin, and it fails loudly in CI if the
package cannot be imported — which would mean the docs environment does not match the code.

### Link checking is split: internal on every push, external on a schedule

`--strict` already catches internal breakage, and it is fast and deterministic. External links
depend on other people's servers, so checking them per push makes the gate flaky and trains people
to ignore it. They are checked on the weekly schedule instead, alongside the other scheduled runs.

### Internal notes move rather than being deleted

`phase_7_physical_texture_plan.md` and `composite_domain_viewing_window_fix.md` are 515 lines of
genuine design history. They move to `docs/internal/`, which `exclude_docs` keeps out of the build:
the history stays in the repository for anyone working on that code, and stops being presented to
users as documentation.

## Risks / Trade-offs

- **The file hook is a custom extension point** → It is ~30 lines against a documented MkDocs API
  (`on_files`), and the strict build is itself the test: if the hook stops adding the assets, every
  gallery reference breaks and CI fails.
- **Notebook rendering makes the site heavy** → Stored outputs only, no execution. If page weight
  becomes a problem, the application notebooks link out to GitHub and only the four core notebooks
  stay embedded.
- **`mkdocs build --strict` adds a CI job that can block on a typo in prose** → That is the point;
  a broken link is a defect. The failure names the file and line.
- **Grouping the API reference by hand can omit a name** → The coverage test makes omission a
  failure rather than an absence.
- **The deploy workflow is the one thing that can damage something live** → Its only triggers are a
  `v3.*` tag and manual dispatch; a branch push builds and checks but cannot publish.

## Migration Plan

The site replaces the 2.0 site at the same URL when `v3.0.0` is tagged. Until then every push
builds and checks it, and nothing is published. Rollback is a redeploy of the previous tag, since
`gh-pages` keeps the built site's history.

## Open Questions

- Whether the four application notebooks are embedded or linked; settled by the page weight the
  first full build reports.
- Whether `docs/pyvista_usage_guide.md` survives as its own page or is folded into the 3D page. Its
  claims are C6's to soften either way.
