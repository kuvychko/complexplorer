## Why

Everything needed to *use* 3.0 now exists. What is missing is everything needed to *ship* it:
the upgrade path, an honest changelog, a README that matches the library, and the repository
metadata a released project is expected to carry.

Three of these are wrong rather than merely absent, which is why this is the last change and not
an afterthought:

- **The changelog is factually wrong about the version users have.** It says 2.0.0 was "tagged in
  git but never published to PyPI". PyPI's own API says otherwise: 2.0.0 was uploaded on
  2025-10-19 and is the current latest release. Every person who runs `pip install complexplorer`
  today gets 2.0.0, so it is precisely the version the upgrade notes must be written against.
- **The README makes claims the project cannot support.** "15-30x faster 3D rendering" appears
  twice with no benchmark behind it; "cinema-quality", "the first library to offer direct STL
  export", "no supports needed" and "the ultimate visualization experience" are the same kind of
  claim. A release is the wrong moment to ship numbers nobody measured.
- **`CONTRIBUTING.md`, `CITATION.cff` and issue templates do not exist**, and `[project.urls]`
  lists only two links, so the PyPI page cannot point at the documentation site C3 published.

## What Changes

**A migration guide written for the version people actually have**
- Extend `docs/migration-3.0.md` into a full 2.0.0 to 3.0 guide with a short 1.x appendix: the
  three things to do first, then old-to-new tables for the matplotlib 3D entry points,
  `n_phi`/`n_theta`, `show` and the `*_preset` functions, the removed logging API, the exception
  mapping, the `HAS_*` flags, and `Presets` to `PlotPresets` from C5.
- Add the two entries this change creates: the retired backend helpers, and the `Rectangle.contains`
  correction.
- One complete before-and-after example, and a troubleshooting section.

**A changelog that is true**
- Correct the 2.0.0 narrative against the PyPI record: published 2025-10-19; 2.1 to 2.4 were
  internal milestones that were never released.
- Describe the upgrade from 2.0.0, since that is what users are on.
- Add compare links for `[Unreleased]`, `[3.0.0]` and the earlier versions, and leave the 3.0.0
  date to be set when the tag is cut.

**A README that matches the library**
- Lead with the C2 hero and one minimal 2D example, before anything else.
- A "what's new in 3.0" strip, and a link to the migration guide.
- Move the specialized material (advanced examples, the PyVista discussion, the physical workflow
  detail) to the documentation site, which now exists to hold it.
- Remove the unsupportable claims rather than restating them more cautiously: drop the "15-30x"
  number entirely, and the "first library", "cinema-quality" and "no supports needed" claims with
  it. What is true and specific stays: matplotlib's 3D engine is not built for this and PyVista is.
- Fewer emoji: 25 in 297 lines is noise rather than navigation.

**Two helpers leave the public surface**
- `setup_matplotlib_backend` and `ensure_interactive_plots` are environment plumbing, not
  visualization API. They are removed from `__all__` and documented as removed. They were exported
  by 2.0.0, so this is a real break for anyone who used them and it gets a migration entry with the
  matplotlib call to use instead.

**Repository metadata**
- `CONTRIBUTING.md`, carrying the release runbook and the local artifact gate currently parked in
  the closeout tracker.
- `CITATION.cff`, with the README's BibTeX derived from it rather than hard-coding `year = {2024}`.
- `[project.urls]`: Documentation, Source, Changelog and Release notes.
- `.github/ISSUE_TEMPLATE/`: a bug report and a visual/rendering report that asks for OS, Python,
  PyVista/VTK, GPU or display, and a minimal example — the things every rendering bug needs.

**Not changed**
- The Python 3.14 classifier stays off. The lane is green but non-blocking, and claiming support
  while the lane cannot fail the build would contradict the matrix requirement C4 established.

## Capabilities

### New Capabilities

_None._

### Modified Capabilities

- `packaging`: the distribution metadata declares the documentation, changelog and release-notes
  URLs, and the project carries citation metadata.
- `high-level-api`: the curated surface drops the two backend helpers.
- `docs`: the migration guide is a required page with stated contents, and published claims about
  the project must be supportable.

## Impact

- **Breaking:** `cp.setup_matplotlib_backend` and `cp.ensure_interactive_plots` leave `__all__`.
  Both were exported by the published 2.0.0, so the migration guide names the replacement.
- **Docs:** `docs/migration-3.0.md` rewritten; README restructured; `CONTRIBUTING.md` added and the
  closeout tracker's local-gate section moved into it.
- **Metadata:** `pyproject.toml` URLs, `CITATION.cff`, `.github/ISSUE_TEMPLATE/`.
- **Code:** `complexplorer/__init__.py` only, for the two removed exports.
- **After this change:** the release runbook itself — merge, tag, build, upload, deploy — which is
  documentation rather than an OpenSpec change.
