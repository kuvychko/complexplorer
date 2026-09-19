## Context

This is the last change before the tag. Everything it touches is read by someone deciding whether
to use the library, or by someone whose code just broke — so the standard is accuracy rather than
polish.

Two facts were checked rather than assumed, because the existing text gets them wrong:

- **PyPI's API reports 2.0.0 uploaded on 2025-10-19, and it is the current latest release.** The
  changelog says 2.0.0 was "tagged in git but never published to PyPI". Everyone running
  `pip install complexplorer` today has 2.0.0, which makes it the baseline the upgrade notes must
  be written against.
- **2.0.0 exported `setup_matplotlib_backend` and `ensure_interactive_plots`.** Retiring them is
  therefore a real break for real users, not a tidy-up of a name nobody had.

## Goals / Non-Goals

**Goals:**

- A reader upgrading from 2.0.0 can resolve any break by looking up the name they used.
- The README is honest and short enough that the interesting part is visible without scrolling.
- The repository carries what a released project is expected to carry.

**Non-Goals:**

- Benchmarking the 3D backend. The "15-30x" claim is removed, not replaced with a measured one;
  producing a defensible benchmark is a project of its own, and the release does not need it.
- Trusted publishing and TestPyPI dry runs, which the closeout defers to 3.1.
- The Python 3.14 classifier, which stays off while its lane is non-blocking.

## Decisions

### The claims are deleted, not hedged

"15-30x faster" has no benchmark in the repository, and a softened version ("substantially
faster") would be the same unmeasured claim in quieter language. What replaces it is the specific
statement that is true and checkable: matplotlib's 3D engine renders these surfaces poorly and
slowly because it is not a mesh renderer, and PyVista is. A reader can verify that by running
both, which is the point.

The same reasoning retires "the first library to offer direct STL export" (unverifiable, and the
kind of claim that ages badly), "cinema-quality" and "no supports needed" — the last of which the
C3 work already found to be unsupported by anything in the code.

### The migration guide is organised by symptom, not by subsystem

Someone reads it because something broke. The entry point is therefore the name they used, in a
table they can search, rather than a narrative of what changed in which release. The narrative
lives in the changelog, which is the right place for it.

### The README loses material rather than gaining it

C3 published a documentation site; the README's job changes accordingly. It shows what the library
produces, gives one example that works, says what is new, and points at the site. Everything it
currently carries that the site now covers better — the advanced example, the PyVista discussion,
the physical workflow detail — is removed rather than duplicated, because two copies of the same
material drift.

### `CONTRIBUTING.md` absorbs the runbook the closeout is holding

C4 parked the local artifact-gate instructions in `openspec/REV3_CLOSEOUT.md` explicitly "until
`CONTRIBUTING.md` exists". This change creates it, so that material moves, and the release runbook
joins it. The closeout tracker goes back to being a tracker.

## Risks / Trade-offs

- **Removing two exported names breaks real 2.0.0 users** → The replacement is two lines of
  matplotlib, given in the migration guide; they were wrappers over `matplotlib.use` and `plt.ion`.
- **A rewritten README can lose something users relied on** → Nothing is deleted without a
  destination: every section either survives, moves to a named documentation page, or is a claim
  being retired on purpose.
- **The changelog correction contradicts what the repository has said for months** → It is checked
  against the package index, and the correction is stated plainly rather than quietly edited.
- **Removing claims could read as under-selling the project** → The hero image, the gallery and the
  physical ornaments do the selling, and they are evidence rather than adjectives.

## Open Questions

- Whether `CITATION.cff` should carry a DOI. It cannot until the release is archived somewhere that
  issues one; the file is written so a DOI can be added without restructuring it.
