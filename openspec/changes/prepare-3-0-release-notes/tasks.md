## 1. The migration guide

- [x] 1.1 Rewrite `docs/migration-3.0.md` for 2.0.0 to 3.0, opening with the three things to do
  first, and a 1.x appendix.
- [x] 1.2 Old-to-new tables: the matplotlib 3D entry points; `n_phi` and `n_theta`; `show` and the
  `*_preset` functions; the removed logging API; the exception mapping; the `HAS_*` flags;
  `Presets` to `PlotPresets`; and the two backend helpers retired in task 3.
- [x] 1.3 Record the `Rectangle.contains` correction.
- [x] 1.4 One complete before-and-after example, plus a troubleshooting section for the errors a
  2.0.0 user will actually hit.
- [x] 1.5 Check that every name removed or renamed since 2.0.0 appears in the guide, by diffing
  `git show v2.0.0:complexplorer/__init__.py` against the current `__all__`.

## 2. The changelog

- [x] 2.1 Correct the 2.0.0 narrative: published to PyPI on 2025-10-19 (verified against the
  package index), latest release until 3.0; 2.1 to 2.4 were internal milestones, never released.
- [x] 2.2 Write the 3.0.0 entry as an upgrade from 2.0.0, covering everything C1 through C6 changed
  on the public surface.
- [x] 2.3 Add compare links for `[Unreleased]`, `[3.0.0]` and the earlier versions; leave the
  3.0.0 date to be set when the tag is cut.

## 3. Retire the backend helpers

- [x] 3.1 Remove `setup_matplotlib_backend` and `ensure_interactive_plots` from
  `complexplorer/__init__.py`'s exports and `__all__`. The functions stay where they are for
  internal use.
- [x] 3.2 Remove them from the API reference and the API map, and give them a migration entry with
  the matplotlib calls to use instead.
- [x] 3.3 Check nothing in `examples/`, `tests/` or the notebooks imports them from the top level.

## 4. The README

- [x] 4.1 Restructure: hero, one minimal 2D example, "what's new in 3.0", then short sections that
  link to the documentation site.
- [x] 4.2 Remove the unsupportable claims -- the "15-30x" figure in both places, "cinema-quality",
  "the first library", "no supports needed", "the ultimate visualization experience" -- and replace
  the performance one with the specific, checkable statement about why the 3D backend is PyVista.
- [x] 4.3 Move the advanced example, the PyVista discussion and the physical-workflow detail to the
  documentation site, without duplicating them.
- [x] 4.4 Reduce the emoji (25 in 297 lines today) to the few that aid navigation.
- [x] 4.5 Link the migration guide, the documentation site and the changelog.
- [x] 4.6 Derive the BibTeX block from `CITATION.cff` instead of a hard-coded year.

## 5. Repository metadata

- [x] 5.1 Add `CONTRIBUTING.md`: setup, the checks, the OpenSpec workflow, notebooks, gallery
  regeneration, the artifact gate (moved out of `openspec/REV3_CLOSEOUT.md`), and the release
  runbook.
- [x] 5.2 Add `CITATION.cff` with the version, authors and repository, written so a DOI can be
  added later.
- [x] 5.3 Extend `[project.urls]` with Documentation, Source, Changelog and Release notes.
- [x] 5.4 Add `.github/ISSUE_TEMPLATE/`: a bug report, and a visual/rendering report asking for OS,
  Python, PyVista/VTK, GPU or display, and a minimal example.

## 6. Tests

- [x] 6.1 A test that the retired helpers are absent from `__all__` and that the migration guide
  names them.
- [x] 6.2 A test that the README and the package description carry no unsupported claim, checking
  for the specific phrases retired here so they cannot come back.
- [x] 6.3 A test that `CITATION.cff` and `complexplorer.__version__` agree.

## 7. Verification

- [x] 7.1 `pytest`, `ruff check`/`format`, `mkdocs build --strict`, `openspec validate --specs`,
  `openspec validate prepare-3-0-release-notes`, and the artifact gate (`uv build`, `twine check`,
  `scripts/check_distribution.py`).
- [x] 7.2 Render the README as PyPI will: `twine check` plus a look at the long description, with
  every image and link resolving by absolute URL.
- [x] 7.3 Run every code block in the migration guide and the README against the built wheel.
  **Three of five ran clean**: the 2D portrait with a legend, `plot_landscape_pv`, and the
  backend-helper replacement. The remaining two open interactive PyVista windows -- which is
  correct for the desktop user they are written for, and exactly what made them unrunnable here:
  they block, and neither `PYVISTA_OFF_SCREEN=true` nor `pv.OFF_SCREEN = True` suppressed the
  window on that path. They were skipped by decision rather than silently. They use the same
  four calls the verified snippets exercise, and the wheel smoke test renders 3D off-screen on
  every CI run. To run them without a window, patch `pv.Plotter.show` to a no-op in the harness.
  **Followed up immediately, not deferred:** `pv.OFF_SCREEN` not being honoured was a real
  headless bug, fixed by the `honour-pyvista-off-screen` change. With that fix in a rebuilt
  wheel, **both skipped snippets now run clean in 2-3 seconds with no window**, so all five
  blocks of this task are verified after all.
- [ ] 7.4 Confirm on CI, then flip C6's status in `openspec/ROADMAP.md` and complete the final
  closeout checklist in `openspec/REV3_CLOSEOUT.md`.
