## 1. The typing contract

- [x] 1.1 Add `complexplorer/typing.py` with the `ComplexFunction` protocol (array in, array out,
  scalars accepted) and the `TypedDict`s for `domain_spec`, `cmap_spec`, `scaling_spec` and the
  singularity records.
- [x] 1.2 Replace `Callable[[complex], complex]` with `ComplexFunction` in `api.py` and anywhere
  else a user-supplied function is annotated.
- [x] 1.3 Apply the spec `TypedDict`s in `core/presets.py`, keeping the serialized shape identical.
- [x] 1.4 Give every public callable an explicit return type, starting with the three that have
  none: `quick_plot`, `setup_matplotlib_backend`, `ensure_interactive_plots`.
- [x] 1.5 Declare and document `TransferFunction.__call__` for scalar and array arguments.

## 2. The rename

- [x] 2.1 Rename `Presets` to `PlotPresets` in `api.py` and `__init__.py`, with return types on its
  three methods.
- [x] 2.2 Update every reference: `CLAUDE.md`, the C3 documentation pages (`docs/api/catalog.md`,
  `docs/api/index.md`, `docs/guide/presets-and-catalog.md`, `docs/function-presets.md`), and the
  baseline specs that name it.
- [x] 2.3 State the distinction in one sentence wherever both appear: PlotPresets configures a
  render; catalog supplies a function.

## 3. Validation and error messages

- [x] 3.1 Validate `phase_sectors` in `BasePhasePortrait`: a positive integer, or a
  `ValidationError` naming the value and what is accepted. This covers `Phase` and the nine
  perceptual families at once.
- [x] 3.2 Sweep the public surface for bare Python exceptions reachable from documented use, and
  convert them to `ComplexplorerError` subclasses.
- [x] 3.3 Audit the `ValidationError` messages: each names the offending value, the accepted
  values, and any 2.x replacement.

## 4. The typing gate

- [x] 4.1 Add `tests/typing/downstream.py`: a small program that uses complexplorer as a consumer
  does, covering a lambda, an array-annotated def, an unannotated def and a `TransferFunction`.
- [x] 4.2 Add a pyright configuration scoped to the public surface and that fixture.
- [x] 4.3 Add a CI job that fails on those, and reports the whole-package error count without
  failing on it.

## 5. Discoverability

- [x] 5.1 Add the API map page to the site: one entry point per workflow, with what it returns.
- [x] 5.2 Add a test that every name in `__all__` has a docstring.

## 6. Verification

- [x] 6.1 `pytest`, `ruff check`/`format`, `mkdocs build --strict`, `openspec validate --specs`,
  and `openspec validate finalize-public-api-contract`.
- [x] 6.2 Confirm the fixture fails when an annotation is wrong, by breaking one deliberately.
- [x] 6.3 Confirm on CI, then flip C5's status in `openspec/ROADMAP.md` and record the outcome in
  `openspec/REV3_CLOSEOUT.md`.

## 7. What the gate found

- [x] 7.1 **The old annotation rejected correct user code and the library's own objects.** With
  `Callable[[complex], complex]` restored, pyright rejects three lines of the downstream fixture:
  an array-annotated function, a `TransferFunction`, and `preset.func` from the catalog. That is
  the defect the protocol fixes, demonstrated rather than asserted.
- [x] 7.2 **`Phase(r_log_base=1)` rendered every pixel as NaN** and raised nothing. `log(x)/log(1)`
  divides by zero. Found while auditing for bare exceptions, fixed alongside `phase_sectors`, and
  covered by `tests/unit/core/test_colormap_validation.py`.
- [x] 7.3 **`r_linear_step <= 0` and `scale_radius <= 0`** were accepted too, producing degenerate
  but finite output. Validated with the same block.
- [x] 7.4 **The audit found no remaining bare Python exceptions** on the public surface: seventeen
  probes with invalid arguments each returned a `ComplexplorerError` subclass naming the problem.
- [x] 7.5 **C3's documentation test caught the rename drift immediately**, exactly as the design
  predicted: renaming `Presets` failed
  `test_the_reference_documents_nothing_invented` until every page was updated.
