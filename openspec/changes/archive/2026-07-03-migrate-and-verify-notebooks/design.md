## Context

The final milestone of `migrate-examples-and-docs`. After M1 (structure) and M2 (gallery), the
four tutorials under `examples/notebooks/` remain on the pre-3.0 surface:

```
 getting_started.ipynb    ✗ HAS_PYVISTA try/except + `if HAS_PYVISTA:` guards;
                            2× cp.plot_landscape(..., ax=ax)   (mpl-3D, ax= subplot)
 advanced_features.ipynb  ✗ 1× cp.plot_landscape(..., ax=ax)
 api_cookbook.ipynb       ✓ clean symbols (all _pv)
 stl_export_demo.ipynb    ✓ clean symbols
```

Careful review of the actual cells reframed the breakage. The `cp.plot_landscape` calls are **not**
standalone cells to swap to `_pv`:
- `getting_started` cell 18 and `advanced_features` cell 16 **time `plot_landscape_pv` against
  `cp.plot_landscape`** to claim "PyVista is Nx faster." With mpl-3D removed there is nothing to
  compare — these comparison cells are dead and must be deleted, not rewritten.
- `getting_started` cell 14 is `if HAS_PYVISTA: …_pv… else: …plot_landscape(ax=ax)…` — the `else`
  is a dead mpl fallback.
- `getting_started` cell 16 is an "inline `notebook=True` (low quality) vs external `notebook=False`
  (high quality)" demo — obsolete pedagogy that also fights the static backend.

Two spikes (run during exploration/review):
- **Spike 1 (execution):** a notebook setting `pv.set_jupyter_backend('static')` then calling
  `cp.plot`, `cp.plot_landscape_pv`, `cp.riemann_surface_pv` executed cleanly under `nbconvert`
  and embedded `image/png` in every render cell. Static backend is the right strategy.
- **Spike 2 (the kwarg trap):** a `_pv` call with `notebook=False` **opened an external window**
  (it produced only a "static backend, trame recommended" stderr warning and *no* inline image;
  on a machine with a display it pops an actual window). `notebook=True`/default embeds the image.
  Since the existing notebooks pass `notebook=`/`show=`/`off_screen=` in ~15 cells across all four,
  those kwargs must be **stripped** so the static backend governs and images embed.

The committed notebooks also embed stale mpl-3D output images and cover none of the 3.0 headline
features.

## Goals / Non-Goals

**Goals:**
- All four notebooks execute top-to-bottom on the 3.0 surface, verifiably and repeatably.
- Reproducible, committed, readable output (static images render on GitHub/nbviewer).
- Cover the 3.0 headline additions: Riemann surfaces and the preset registry/gallery.
- A one-command verification harness (nbmake), opt-in and local.

**Non-Goals:**
- No library code/API changes. No gallery changes (M2). No CI gating of notebook execution.
- No interactive/trame backend in the committed notebooks (static only; a note points to the
  interactive path). No new notebooks — the four existing ones are modernized in place.

## Decisions

**D1 — Static PyVista backend in every notebook (incl. the two "clean" ones).** Each notebook's
setup cell calls `pv.set_jupyter_backend('static')`. This is needed by *all four*, not just the
two with removed symbols: without it, the `_pv` calls in `api_cookbook`/`stl_export_demo` would
try to open interactive windows and hang/fail under `nbconvert`. Static rendering executes
headlessly and embeds images. *Alternative:* set the backend externally (env var / conftest)
during the verify run only — rejected; the committed notebook would not be self-contained, and a
reader running it interactively could hit the same hang. The setup cell is paired with a markdown
note documenting `notebook=False` / the terminal scripts for higher quality, consistent with the
existing CLAUDE.md/README guidance.

**D2 — Keep executed output, committed (user decision).** Re-execute each notebook and commit the
fresh static-image output, replacing the stale mpl-3D images. GitHub/nbviewer then render the
tutorials richly, and the committed file is its own proof-of-execution. This matches M2's
"committed but regenerable" stance. The cost — ~10–15 MB of embedded images and noisier base64
diffs on re-run — is accepted. *Alternative:* strip output — rejected by the user (loses
GitHub-rendered visuals), though the gallery already carries the polished imagery.

**D3 — Producer / verifier split.** Two commands, two jobs (mirrors M2's showcase-vs-guard):
- *Regenerate committed output:* `jupyter nbconvert --to notebook --execute --inplace
  examples/notebooks/*.ipynb` (the notebook analogue of `python examples/showcase.py`).
- *Verify (DoD gate):* `pytest --nbmake examples/notebooks/` — re-executes and asserts no cell
  error. nbmake does not write output back, so the two steps stay cleanly separated.

**D4 — nbmake is opt-in, not in the default suite or CI.** The roadmap defers CI-gating; the
notebooks are PyVista-heavy (minutes to run) and off-screen screenshots crash on headless CI. The
opt-in is *structural*, not just convention: `pytest.ini` already sets `testpaths = tests` and
`python_files = test_*.py`, so the default run never reaches `examples/`; and nbmake only collects
notebooks when the explicit `--nbmake` flag is passed. So `pytest --nbmake examples/notebooks/` is
the only way notebooks execute. *Alternative:* wire nbmake into CI — rejected for now (matches the
established local-only stance; revisitable once a headless-safe render path exists).

**D5 — `[examples]` extra for notebook tooling.** `nbmake`, `nbconvert`, `ipykernel` go in a new
`[project.optional-dependencies] examples` group (and `[dev]` includes it). Keeps the notebook
verification tooling explicit and installable, without bloating the core/runtime deps.

**D6 — Content placement.** Riemann-surface section → `advanced_features` (its 3D/advanced home).
Registry recipe → `api_cookbook` (the "recipes" notebook). Backend-policy note → `getting_started`
(first contact). No fifth notebook — the additions fold into the existing four.

**D7 — Excise the obsolete matplotlib-vs-PyVista narrative (review finding).** The notebooks were
written when matplotlib had a 3D backend; they carry "PyVista vs matplotlib" *speed* (timing) and
*quality* (inline-vs-external) comparisons whose premise is gone at 3.0. Rather than contrive a
PyVista-only "comparison," the dead cells are removed: the two timing cells (`getting_started` 18,
`advanced_features` 16) are deleted (optionally replaced by a one-line "PyVista is the sole 3D
backend" note), `getting_started` cell 14's mpl `else` fallback is dropped, and the inline-vs-
external quality demo (cell 16) becomes a short note (static backend handles inline; the terminal
scripts handle interactive). This is content rework, not symbol-swapping. *Alternative:* keep the
cells, swapping the mpl side for a second `_pv` call — rejected as meaningless (comparing PyVista
to itself).

**D8 — Strip `notebook=`/`show=`/`off_screen=` from every `_pv` call (spike-grounded).** Spike 2
showed `notebook=False` opens an external window and embeds no inline image under the static
backend. The committed-output goal (D2) and headless execution both require the static backend to
govern rendering, so these kwargs are removed from all ~15 `_pv` cells across the four notebooks
(the call becomes e.g. `cp.plot_landscape_pv(domain, f, cmap=…, modulus_mode=…)`), letting the
static backend embed the image. This is a whole-set normalization, broader than the two notebooks
with removed symbols. *Alternative:* set `notebook=True` explicitly everywhere — rejected; dropping
the kwarg is cleaner and the static backend already implies inline rendering.

## Risks / Trade-offs

- **[Re-execution is non-deterministic / noisy diffs]** → Accepted (D2). The notebooks are tutorials,
  not byte-stable contracts; the deterministic artifacts live in `index.json`/`showcase.json`.
- **[nbmake run is slow (minutes)]** → Mitigated by D4 (opt-in, local). Not in the default suite,
  so day-to-day `pytest` stays fast.
- **[Static backend removes interactivity for notebook readers]** → Mitigated by the markdown note
  pointing to `notebook=False` and the terminal scripts; quality-seekers already use those.
- **[Headless environments can't run nbmake (screenshot crash)]** → Accepted; the harness is a
  local gate by design. The static-symbol scan (no removed symbols) remains a cheap CI-safe check.
- **[Repo size growth]** → Accepted, user-acknowledged; ~10–15 MB atop M2's 12 MB gallery.

## Open Questions

- Exact opt-in mechanism for nbmake (a dedicated `testpaths` exclusion vs a custom marker vs a
  separate `pytest --nbmake <dir>` invocation) — an implementation detail settled when wiring
  `pyproject`/`pytest.ini`; the contract is only "not in the default run."
- Whether to add a tiny `examples/notebooks/README.md` documenting the regenerate/verify commands,
  or fold that into the top-level `examples/README.md` — minor; default to a short note in the
  examples README.
