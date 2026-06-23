# Add CLI

## Why

Phase 2 (v2.3) makes Complexplorer usable as a tool, not just a library. With the surface
kernel (Phase 1) and the preset registry (`cp.catalog`) in place, a command-line interface
is now almost entirely **glue**: resolve a function, resolve a domain, dispatch to the
existing rendering/export machinery, write a file. The one genuinely new piece is turning a
user's **expression string** into a callable — which the catalog deliberately avoided.

That evaluator is also the missing portable contract: the same grammar the CLI evaluates is
what preset `expression` strings conform to and what Godot reimplements from (the games
boundary). Putting it in `core/` makes it reusable and lets it cross-check the catalog.

## What changes

- **`core/expression.py`** — `evaluate(expression, z)` (and a `compile`-style helper) that
  turns a complex-function expression into values over a **tight, portable math grammar**:
  `z`, numeric/imaginary literals, arithmetic (`+ - * / **`, unary minus, parens), and
  calls to a curated set of numpy functions (`sin/cos/tan/exp/log/sqrt/abs/conj/real/imag/…`)
  + constants (`pi`, `e`, `j`). Implemented as a thin **AST grammar gate** (rejects attribute
  access, string literals, comprehensions, and any non-math node) **in front of** an
  `asteval.Interpreter(minimal=True)` with the curated symbol table. asteval does the safe
  evaluation (its `minimal` mode + dunder/format mitigations block code-execution escapes);
  the AST gate keeps the grammar pure and Godot-portable. PyVista-free. asteval does not
  raise — the wrapper inspects `aev.error` and raises `ValidationError`.
- **`asteval` added to core dependencies** (tiny; **no runtime dependencies of its own**).
- **`complexplorer/cli/`** — an `argparse` CLI exposed via `[project.scripts]`
  (`complexplorer = "complexplorer.cli.main:main"`) with three commands:
  - `render <func> [--mode 2d|3d|riemann] [--domain …] [--cmap …] [--scaling …]
    [--resolution N] [--output FILE] [--show]`
  - `stl <func> [--size-mm …] [--resolution N] [--scaling …] [--output FILE]`
  - `list [--tag TAG]`
  - `<func>` is either `preset:<id>` (→ `cp.catalog`, using its recommended specs as
    defaults) or an expression string (→ `core/expression.py`).
- **Shorthand → spec → factory reuse:** `--domain annulus:0.2:3` →
  `{"type": "annulus", "inner_radius": 0.2, "outer_radius": 3}` → `domain_from_spec` (the
  catalog factory). Same idea for `--cmap`.
- **Dispatch reuses existing machinery:** `render` → `quick_plot(func, domain, mode=…,
  filename=…)`; `stl` → `build_relief` / `OrnamentGenerator` → `save_stl`; `list` →
  `catalog`.
- **Catalog drift test:** assert `evaluate(p.expression) ≈ p.func` for every preset — a
  cheap cross-check that the registry's two function sources agree.

## Non-goals

- **No gallery command** — batch generation is `add-gallery-generator` (a later CLI
  subcommand can invoke it).
- **No high-level-API string input.** `core/expression.py` *enables* `cp.show("z**2")`, but
  wiring strings into `quick_plot`/`show` is a small, separate follow-on (this change keeps
  the evaluator's only consumer the CLI + the drift test).
- **No expression-derived singularity detection** — the evaluator computes values, not
  answer keys (those stay hand-authored in the catalog).

## Impact

- New deps: `asteval` (core). New modules: `core/expression.py`, `complexplorer/cli/`.
  New entry point in `pyproject.toml` `[project.scripts]`.
- Reuses: `cp.catalog` + `domain_from_spec`/`cmap_from_spec`, `quick_plot`, `build_relief` /
  `OrnamentGenerator` / `SurfaceMesh.save_stl`.
- Affected specs: new `expression` and `cli` capabilities.
- Graceful degradation: `render --mode 2d` and `list` need no PyVista; `3d`/`riemann`/`stl`
  require it and error cleanly if absent (consistent with the backend policy).
- Risk: low–moderate. Main care: the asteval symbol table must be curated (not the full
  numpy surface) so the grammar is safe and portable; CLI rendering must run headless.
