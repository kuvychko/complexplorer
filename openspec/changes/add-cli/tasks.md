# Tasks — add-cli

## 1. Expression evaluator (core/expression.py — PyVista-free)
- [x] 1.1 Add `asteval` to `[project.dependencies]` in `pyproject.toml` (no runtime deps).
- [x] 1.2 AST grammar gate: parse with stdlib `ast` and reject any node outside the
      whitelist (`Expression`, `BinOp`, `UnaryOp` with `+ - **` etc., `Call` whose target is
      a `Name` in the curated function set, `Name` = `z`/constants, numeric `Constant`).
      Reject `Attribute`, string `Constant`, comprehensions, lambdas, assignments → raise
      `ValidationError`.
- [x] 1.3 Evaluate via `asteval.Interpreter(symtable=<curated>, minimal=True, no_print=True)`.
      Symtable: `z`; numpy funcs `sin cos tan sinh cosh tanh asin acos atan exp log log10
      sqrt abs conj real imag angle power`; constants `pi e j i` (complex-valued). After
      eval, inspect `aev.error`; if non-empty, raise `ValidationError` with the message
      (asteval does NOT raise on its own).
- [x] 1.4 Add `compile_expression(expression) -> Callable[[ndarray], ndarray]` (a closure
      reusable by the CLI / future API string input).
- [x] 1.5 Unit tests: evaluation matches numpy; complex functions (`sqrt`, `exp(1/z)`,
      `z**(1/3)`); off-grammar raises (`z.real`, `z.__class__`, `'a'.upper()`,
      `[i for i in z]`, `__import__('os')`, `foo(z)`); malformed raises; module PyVista-free.

## 2. Catalog cross-check
- [x] 2.1 Drift test: for every `cp.catalog` preset, `evaluate(p.expression, z) ≈ p.func(z)`
      on a sample grid away from singularities (proves the grammar covers the curated set
      and the two function sources agree).

## 3. CLI package (complexplorer/cli/, argparse)
- [x] 3.1 `cli/main.py` with `main(argv=None) -> int`, argparse subparsers for `render`,
      `stl`, `list`. Add `[project.scripts] complexplorer = "complexplorer.cli.main:main"`.
- [x] 3.2 `<func>` resolution: `preset:<id>` → `cp.catalog` (recommended specs as defaults);
      else expression → `compile_expression`.
- [x] 3.3 Shorthand parsers: `--domain rect:4:4 | disk:2 | annulus:0.2:3` and `--cmap
      phase:6` → spec dicts → `domain_from_spec` / `cmap_from_spec`.
- [x] 3.4 `render`: dispatch to `quick_plot(func, domain, mode=2d|3d|riemann, filename=out,
      **opts)`; support `--resolution`, `--cmap`, `--scaling`, `--output`, `--show`. Pass
      `interactive=False` to the pyvista modes for headless file output. NOTE: `quick_plot`'s
      3D path defaults to the deprecated matplotlib renderer (and leaks `backend`), so the
      CLI dispatches DIRECTLY to `plot_landscape_pv`/`riemann_pv` (per the backend policy).
- [x] 3.5 `stl`: resolve func → `OrnamentGenerator` (kernel-backed) → `save_stl`; support
      `--size-mm`, `--resolution`, `--scaling`, `--output`.
- [x] 3.6 `list [--tag TAG]`: print ids (and titles/tags) from `cp.catalog`.
- [x] 3.7 Graceful degradation: `render --mode 2d` / `list` work without PyVista;
      `3d`/`riemann`/`stl` check `HAS_PYVISTA` and exit non-zero with a clear message.

## 4. Tests
- [x] 4.1 CLI tests via `main([...])`: `list` (no PyVista needed), `render` 2D to a temp PNG,
      `render preset:…`, expression + domain shorthand, and the clear PyVista-absent error
      for `stl` (simulate by patching `HAS_PYVISTA`).
- [x] 4.2 PyVista-gated tests: `render --mode riemann` and `stl` to temp files produce
      non-empty output.

## 5. Docs & close out
- [x] 5.1 Docs: a "Command-line interface" page with the command surface, the `preset:`/
      expression argument, and the shorthand grammar.
- [x] 5.2 Run `pytest tests/` green; `ruff` clean; `openspec validate --specs`.
- [x] 5.3 Update `openspec/ROADMAP.md` STATUS for this change.
