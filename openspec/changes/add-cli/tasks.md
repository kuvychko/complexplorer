# Tasks — add-cli

## 1. Expression evaluator (core/expression.py — PyVista-free)
- [ ] 1.1 Add `asteval` to `[project.dependencies]` in `pyproject.toml` (no runtime deps).
- [ ] 1.2 AST grammar gate: parse with stdlib `ast` and reject any node outside the
      whitelist (`Expression`, `BinOp`, `UnaryOp` with `+ - **` etc., `Call` whose target is
      a `Name` in the curated function set, `Name` = `z`/constants, numeric `Constant`).
      Reject `Attribute`, string `Constant`, comprehensions, lambdas, assignments → raise
      `ValidationError`.
- [ ] 1.3 Evaluate via `asteval.Interpreter(symtable=<curated>, minimal=True, no_print=True)`.
      Symtable: `z`; numpy funcs `sin cos tan sinh cosh tanh asin acos atan exp log log10
      sqrt abs conj real imag angle power`; constants `pi e j i` (complex-valued). After
      eval, inspect `aev.error`; if non-empty, raise `ValidationError` with the message
      (asteval does NOT raise on its own).
- [ ] 1.4 Add `compile_expression(expression) -> Callable[[ndarray], ndarray]` (a closure
      reusable by the CLI / future API string input).
- [ ] 1.5 Unit tests: evaluation matches numpy; complex functions (`sqrt`, `exp(1/z)`,
      `z**(1/3)`); off-grammar raises (`z.real`, `z.__class__`, `'a'.upper()`,
      `[i for i in z]`, `__import__('os')`, `foo(z)`); malformed raises; module PyVista-free.

## 2. Catalog cross-check
- [ ] 2.1 Drift test: for every `cp.catalog` preset, `evaluate(p.expression, z) ≈ p.func(z)`
      on a sample grid away from singularities (proves the grammar covers the curated set
      and the two function sources agree).

## 3. CLI package (complexplorer/cli/, argparse)
- [ ] 3.1 `cli/main.py` with `main(argv=None) -> int`, argparse subparsers for `render`,
      `stl`, `list`. Add `[project.scripts] complexplorer = "complexplorer.cli.main:main"`.
- [ ] 3.2 `<func>` resolution: `preset:<id>` → `cp.catalog` (recommended specs as defaults);
      else expression → `compile_expression`.
- [ ] 3.3 Shorthand parsers: `--domain rect:4:4 | disk:2 | annulus:0.2:3` and `--cmap
      phase:6` → spec dicts → `domain_from_spec` / `cmap_from_spec`.
- [ ] 3.4 `render`: dispatch to `quick_plot(func, domain, mode=2d|3d|riemann, filename=out,
      **opts)`; support `--resolution`, `--cmap`, `--scaling`, `--output`, `--show`. Pass
      `interactive=False` to the pyvista modes for headless file output (quick_plot defaults
      it to True); 2D saves via matplotlib `Agg` + `filename`. `--show` opens a window.
- [ ] 3.5 `stl`: resolve func → `OrnamentGenerator` (kernel-backed) → `save_stl`; support
      `--size-mm`, `--resolution`, `--scaling`, `--output`.
- [ ] 3.6 `list [--tag TAG]`: print ids (and titles/tags) from `cp.catalog`.
- [ ] 3.7 Graceful degradation: `render --mode 2d` / `list` work without PyVista;
      `3d`/`riemann`/`stl` check `HAS_PYVISTA` and exit non-zero with a clear message.

## 4. Tests
- [ ] 4.1 CLI tests via `main([...])`: `list` (no PyVista needed), `render` 2D to a temp PNG,
      `render preset:…`, expression + domain shorthand, and the clear PyVista-absent error
      for `stl` (simulate by patching `HAS_PYVISTA`).
- [ ] 4.2 PyVista-gated tests: `render --mode riemann` and `stl` to temp files produce
      non-empty output.

## 5. Docs & close out
- [ ] 5.1 Docs: a "Command-line interface" page with the command surface, the `preset:`/
      expression argument, and the shorthand grammar.
- [ ] 5.2 Run `pytest tests/` green; `ruff` clean; `openspec validate --specs`.
- [ ] 5.3 Update `openspec/ROADMAP.md` STATUS for this change.
