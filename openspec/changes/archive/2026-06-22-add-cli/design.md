# Design — CLI

## Context

The CLI is ~90% glue over machinery that already exists:

- `quick_plot(func, domain, mode="2d"|"3d"|"riemann", **kwargs)` already dispatches the
  render and forwards kwargs; `plot()` already takes `filename=`.
- `cp.catalog` resolves `preset:<id>` to a callable + recommended specs.
- `domain_from_spec` / `cmap_from_spec` build live objects from spec dicts.
- `build_relief` / `OrnamentGenerator` / `SurfaceMesh.save_stl` cover STL.

The only substantive new code is the **expression evaluator** and the arg tree.

## Pipeline

```
   parse args
      │
   resolve <func> ──┬── "preset:id"  → cp.catalog.get(id): func + (domain/cmap/scaling
      │             │                   defaults unless overridden by flags)
      │             └── "expression"  → core.expression.evaluate (asteval)
   resolve domain ── "annulus:0.2:3" → spec dict → domain_from_spec   (reused)
   resolve cmap   ── "phase:6"        → spec dict → cmap_from_spec     (reused)
      │
   dispatch ──┬── render → quick_plot(func, domain, mode, filename=out, **opts)
              ├── stl    → build_relief(sample_sphere(func,…)) → SurfaceMesh.save_stl
              │            (or OrnamentGenerator for parity with existing defaults)
              └── list   → catalog.list() / filter(tag)
```

## Decisions

### D1 — Expression evaluator: AST grammar gate + asteval, in `core/expression.py`

`evaluate(expression: str, z: np.ndarray) -> np.ndarray` enforces a tight, portable math
grammar, then evaluates it safely. Curated symbol table:

```
names:   z
funcs:   sin cos tan  sinh cosh tanh  asin acos atan  exp log log10 sqrt  abs
         conj real imag angle  power
consts:  pi  e  j (= 1j)  i (= 1j)
ops:     + - * / ** unary-  parentheses
```

**Why two layers (empirically determined):**

- `asteval.Interpreter(minimal=True)` with *only* the curated symtable provides the safe
  evaluation: `minimal` blocks comprehensions/lambdas; dunder attribute access and the
  format-string escape are blocked; unknown names error. Verified: no code-execution path.
- **But asteval still allows non-dunder attribute access and string methods** (`z.real`,
  `'a'.upper()`, `z.flatten().tolist()`). That is safe but **not portable math** (Godot
  can't mirror it). So a **thin AST pre-check** (parse with stdlib `ast`; reject `Attribute`,
  string `Constant`s, and any node outside `Expression/BinOp/UnaryOp/Call(Name in
  whitelist)/Name/numeric Constant`) runs first and enforces the pure grammar. asteval still
  performs the actual evaluation — the gate is *validation only*, not a re-implementation.

**Error model (important):** asteval does **not** raise — it returns `None` and records
problems in `aev.error` (each entry's `.get_error()` is `(exc_name, message)`). The wrapper
checks `aev.error` after evaluation and raises `ValidationError`. The AST gate raises
`ValidationError` directly for off-grammar / malformed input.

PyVista-free; importable in the 2D/core path. (A convenience `compile_expression(expr) ->
Callable[[ndarray], ndarray]` returns a closure for reuse by `quick_plot`/`stl`.)

### D2 — asteval is a core dependency

The evaluator lives in `core/` and underpins both the CLI and (later) string input to the
high-level API, so asteval is added to `[project.dependencies]`. It is tiny and pure-python;
gating it behind an extra would gate `core/expression.py` and is not worth the friction.

### D3 — `<func>` resolution and spec defaults

`preset:<id>` resolves via `cp.catalog`; the preset's `domain_spec`/`cmap_spec`/`scaling_spec`
become **defaults** that explicit flags override. A bare string is an expression. The
`preset:` prefix is the disambiguator (an expression never starts with `preset:`).

### D4 — Shorthand → spec → existing factory

CLI shorthands are terse front-ends for the catalog's spec dicts, so there is exactly one
place that knows how to build a `Domain`/`Colormap`:

```
--domain rect:4:4        → {"type":"rectangle","re_length":4,"im_length":4}
--domain disk:2          → {"type":"disk","radius":2}
--domain annulus:0.2:3   → {"type":"annulus","inner_radius":0.2,"outer_radius":3}
--cmap   phase:6         → {"type":"Phase","n_phi":6}
                         → domain_from_spec / cmap_from_spec
```

### D5 — argparse, headless, graceful degradation

`argparse` (stdlib) with subparsers; `main(argv=None)` returns an exit code. Rendering is
headless: 2D via matplotlib `Agg` + `filename`; 3D/relief via the pyvista functions with
**`interactive=False`** (so no window opens) + `filename`, or `SurfaceMesh.screenshot`
(off-screen). Note `quick_plot` defaults the pyvista paths to `interactive=True`, so the CLI
must pass `interactive=False` explicitly. `--mode 2d` and `list` work without PyVista;
`3d`/`riemann`/`stl` check `HAS_PYVISTA` and exit non-zero with a clear message if absent.

## One grammar, cross-checked

The evaluator's grammar is the preset-expression grammar. A test asserts, for every catalog
preset, that `evaluate(p.expression, z) ≈ p.func(z)` on a sample grid (away from
singularities) — keeping the registry's callable and expression honest, and proving the
grammar covers the curated content.

## Open questions (proposal-level)

- Default `render` mode when a preset declares a "natural" mode (lean: `2d` default; presets
  may later carry a `recommended_mode` tag).
- Whether `stl` routes through `build_relief` directly or keeps `OrnamentGenerator` for
  byte-parity with current STL defaults (lean: `OrnamentGenerator`, already kernel-backed).
- Exact `--cmap`/`--scaling` shorthand grammar beyond the common cases.

## Risks

| Risk | Mitigation |
|---|---|
| asteval allows non-math grammar (attribute access, string methods) | AST grammar gate rejects attribute access / string literals / off-math nodes; tests assert `z.real`, `'a'.upper()`, `__import__`, comprehensions all raise `ValidationError` |
| asteval silently returns None on bad input | Wrapper inspects `aev.error` and raises `ValidationError` (verified behavior; asteval has no `raise_errors` call arg in 1.x) |
| CLI 3D rendering opens a window / needs a display in CI | `interactive=False` headless + the existing PyVista CI lane (xvfb) |
| Expression grammar doesn't cover a preset | The catalog drift test fails loudly; extend the symtable |
| Scope creep into gallery / api string-input | Hard non-goals; gallery + api wiring are separate |
