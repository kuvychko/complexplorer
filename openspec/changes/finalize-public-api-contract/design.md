## Context

C1 restored the 2.0 surface, C3 published documentation for it, and C4 proved the artifact
installs and runs. What is left is the contract itself: the names, the types, and the errors.

Three measurements shape this design, all taken before writing it:

- **pyright in basic mode reports 86 errors across the package.** The three most suspicious were
  checked individually and all three are false positives: `core/field.py` guards with a compound
  `if domain is None and z is None`, which pyright cannot narrow but which is correct;
  `mesh_repair.py` sets `n_boundary` and reads it under the same `verbose` condition; and
  `clean(tolerance=...)` is a real pyvista parameter that the stubs resolve on a union type. The
  remainder are dominated by `float | ndarray` unions -- the shape of numerical Python.
- **The validation gap is exactly one family.** `Chessboard`, `PolarChessboard` and `LogRings`
  all raise `ColormapError` for a bad spacing or sector count, and `Phase(v_base=1.5)` raises.
  But `phase_sectors` is validated nowhere in the phase-portrait base, so `0` reaches a division
  and `-1` and `2.5` are accepted.
- **Three public functions have no return annotation** (`quick_plot`,
  `setup_matplotlib_backend`, `ensure_interactive_plots`), and all three `Presets` methods return
  bare dicts.

## Goals / Non-Goals

**Goals:**

- Annotations that accept correct user code and describe what the library really calls.
- One name for render configuration that cannot be confused with the function registry.
- Every public error being the library's own, with a message that says what to do.
- A typing gate that fails only for real defects.

**Non-Goals:**

- Zero pyright errors inside the package. That is a different, larger job with a much worse
  cost-to-value ratio, and it is not what users see.
- Runtime type enforcement. The protocol is a static description, not a check.
- Changing what any function computes.

## Decisions

### `ComplexFunction` is a Protocol, not an alias

The annotation must accept a plain lambda, a `TransferFunction`, and a numpy ufunc, none of which
share a base class. A `Protocol` with `__call__` describes exactly the obligation -- takes an array
of complex values, returns an array of the same shape -- and is satisfied structurally, so no user
has to import or inherit anything. An alias such as `Callable[[np.ndarray], np.ndarray]` would be
shorter but would reject a function annotated for scalars-or-arrays, which is how most users write
them.

### The pyright gate covers the public surface and a downstream fixture

The fixture is the honest test: a small program that imports complexplorer, builds a domain, a
colormap and a function, calls the entry points, and is type-checked as a *consumer*. That is
precisely the experience `py.typed` promises, and it catches the failure mode this change exists
to fix -- an annotation that rejects correct user code -- which a gate over the library's own
internals would not.

Internal errors stay visible without being blocking: the job reports the whole-package count so a
regression is noticeable, and fails only on the public surface and the fixture.

### `phase_sectors` is validated in the base class

Every phase portrait inherits `BasePhasePortrait`, including the nine perceptual families ported
in C1, so validating there fixes all of them at once and makes the gap impossible to reintroduce
in a tenth. The check belongs at construction, where the user can see the cause, rather than at
render time in the middle of a batch.

### The rename is a rename, not a deprecation

`Presets` has never been published: 2.0.0 exposed `publication_preset()` and friends, and the name
exists only on this branch. A deprecation alias would preserve compatibility with nothing while
keeping the confusing name discoverable, so it is removed outright and recorded in the migration
guide by C6.

## Risks / Trade-offs

- **The rename touches the documentation C3 just published** -> It is mechanical, and C3's own test
  that every public name reaches the API reference will fail if a page is missed.
- **A Protocol can be over-tight and reject valid callables** -> The fixture exercises the forms
  users actually write (a lambda, a def with array annotations, an unannotated def and a
  `TransferFunction`) and is part of the gate.
- **Validating `phase_sectors` could break a caller relying on a float** -> It is accepted today
  only by accident, and it produces a meaningless sector count; the error names the fix.
- **The gate's narrow scope could be read as hiding problems** -> The whole-package count is
  printed on every run, and the reason it is not blocking is recorded here with the evidence.

## Open Questions

- Whether `setup_matplotlib_backend` and `ensure_interactive_plots` belong on the public surface at
  all. They are environment helpers rather than visualization API; C6 may retire them from
  `__all__` rather than this change typing them.
