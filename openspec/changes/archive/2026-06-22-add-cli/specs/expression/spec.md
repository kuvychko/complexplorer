# Expression Evaluation

## ADDED Requirements

### Requirement: Safe complex-expression evaluation

The library SHALL provide an evaluator that turns a complex-function expression string into
values over a complex grid `z`, using a curated, complex-aware symbol table. The grammar
SHALL support `z`, arithmetic (`+ - * / **`, unary minus, parentheses), a curated set of
numpy functions (e.g. `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `abs`, `conj`, `real`,
`imag`, `angle`), and constants (`pi`, `e`, `j`). The evaluator SHALL be PyVista-free.

#### Scenario: Evaluate an expression over a grid

- **WHEN** `evaluate("z / (z**10 - 1)", z)` is called with a complex array `z`
- **THEN** it returns the array of `f(z)` values, equal to the equivalent numpy computation

#### Scenario: Curated functions resolve on complex input

- **WHEN** an expression uses a supported function (e.g. `sqrt(z)`, `exp(1/z)`) on complex
  `z`
- **THEN** the function is the complex-valued numpy implementation and the result is complex

### Requirement: The grammar is restricted to portable math

The evaluator SHALL accept only the curated math grammar — `z`, numeric/imaginary literals,
arithmetic operators, parentheses, and calls to curated function names — and SHALL reject
everything else (attribute access, string literals, comprehensions, lambdas, assignments,
imports, unknown names) by raising the library's domain exception (`ValidationError`). This
keeps expressions safe to exchange across boundaries (shared preset/level files) AND portable
to a native reimplementation (no Python-isms like `z.real` or `'a'.upper()`).

#### Scenario: Off-grammar constructs are rejected

- **WHEN** an expression uses attribute access, a string literal/method, a comprehension, an
  import, or an unknown name (e.g. `z.real`, `z.__class__`, `'a'.upper()`,
  `[i for i in z]`, `__import__('os')`, `foo(z)`)
- **THEN** a `ValidationError` is raised and no arbitrary code executes

#### Scenario: Malformed expression is reported

- **WHEN** a syntactically invalid expression is evaluated
- **THEN** a `ValidationError` is raised describing the problem
