"""Safe, portable evaluation of complex-function expression strings.

Turns a string like ``"z / (z**10 - 1)"`` into values over a complex grid ``z``. The grammar
is deliberately tight — only ``z``, numeric/imaginary literals, arithmetic, and calls to a
curated set of functions — so expressions are both **safe** to exchange across boundaries
(shared preset/level files) and **portable** to a native reimplementation (no Python-isms
like ``z.real`` or string methods).

Two layers:

1. An AST grammar gate (stdlib ``ast``) rejects anything off-grammar (attribute access,
   string literals, comprehensions, lambdas, assignments, unknown names).
2. ``asteval.Interpreter(minimal=True)`` with the curated symbol table performs the actual
   evaluation (and blocks dunder/format escapes). asteval does not raise — this module
   inspects ``aev.error`` and raises ``ValidationError``.

PyVista-free; importable in the 2D/core path.
"""

from __future__ import annotations

import ast
from collections.abc import Callable

import numpy as np
from asteval import Interpreter

from ..utils.validation import ValidationError

# Curated, complex-aware function table. All resolve to numpy on complex input.
_FUNCTIONS: dict[str, Callable] = {
    name: getattr(np, name)
    for name in (
        "sin cos tan sinh cosh tanh exp log log10 sqrt abs conj real imag angle power"
    ).split()
}
_FUNCTIONS.update(asin=np.arcsin, acos=np.arccos, atan=np.arctan)

_CONSTANTS: dict[str, complex] = {"pi": np.pi, "e": np.e, "j": 1j, "i": 1j}

_ALLOWED_NAMES = {"z"} | set(_CONSTANTS)
_ALLOWED_CALLS = set(_FUNCTIONS)

# AST node types permitted by the math grammar.
_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Constant,
    ast.Load,
    # operators
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.USub,
    ast.UAdd,
)


def _gate(expression: str) -> ast.Expression:
    """Parse and validate the expression against the portable math grammar."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValidationError(f"Invalid expression {expression!r}: {exc.msg}") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            raise ValidationError(
                f"Attribute access is not allowed in expressions (in {expression!r}); "
                "use a function like real(z) instead of z.real"
            )
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float, complex)):
            raise ValidationError(
                f"Only numeric literals are allowed in expressions (in {expression!r})"
            )
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_CALLS):
                raise ValidationError(
                    f"Unsupported function call in {expression!r}; "
                    f"allowed: {sorted(_ALLOWED_CALLS)}"
                )
        if isinstance(node, ast.Name) and node.id not in (_ALLOWED_NAMES | _ALLOWED_CALLS):
            raise ValidationError(
                f"Unknown name {node.id!r} in {expression!r}; "
                f"allowed: 'z', {sorted(_CONSTANTS)}, and functions {sorted(_ALLOWED_CALLS)}"
            )
        if not isinstance(node, _ALLOWED_NODES):
            raise ValidationError(
                f"Construct {type(node).__name__} is not allowed in expressions "
                f"(in {expression!r}); only z, arithmetic, and curated functions are permitted"
            )
    return tree


def evaluate(expression: str, z: np.ndarray) -> np.ndarray:
    """Evaluate a complex-function ``expression`` over the grid ``z``.

    Parameters
    ----------
    expression : str
        A math expression in terms of ``z`` (e.g. ``"z / (z**10 - 1)"``).
    z : np.ndarray
        Complex sample grid.

    Returns
    -------
    np.ndarray
        ``f(z)`` values.

    Raises
    ------
    ValidationError
        If the expression is off-grammar, malformed, or fails to evaluate.
    """
    _gate(expression)
    aev = Interpreter(symtable={"z": z, **_FUNCTIONS, **_CONSTANTS}, minimal=True, no_print=True)
    with np.errstate(all="ignore"):
        result = aev(expression)
    if aev.error:
        exc_name, message = aev.error[0].get_error()
        raise ValidationError(f"Could not evaluate {expression!r}: {exc_name}: {message}")
    return np.asarray(result)


def compile_expression(expression: str) -> Callable[[np.ndarray], np.ndarray]:
    """Validate ``expression`` once and return a callable ``f(z) -> f(z)`` for reuse."""
    _gate(expression)  # fail fast on bad grammar
    return lambda z: evaluate(expression, z)


__all__ = ["evaluate", "compile_expression"]
