"""Unit tests for the safe expression evaluator (core/expression.py)."""

import warnings

import numpy as np
import pytest

import complexplorer as cp
from complexplorer.core.expression import compile_expression, evaluate
from complexplorer.utils.validation import ValidationError

Z = np.array([0.37 + 0.41j, 1.2 - 0.6j, -0.8 + 0.2j])


class TestEvaluate:
    def test_arithmetic_matches_numpy(self):
        np.testing.assert_allclose(evaluate("z**2 - 1", Z), Z**2 - 1)
        np.testing.assert_allclose(evaluate("(z - 1) / (z + 1)", Z), (Z - 1) / (Z + 1))

    def test_complex_functions(self):
        warnings.simplefilter("ignore")
        np.testing.assert_allclose(evaluate("sqrt(z)", Z), np.sqrt(Z))
        np.testing.assert_allclose(evaluate("exp(1/z)", Z), np.exp(1 / Z))
        np.testing.assert_allclose(evaluate("z**(1/3)", Z), Z ** (1 / 3))

    def test_constants_and_imaginary(self):
        np.testing.assert_allclose(evaluate("z*j + pi", Z), Z * 1j + np.pi)
        np.testing.assert_allclose(evaluate("2j * z", Z), 2j * Z)

    def test_compile_expression_returns_callable(self):
        f = compile_expression("z**2")
        np.testing.assert_allclose(f(Z), Z**2)


class TestGrammarSafety:
    @pytest.mark.parametrize(
        "expr",
        [
            "z.real",  # attribute access
            "z.__class__",  # dunder
            "'a'.upper()",  # string literal + method
            "[i for i in z]",  # comprehension
            "__import__('os')",  # import escape
            "foo(z)",  # unknown function
            "open('x')",  # unknown name
            "lambda x: x",  # lambda
            "z = 1",  # assignment / not an expression
        ],
    )
    def test_off_grammar_raises(self, expr):
        with pytest.raises(ValidationError):
            evaluate(expr, Z)

    def test_malformed_raises(self):
        with pytest.raises(ValidationError):
            evaluate("z +", Z)

    def test_no_arbitrary_code_executes(self):
        # A format-string / attribute escape must never reach object internals.
        with pytest.raises(ValidationError):
            evaluate("'{0.__class__}'.format(z)", Z)


def test_expression_module_is_pyvista_free():
    import complexplorer.core.expression as mod

    with open(mod.__file__, encoding="utf-8") as fh:
        assert "pyvista" not in fh.read()


def test_catalog_expressions_match_callables():
    """Drift guard: every preset's expression evaluates to ~its callable (away from poles)."""
    warnings.simplefilter("ignore")
    for pid in cp.catalog.list():
        p = cp.catalog.get(pid)
        got = evaluate(p.expression, Z)
        want = np.asarray(p.func(Z))
        np.testing.assert_allclose(got, want, equal_nan=True, err_msg=f"{pid}: {p.expression}")
