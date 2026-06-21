"""Backend-policy deprecation tests for the matplotlib 3D entry points.

The matplotlib 3D surface functions are deprecated in favor of their PyVista equivalents
and are scheduled for removal at 3.0 (see docs/development/backend-policy.md). The 2D
stereographic charts (`riemann_chart`, `riemann_hemispheres`) are matplotlib 2D and must
NOT be deprecated.
"""

import warnings

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from complexplorer.core.domain import Rectangle
from complexplorer.plotting.matplotlib.plot_2d import riemann_chart, riemann_hemispheres
from complexplorer.plotting.matplotlib.plot_3d import (
    pair_plot_landscape,
    plot_landscape,
    riemann,
)


def _func(z):
    return z**2 - 1


def teardown_function():
    plt.close("all")


@pytest.mark.parametrize(
    "call, replacement",
    [
        (lambda: plot_landscape(Rectangle(2, 2), func=_func, resolution=15), "plot_landscape_pv"),
        (
            lambda: pair_plot_landscape(Rectangle(2, 2), func=_func, resolution=15),
            "pair_plot_landscape_pv",
        ),
        (lambda: riemann(func=_func, resolution=15), "riemann_pv"),
    ],
)
def test_mpl3d_functions_emit_deprecation_warning(call, replacement):
    """Each matplotlib 3D entry point warns and names its PyVista replacement + 3.0."""
    with pytest.warns(DeprecationWarning) as record:
        call()
    messages = [str(w.message) for w in record]
    assert any(replacement in m and "3.0" in m for m in messages), messages


@pytest.mark.parametrize(
    "call",
    [
        lambda: riemann_chart(_func, resolution=20),
        lambda: riemann_hemispheres(_func, resolution=20),
    ],
)
def test_2d_stereographic_charts_not_deprecated(call):
    """The 2D stereographic charts must not emit a backend-deprecation warning."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        call()
    backend_deprecations = [
        w
        for w in record
        if issubclass(w.category, DeprecationWarning) and "matplotlib 3D backend" in str(w.message)
    ]
    assert not backend_deprecations, [str(w.message) for w in backend_deprecations]
