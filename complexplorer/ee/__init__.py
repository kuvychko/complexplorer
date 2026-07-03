"""Engineering mode: transfer functions and their canonical views.

Usage::

    import complexplorer as cp

    H = cp.ee.TransferFunction([1], [1, 2, 2])   # 1 / (s^2 + 2s + 2)
    cp.ee.transfer_portrait(H, legend=True)
    cp.ee.bode_plot(H)

EE names are namespaced under ``cp.ee`` (not the top-level package) to keep the core
surface math-generic.
"""

from .transfer_function import (
    TransferFunction,
    bode_plot,
    nyquist_plot,
    pole_zero_plot,
    transfer_portrait,
)

__all__ = [
    "TransferFunction",
    "pole_zero_plot",
    "bode_plot",
    "nyquist_plot",
    "transfer_portrait",
]
