"""Rational transfer functions H = N/D and their canonical views.

The design center is that :class:`TransferFunction` is a plain complex **callable**:
``tf(s)`` evaluates ``polyval(num, s) / polyval(den, s)``, so the object composes with every
complexplorer renderer directly — ``cp.plot(domain, tf)``, ``cp.plot_landscape_pv``,
``cp.riemann_pv``, ``cp.quick_plot``, and STL export all accept it as an ordinary function.
The views in this module add the EE-specific annotations on top:

- :func:`pole_zero_plot` — poles ``×``, zeros ``○``, stability boundary
- :func:`bode_plot` — magnitude (dB) and phase (degrees) over log frequency
- :func:`nyquist_plot` — the ``H(jω)`` locus with the critical point ``−1`` marked
- :func:`transfer_portrait` — a phase portrait of ``H`` with poles/zeros and the stability
  boundary overlaid (composes the standard 2D ``plot()`` path)

Continuous-time systems (``system="s"``) use the imaginary axis as frequency contour and
stability boundary; discrete-time systems (``system="z"``) use the unit circle.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..core.domain import Rectangle
from ..exceptions import ValidationError

_SYSTEMS = ("s", "z")


class TransferFunction:
    """A rational transfer function ``H = N/D`` from polynomial coefficients.

    Parameters
    ----------
    num : sequence of numbers
        Numerator coefficients in ``numpy.polyval`` order (highest degree first).
    den : sequence of numbers
        Denominator coefficients (same order). Must not be identically zero.
    system : str, default="s"
        ``"s"`` for continuous time (frequency contour and stability boundary are the
        imaginary axis) or ``"z"`` for discrete time (the unit circle).

    Examples
    --------
    >>> H = TransferFunction([1], [1, 2, 2])   # 1 / (s^2 + 2s + 2)
    >>> H.is_stable
    True
    >>> import complexplorer as cp
    >>> cp.plot(cp.Rectangle(6, 6), H)  # doctest: +SKIP
    """

    def __init__(self, num, den, system: str = "s"):
        self.num = _validate_coefficients(num, "num")
        self.den = _validate_coefficients(den, "den")
        if not np.any(self.den):
            raise ValidationError("den must not be identically zero")
        if system not in _SYSTEMS:
            raise ValidationError(f"Unknown system {system!r}; supported: {_SYSTEMS}")
        self.system = system

    # -- structure ----------------------------------------------------------------------

    @property
    def zeros(self) -> np.ndarray:
        """Roots of the numerator (no pole/zero cancellation is attempted)."""
        return np.roots(self.num)

    @property
    def poles(self) -> np.ndarray:
        """Roots of the denominator (no pole/zero cancellation is attempted)."""
        return np.roots(self.den)

    @property
    def is_stable(self) -> bool:
        """Strict stability: all poles in the open left half-plane (``s``) / unit disk (``z``).

        Marginal poles (on the boundary) count as unstable — the conservative convention.
        """
        poles = self.poles
        if poles.size == 0:
            return True
        if self.system == "s":
            return bool(np.all(poles.real < 0))
        return bool(np.all(np.abs(poles) < 1))

    # -- evaluation ---------------------------------------------------------------------

    def __call__(self, s):
        """Evaluate ``H`` at complex point(s) ``s`` (scalar or array)."""
        s = np.asarray(s, dtype=complex)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.polyval(self.num, s) / np.polyval(self.den, s)

    def frequency_response(self, omega=None) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate ``H`` along the frequency contour.

        ``s = jω`` for continuous systems, ``z = e^{jω}`` for discrete systems.

        Parameters
        ----------
        omega : array-like, optional
            Frequencies (rad/s or rad/sample). If omitted, a log-spaced grid is chosen
            spanning a decade beyond the smallest and largest nonzero pole/zero magnitude
            (fallback ``[0.01, 100]``); for ``"z"`` systems, one period ``(0, π]``.

        Returns
        -------
        (omega, response) : tuple of ndarray
            The frequency grid and the complex response ``H`` along the contour.
        """
        if omega is None:
            omega = self._auto_omega()
        omega = np.asarray(omega, dtype=float)
        contour = 1j * omega if self.system == "s" else np.exp(1j * omega)
        return omega, self(contour)

    def _auto_omega(self, n: int = 500) -> np.ndarray:
        if self.system == "z":
            return np.linspace(np.pi / n, np.pi, n)
        features = np.abs(np.concatenate([self.poles, self.zeros]))
        features = features[features > 0]
        if features.size == 0:
            lo, hi = 1e-2, 1e2
        else:
            lo, hi = features.min() / 10.0, features.max() * 10.0
        return np.logspace(np.log10(lo), np.log10(hi), n)

    def __repr__(self) -> str:
        return (
            f"TransferFunction(num={self.num.tolist()}, den={self.den.tolist()}, "
            f"system={self.system!r})"
        )


def _validate_coefficients(coeffs, name: str) -> np.ndarray:
    arr = np.atleast_1d(np.asarray(coeffs, dtype=complex))
    if arr.ndim != 1 or arr.size == 0:
        raise ValidationError(f"{name} must be a non-empty 1-D coefficient sequence")
    return arr


# ---------------------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------------------


def _draw_stability_boundary(ax: Axes, tf: TransferFunction) -> None:
    """Imaginary axis (``s``) or unit circle (``z``), drawn as a dashed guide."""
    if tf.system == "s":
        ax.axvline(0.0, color="0.3", linestyle="--", linewidth=1.0, zorder=2)
    else:
        circle = plt.Circle((0, 0), 1.0, fill=False, color="0.3", linestyle="--", linewidth=1.0)
        ax.add_patch(circle)


def _draw_pole_zero_markers(ax: Axes, tf: TransferFunction) -> None:
    poles, zeros = tf.poles, tf.zeros
    if zeros.size:
        ax.plot(zeros.real, zeros.imag, "o", mfc="none", mec="black", mew=1.5, ms=9, label="zeros")
    if poles.size:
        ax.plot(poles.real, poles.imag, "x", color="black", mew=2.0, ms=9, label="poles")


def pole_zero_plot(tf: TransferFunction, ax: Axes | None = None, title: str | None = None):
    """Pole-zero map: poles ``×``, zeros ``○``, stability boundary dashed.

    Returns the matplotlib axes.
    """
    if ax is None:
        _, ax = plt.subplots()
    _draw_stability_boundary(ax, tf)
    _draw_pole_zero_markers(ax, tf)
    pts = np.concatenate([tf.poles, tf.zeros, [0j]])
    r = max(np.abs(pts).max() * 1.3, 1.3)
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Re(s)" if tf.system == "s" else "Re(z)")
    ax.set_ylabel("Im(s)" if tf.system == "s" else "Im(z)")
    if tf.poles.size or tf.zeros.size:
        ax.legend(loc="upper right", fontsize="small")
    ax.set_title(title or "Pole-zero map")
    return ax


def bode_plot(tf: TransferFunction, omega=None, title: str | None = None) -> Figure:
    """Bode plot: magnitude ``20·log10|H|`` (dB) and phase (degrees) over log frequency.

    Returns the matplotlib figure (two stacked panels sharing the frequency axis).
    """
    omega, resp = tf.frequency_response(omega)
    fig, (ax_mag, ax_ph) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    with np.errstate(divide="ignore", invalid="ignore"):
        mag_db = 20.0 * np.log10(np.abs(resp))
    ax_mag.semilogx(omega, mag_db)
    ax_mag.set_ylabel("Magnitude (dB)")
    ax_mag.grid(True, which="both", alpha=0.3)
    ax_ph.semilogx(omega, np.degrees(np.unwrap(np.angle(resp))))
    ax_ph.set_ylabel("Phase (deg)")
    ax_ph.set_xlabel("ω (rad/s)" if tf.system == "s" else "ω (rad/sample)")
    ax_ph.grid(True, which="both", alpha=0.3)
    fig.suptitle(title or "Bode plot")
    return fig


def nyquist_plot(
    tf: TransferFunction, omega=None, ax: Axes | None = None, title: str | None = None
):
    """Nyquist plot: the locus of ``H`` along the frequency contour, ``−1`` marked.

    Positive frequencies are drawn solid; the mirrored negative-frequency branch dashed.
    Returns the matplotlib axes.
    """
    omega, resp = tf.frequency_response(omega)
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(resp.real, resp.imag, "-", color="C0")
    # Negative frequencies: H(-jw) = conj(H(jw)) for real coefficients; evaluate honestly.
    _, resp_neg = tf.frequency_response(-omega[::-1])
    ax.plot(resp_neg.real, resp_neg.imag, "--", color="C0", alpha=0.7)
    ax.plot([-1], [0], "+", color="red", mew=2, ms=10)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Re H")
    ax.set_ylabel("Im H")
    ax.set_title(title or "Nyquist plot")
    return ax


def transfer_portrait(
    tf: TransferFunction,
    domain: Rectangle | None = None,
    resolution: int = 400,
    ax: Axes | None = None,
    title: str | None = None,
    **plot_kwargs,
):
    """Phase portrait of ``H`` with poles/zeros and the stability boundary overlaid.

    Composes the standard 2D ``plot()`` path (so options like ``cmap=`` and ``legend=True``
    are honored). When no domain is given, a square domain enclosing all poles and zeros
    (with margin) is used.

    Returns the matplotlib axes.
    """
    from ..plotting.matplotlib.plot_2d import plot

    if domain is None:
        pts = np.concatenate([tf.poles, tf.zeros, [0j]])
        half = max(float(np.abs(pts.real).max()), float(np.abs(pts.imag).max()), 1.0) * 1.5
        domain = Rectangle(2 * half, 2 * half)

    ax = plot(domain, tf, resolution=resolution, ax=ax, **plot_kwargs)
    _draw_stability_boundary(ax, tf)
    _draw_pole_zero_markers(ax, tf)
    var = tf.system
    ax.set_xlabel(f"Re({var})")
    ax.set_ylabel(f"Im({var})")
    ax.set_title(title or f"H({var}) phase portrait")
    return ax


__all__ = [
    "TransferFunction",
    "pole_zero_plot",
    "bode_plot",
    "nyquist_plot",
    "transfer_portrait",
]
