"""Tests for the transfer-function explorer (add-transfer-function-explorer)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

import complexplorer as cp
from complexplorer.ee import (
    TransferFunction,
    bode_plot,
    nyquist_plot,
    pole_zero_plot,
    transfer_portrait,
)
from complexplorer.exceptions import ValidationError


class TestConstructionAndStructure:
    def test_zeros_and_poles_match_roots(self):
        tf = TransferFunction([1, 1], [1, 2, 2])  # zero at -1; poles at -1 +/- j
        np.testing.assert_allclose(sorted(tf.zeros.real), [-1.0])
        np.testing.assert_allclose(sorted(tf.poles.imag), [-1.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(tf.poles.real, [-1.0, -1.0], atol=1e-12)

    def test_callable_matches_polyval_ratio(self):
        tf = TransferFunction([2, 0, 1], [1, 3])
        s = np.array([0.5 + 0.5j, -1.0 + 2.0j, 3.0])
        expected = np.polyval([2, 0, 1], s) / np.polyval([1, 3], s)
        np.testing.assert_allclose(tf(s), expected)

    def test_callable_on_scalar(self):
        tf = TransferFunction([1], [1, 1])
        assert np.isclose(tf(0), 1.0)

    def test_repr_round_trips_info(self):
        tf = TransferFunction([1], [1, 1], system="z")
        assert "system='z'" in repr(tf)

    def test_empty_coefficients_raise(self):
        with pytest.raises(ValidationError):
            TransferFunction([], [1, 1])

    def test_zero_denominator_raises(self):
        with pytest.raises(ValidationError, match="identically zero"):
            TransferFunction([1], [0, 0])

    def test_unknown_system_raises(self):
        with pytest.raises(ValidationError, match="Unknown system"):
            TransferFunction([1], [1, 1], system="q")


class TestStability:
    def test_stable_s(self):
        assert TransferFunction([1], [1, 2, 2]).is_stable

    def test_unstable_s(self):
        assert not TransferFunction([1], [1, -1]).is_stable  # pole at +1

    def test_marginal_s_counts_as_unstable(self):
        assert not TransferFunction([1], [1, 0]).is_stable  # pole at 0

    def test_stable_z(self):
        assert TransferFunction([1], [1, -0.5], system="z").is_stable  # pole at 0.5

    def test_marginal_z_counts_as_unstable(self):
        assert not TransferFunction([1], [1, -1], system="z").is_stable  # pole at 1


class TestFrequencyResponse:
    def test_explicit_omega_evaluates_on_contour_s(self):
        tf = TransferFunction([1], [1, 2, 2])
        omega = np.array([0.1, 1.0, 10.0])
        w, resp = tf.frequency_response(omega)
        np.testing.assert_allclose(resp, tf(1j * omega))
        np.testing.assert_allclose(w, omega)

    def test_explicit_omega_evaluates_on_contour_z(self):
        tf = TransferFunction([1], [1, -0.5], system="z")
        omega = np.array([0.3, 1.0])
        _, resp = tf.frequency_response(omega)
        np.testing.assert_allclose(resp, tf(np.exp(1j * omega)))

    def test_matches_scipy_freqs(self):
        scipy_signal = pytest.importorskip("scipy.signal")
        num, den = [1.0], [1.0, 1.4142, 1.0]  # 2nd-order Butterworth-like
        tf = TransferFunction(num, den)
        omega = np.logspace(-2, 2, 50)
        _, resp = tf.frequency_response(omega)
        _, h = scipy_signal.freqs(num, den, worN=omega)
        np.testing.assert_allclose(resp, h, rtol=1e-10)

    def test_matches_scipy_freqz(self):
        scipy_signal = pytest.importorskip("scipy.signal")
        num, den = [1.0, 0.5], [1.0, -0.3]
        tf = TransferFunction(num, den, system="z")
        omega = np.linspace(0.01, np.pi, 50)
        _, resp = tf.frequency_response(omega)
        _, h = scipy_signal.freqz(num, den, worN=omega)
        np.testing.assert_allclose(resp, h, rtol=1e-10)

    def test_auto_range_spans_features(self):
        tf = TransferFunction([1, 5], [1, 0.2, 100])  # features at 5 and ~10
        omega, _ = tf.frequency_response()
        assert omega.min() <= 0.5 and omega.max() >= 100.0


class TestViews:
    def test_pole_zero_plot(self):
        tf = TransferFunction([1, 1], [1, 2, 2])
        ax = pole_zero_plot(tf)
        assert ax.get_aspect() == 1.0
        plt.close("all")

    def test_bode_plot_structure(self):
        fig = bode_plot(TransferFunction([1], [1, 0.2, 1]))
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 2
        plt.close("all")

    def test_nyquist_plot(self):
        ax = nyquist_plot(TransferFunction([1], [1, 2, 2]))
        assert ax.get_aspect() == 1.0
        plt.close("all")

    def test_transfer_portrait_composes_plot(self):
        tf = TransferFunction([1], [1, 0.2, 1])
        ax = transfer_portrait(tf, resolution=40, legend=True)
        assert len(ax.get_images()) == 1  # the portrait
        assert len(ax.child_axes) == 1  # the phase-wheel legend
        plt.close("all")

    def test_transfer_function_works_with_cp_plot(self):
        tf = TransferFunction([1], [1, 2, 2])
        ax = cp.plot(cp.Rectangle(4, 4), tf, resolution=30)
        assert ax is not None
        plt.close("all")

    def test_namespace_exposed(self):
        assert cp.ee.TransferFunction is TransferFunction
        assert "ee" in cp.__all__
