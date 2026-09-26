"""Tests for the relief normalization constant and the two-scale transfer.

The closed form is the oracle here. It is exact for a rational function given by a complete divisor,
so the sampled estimator -- which is what the ornament path actually uses, because it needs no
divisor -- is checked against it rather than against a recorded number.
"""

import numpy as np
import pytest

from complexplorer.core.field import sample_sphere
from complexplorer.core.scaling import (
    ModulusScaling,
    normalization_constant,
    sampled_normalization_constant,
)
from complexplorer.exceptions import ValidationError

# Resolution for the sampled estimator. Raising it does NOT tighten the agreement: sample_sphere's
# avoid_poles clamp is fixed rather than shrinking with resolution, so the estimate converges to a
# slightly biased value. See the tolerance note on TestSampledAgainstClosedForm.
RES = 400


def roots_of_unity(n: int) -> np.ndarray:
    return np.exp(2j * np.pi * np.arange(n) / n)


def sphere_samples(func, resolution: int = RES):
    """``(modulus, sphere_z)`` as the estimator consumes them."""
    field = sample_sphere(func, resolution=resolution)
    return (
        np.asarray(field.modulus, dtype=float),
        np.asarray(field.sphere_xyz, dtype=float)[..., 2],
    )


class TestClosedForm:
    """The closed form, against the values quadrature confirms."""

    @pytest.mark.parametrize(
        ("zeros", "poles", "gain", "expected"),
        [
            ([0.0], roots_of_unity(5), 1.0, 4.0 * np.sqrt(2.0)),
            ([0.0], roots_of_unity(10), 1.0, 32.0),
            ([], [0.0], 3.0, 1.0 / 3.0),
            ([1.0], [-1.0], 1.0, 1.0),  # reciprocal-symmetric divisor
            ([0.0, 0.0], [], 1.0, 1.0),  # z**2
        ],
    )
    def test_matches_expected(self, zeros, poles, gain, expected):
        assert normalization_constant(zeros, poles, gain) == pytest.approx(expected, rel=1e-9)

    def test_reciprocal_symmetric_divisor_is_unity(self):
        """A divisor closed under z -> 1/conj(z) normalizes to 1: the sphere sees no net scale."""
        zeros = [2.0, 0.5, 3j, -1j / 3.0]
        poles = [1.0 / 2.0, 2.0, -1j / 3.0, 3j]
        assert normalization_constant(zeros, poles) == pytest.approx(1.0, rel=1e-12)

    def test_is_self_dual(self):
        """Exchanging zeros and poles gives the reciprocal, so 1/f is f turned inside out."""
        zeros, poles = [0.0, 2.0], roots_of_unity(7)
        forward = normalization_constant(zeros, poles, gain=2.5)
        inverse = normalization_constant(poles, zeros, gain=1 / 2.5)
        assert forward * inverse == pytest.approx(1.0, rel=1e-12)

    def test_gain_scales_reciprocally(self):
        """Doubling the gain halves the constant -- it is undoing the gain."""
        base = normalization_constant([0.0], [2.0], gain=1.0)
        assert normalization_constant([0.0], [2.0], gain=4.0) == pytest.approx(base / 4.0)

    def test_rejects_zero_gain(self):
        with pytest.raises(ValidationError, match="non-zero gain"):
            normalization_constant([0.0], [1.0], gain=0.0)

    @pytest.mark.parametrize("bad", [np.inf, np.nan])
    def test_rejects_non_finite_divisor(self, bad):
        with pytest.raises(ValidationError, match="finite"):
            normalization_constant([bad], [1.0])

    def test_accepts_an_empty_divisor(self):
        assert normalization_constant([], []) == pytest.approx(1.0)


class TestSampledAgainstClosedForm:
    """The estimator the ornament path uses, against the exact oracle.

    The tolerance is 0.5% deliberately and MUST NOT be tightened: accuracy is limited by
    ``sample_sphere``'s fixed ``avoid_poles`` clamp, which does not shrink with resolution, so the
    estimate converges to a slightly biased value rather than to the exact one. The observed error
    on these functions drifts *upward* with resolution (0.041% -> 0.085% as resolution goes 100 to
    400), so a tight tolerance would fail on a finer grid, which is the opposite of useful.
    """

    CASES = {
        "z/(z^5-1)": (lambda z: z / (z**5 - 1), [0.0], roots_of_unity(5), 1.0),
        "z/(z^10-1)": (lambda z: z / (z**10 - 1), [0.0], roots_of_unity(10), 1.0),
        "3/z": (lambda z: 3 / z, [], [0.0], 3.0),
        "(z-1)/(z+1)": (lambda z: (z - 1) / (z + 1), [1.0], [-1.0], 1.0),
        "(z-1j)/(z+1j)": (lambda z: (z - 1j) / (z + 1j), [1j], [-1j], 1.0),
    }

    @pytest.mark.parametrize("name", list(CASES))
    def test_agrees_within_half_a_percent(self, name):
        func, zeros, poles, gain = self.CASES[name]
        exact = normalization_constant(zeros, poles, gain)
        estimate = sampled_normalization_constant(*sphere_samples(func))
        assert estimate == pytest.approx(exact, rel=5e-3)

    def test_is_self_dual(self):
        """The constants for f and 1/f are reciprocal, which is what makes the relief invert."""
        func = lambda z: z / (z**3 - 1)  # noqa: E731
        forward = sampled_normalization_constant(*sphere_samples(func))
        inverse = sampled_normalization_constant(*sphere_samples(lambda z: 1 / func(z)))
        assert forward * inverse == pytest.approx(1.0, rel=5e-3)

    def test_median_is_also_self_dual(self):
        """The alternative statistic has to keep that property, or the relief stops mirroring."""
        func = lambda z: z / (z**3 - 1)  # noqa: E731
        forward = sampled_normalization_constant(*sphere_samples(func), statistic="median")
        inverse = sampled_normalization_constant(
            *sphere_samples(lambda z: 1 / func(z)), statistic="median"
        )
        assert forward * inverse == pytest.approx(1.0, rel=5e-2)

    def test_rejects_an_unknown_statistic(self):
        with pytest.raises(ValidationError, match="Unknown statistic"):
            sampled_normalization_constant(np.ones((4, 4)), np.zeros((4, 4)), statistic="mean")

    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValidationError, match="same shape"):
            sampled_normalization_constant(np.ones((4, 4)), np.zeros((4, 5)))

    def test_singular_samples_are_excluded(self):
        """A grid full of zeros and poles still yields the constant of what is left."""
        modulus, sphere_z = sphere_samples(lambda z: z / (z**3 - 1), resolution=60)
        modulus = modulus.copy()
        modulus[0, 0] = 0.0  # a zero, log -> -inf
        modulus[1, 1] = np.inf  # a pole, log -> +inf
        modulus[2, 2] = np.nan
        assert np.isfinite(sampled_normalization_constant(modulus, sphere_z))

    def test_all_singular_returns_unity(self):
        assert sampled_normalization_constant(np.zeros((8, 8)), np.zeros((8, 8))) == 1.0


class TestTheTwoGridCorrections:
    """Both corrections are correctness, not precision -- so they are tested as correctness."""

    @staticmethod
    def _unweighted_seam_kept(func) -> float:
        """The naive estimate: every sample equal, seam meridian counted twice."""
        modulus, _ = sphere_samples(func)
        with np.errstate(all="ignore"):
            log_modulus = np.log(modulus).ravel()
        log_modulus = log_modulus[np.isfinite(log_modulus)]
        return float(np.exp(-log_modulus.mean()))

    @staticmethod
    def _weighted_seam_kept(func) -> float:
        """Area-weighted, but retaining the duplicated seam meridian."""
        modulus, sphere_z = sphere_samples(func)
        with np.errstate(all="ignore"):
            log_modulus = np.log(modulus).ravel()
        weight = np.sqrt(np.clip(1.0 - sphere_z.ravel() ** 2, 0.0, None))
        good = np.isfinite(log_modulus) & (weight > 0)
        return float(np.exp(-np.average(log_modulus[good], weights=weight[good])))

    def test_area_weighting_is_mandatory(self):
        """A lat/long grid crowds the poles, where z/(z^10-1) keeps its order-ten feature."""
        func = lambda z: z / (z**10 - 1)  # noqa: E731
        exact = normalization_constant([0.0], roots_of_unity(10))

        weighted = sampled_normalization_constant(*sphere_samples(func))
        unweighted = self._unweighted_seam_kept(func)

        assert weighted == pytest.approx(exact, rel=5e-3)
        # Wrong by roughly an order of magnitude, not by a few percent.
        assert unweighted / exact > 5.0

    def test_the_seam_is_dropped(self):
        """phi spans 0 to 2*pi inclusive, so the first and last meridian are the same one.

        It lies along the positive real axis, which is exactly where a real-coefficient function
        puts its features -- so double-counting it biases those functions and nothing else.
        """
        on_seam = lambda z: (z - 1) / (z + 1)  # noqa: E731
        off_seam = lambda z: (z - 1j) / (z + 1j)  # noqa: E731
        exact = 1.0  # both divisors are reciprocal-symmetric

        on_dropped = abs(sampled_normalization_constant(*sphere_samples(on_seam)) / exact - 1)
        on_kept = abs(self._weighted_seam_kept(on_seam) / exact - 1)
        off_dropped = abs(sampled_normalization_constant(*sphere_samples(off_seam)) / exact - 1)
        off_kept = abs(self._weighted_seam_kept(off_seam) / exact - 1)

        # Dropping the seam buys two orders of magnitude on the function that sits on it.
        assert on_kept > 100 * on_dropped
        assert on_dropped < 1e-4
        # And the function whose features avoid the seam is unaffected either way.
        assert off_kept == pytest.approx(off_dropped, abs=1e-9)


class TestLogMixture:
    """The two-scale transfer."""

    def test_is_self_dual(self):
        """r(f) and r(1/f) are mirror images about sea level: they sum to r_min + r_max."""
        moduli = np.array([1e-3, 0.01, 0.1, 0.5, 2.0, 10.0, 100.0, 1e3])
        kwargs = {"scale": 2.0, "boost": 3.0, "weight": 0.5, "r_min": 0.2, "r_max": 1.0}

        forward = ModulusScaling.log_mixture(moduli, **kwargs)
        inverse = ModulusScaling.log_mixture(1.0 / moduli, **kwargs)

        np.testing.assert_allclose(forward + inverse, 1.2, rtol=1e-12)

    def test_sea_level_is_the_midpoint(self):
        assert ModulusScaling.log_mixture(np.array([1.0]), r_min=0.2, r_max=1.0)[
            0
        ] == pytest.approx(0.6)

    def test_is_monotone_and_bounded(self):
        moduli = np.logspace(-6, 6, 200)
        radii = ModulusScaling.log_mixture(moduli, r_min=0.2, r_max=1.0)
        assert np.all(np.diff(radii) > 0)
        assert radii.min() >= 0.2 and radii.max() <= 1.0

    def test_handles_zero_and_infinity(self):
        radii = ModulusScaling.log_mixture(np.array([0.0, np.inf]), r_min=0.2, r_max=1.0)
        assert radii[0] == pytest.approx(0.2)
        assert radii[1] == pytest.approx(1.0)

    def test_spreads_more_area_away_from_mid_radius_than_one_logistic(self):
        """The reason the mode exists: bulk contrast on a body a single logistic leaves smooth."""
        # A modulus distribution clustered near sea level, as a few-featured piece produces.
        moduli = np.exp(np.random.default_rng(0).normal(0.0, 0.35, 20_000))

        single = ModulusScaling.logarithmic(moduli, base=np.exp(2.0), r_min=0.2, r_max=1.0)
        mixed = ModulusScaling.log_mixture(
            moduli, scale=2.0, boost=4.0, weight=0.6, r_min=0.2, r_max=1.0
        )

        mid = 0.6
        assert np.abs(mixed - mid).mean() > np.abs(single - mid).mean()

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"scale": 0.0}, "scale > 0"),
            ({"scale": -1.0}, "scale > 0"),
            ({"boost": 0.0}, "boost > 0"),
            ({"weight": -0.1}, r"weight in \[0, 1\]"),
            ({"weight": 1.5}, r"weight in \[0, 1\]"),
        ],
    )
    def test_validates_its_parameters(self, kwargs, match):
        with pytest.raises(ValidationError, match=match):
            ModulusScaling.log_mixture(np.array([1.0]), **kwargs)

    def test_weight_zero_is_the_plain_logistic(self):
        """With no share given to the narrow term, this reduces to logarithmic at the same scale."""
        moduli = np.logspace(-3, 3, 50)
        mixed = ModulusScaling.log_mixture(moduli, scale=2.0, weight=0.0, r_min=0.2, r_max=1.0)
        plain = ModulusScaling.logarithmic(moduli, base=np.exp(2.0), r_min=0.2, r_max=1.0)
        np.testing.assert_allclose(mixed, plain, rtol=1e-12)
