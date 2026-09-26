"""Modulus scaling methods for complex function visualization.

This module provides various methods to map the modulus |f(z)| to a radius value,
allowing visualizations to show both phase and magnitude information effectively.
"""

from collections.abc import Callable, Sequence

import numpy as np

from complexplorer.exceptions import ValidationError

__all__ = [
    "ModulusScaling",
    "SCALING_PRESETS",
    "apply_scaling_mode",
    "get_scaling_preset",
    "normalization_constant",
    "sampled_normalization_constant",
]

# Names of the built-in scaling modes, for error messages.
_SCALING_MODE_NAMES = (
    "constant, linear, arctan, logarithmic, log_mixture, linear_clamp, power, sigmoid, adaptive, "
    "hybrid, custom"
)


def apply_scaling_mode(values: np.ndarray, mode: str, params: dict | None = None) -> np.ndarray:
    """Map ``values`` through a named ``ModulusScaling`` mode (or a ``custom`` callable).

    Shared dispatch for the mesh builders (height scaling) and the sphere distortion (radial
    scaling) so the mode lookup and error messages stay in one place.

    Parameters
    ----------
    values : np.ndarray
        Moduli to transform.
    mode : str
        A ``ModulusScaling`` method name, or ``"custom"`` (which requires a ``scaling_func``
        entry in ``params``).
    params : dict, optional
        Keyword arguments for the scaling method (or the ``scaling_func`` for custom mode).

    Raises
    ------
    ValidationError
        If ``mode`` is unknown, or ``custom`` mode is missing ``scaling_func``.
    """
    params = params or {}
    if mode == "custom" and "scaling_func" not in params:
        raise ValidationError("Custom mode requires 'scaling_func' in scaling params")
    method = getattr(ModulusScaling, mode, None)
    if method is None:
        raise ValidationError(f"Unknown scaling mode: {mode}. Available: {_SCALING_MODE_NAMES}")
    return method(values, **params)


class ModulusScaling:
    """Collection of modulus scaling methods for visualization.

    These methods map the modulus |f(z)| to a radius value, allowing
    visualizations to show both phase and magnitude information.
    """

    @staticmethod
    def constant(moduli: np.ndarray, radius: float = 1.0) -> np.ndarray:
        """Constant radius regardless of modulus.

        Parameters
        ----------
        moduli : np.ndarray
            Modulus values |f(z)|.
        radius : float, default=1.0
            Constant radius value.

        Returns
        -------
        np.ndarray
            Array of constant radius values.
        """
        return np.full_like(moduli, radius, dtype=float)

    @staticmethod
    def linear(moduli: np.ndarray, scale: float = 0.1) -> np.ndarray:
        """Linear scaling: r = 1 + scale * |f(z)|.

        Parameters
        ----------
        moduli : np.ndarray
            Modulus values |f(z)|.
        scale : float, default=0.1
            Scaling factor.

        Returns
        -------
        np.ndarray
            Linearly scaled radius values.
        """
        return 1.0 + scale * moduli

    @staticmethod
    def arctan(moduli: np.ndarray, r_min: float = 0.5, r_max: float = 1.5) -> np.ndarray:
        """Smooth scaling using arctangent.

        Maps [0, ∞) to [r_min, r_max] smoothly.

        Parameters
        ----------
        moduli : np.ndarray
            Modulus values |f(z)|.
        r_min : float, default=0.5
            Minimum radius value.
        r_max : float, default=1.5
            Maximum radius value.

        Returns
        -------
        np.ndarray
            Smoothly scaled radius values.
        """
        # Normalize modulus to [0, 1] using arctan
        normalized = (2 / np.pi) * np.arctan(moduli)
        # Map to [r_min, r_max]
        return r_min + (r_max - r_min) * normalized

    @staticmethod
    def logarithmic(
        moduli: np.ndarray, base: float = np.e, r_min: float = 0.5, r_max: float = 1.5
    ) -> np.ndarray:
        """Logarithmic scaling for large dynamic range.

        Good for functions with exponential growth.

        Parameters
        ----------
        moduli : np.ndarray
            Modulus values |f(z)|.
        base : float, default=e
            Logarithm base.
        r_min : float, default=0.5
            Minimum radius value.
        r_max : float, default=1.5
            Maximum radius value.

        Returns
        -------
        np.ndarray
            Logarithmically scaled radius values.
        """
        # Avoid log(0)
        safe_moduli = np.maximum(moduli, 1e-10)
        # Log scaling
        log_moduli = np.log(safe_moduli) / np.log(base)
        # Use sigmoid to map to [0, 1]
        normalized = 1 / (1 + np.exp(-log_moduli))
        # Map to [r_min, r_max]
        return r_min + (r_max - r_min) * normalized

    @staticmethod
    def log_mixture(
        moduli: np.ndarray,
        scale: float = 2.0,
        boost: float = 3.0,
        weight: float = 0.5,
        r_min: float = 0.2,
        r_max: float = 1.0,
    ) -> np.ndarray:
        """Two-scale logistic in the log-modulus: bulk contrast without blunting the tips.

        A single logistic makes one parameter do two jobs. Lowering ``scale`` adds contrast across
        the bulk of the surface but blunts the tips (the tip exponent ``mu / scale`` rises above 1);
        raising it sharpens the tips but squeezes the whole body toward mid-radius. On a shape with
        only a few widely separated features the result is a near-perfect sphere with features poked
        into it.

        So this mixes two: a narrow logistic at ``scale / boost`` carrying bulk contrast, and the
        original wide one at ``scale`` carrying the tips, weighted ``weight`` to ``1 - weight``.

        Both terms are odd in ``log|f|`` about zero and the weights sum to one, so the transfer stays
        self-dual -- a zero of order ``k`` carves the mirror image of what a pole of order ``k``
        raises, and normalization keeps working. It is also **smooth** at ``|f| = 1``, which matters
        more than it sounds: a signed power of the log modulus is monotone, odd and tempting, but has
        infinite derivative at sea level, creases the surface along every sea-level contour, and
        splits the mesh seam badly enough that the weld fails and the solid never closes.

        Parameters
        ----------
        moduli : np.ndarray
            Modulus values |f(z)|.
        scale : float, default=2.0
            The wide logistic's e-folding in log modulus, as in ``logarithmic``'s ``ln(base)``. The
            tip exponent is ``mu / scale`` for a feature of order ``mu``.
        boost : float, default=3.0
            How much narrower the bulk-contrast logistic is. Must be greater than zero; values
            around 3 to 4 are what the reference collection uses.
        weight : float, default=0.5
            Share given to the narrow term, in ``[0, 1]``.
        r_min, r_max : float
            Radius bounds. ``r_min`` is the relief depth.

        Returns
        -------
        np.ndarray
            Radius values in ``[r_min, r_max]``.

        Raises
        ------
        ValidationError
            If ``scale`` or ``boost`` is not positive, or ``weight`` is outside ``[0, 1]``.

        Notes
        -----
        Honest limit: asymptotically the wide term still sets the tip exponent, but that asymptote
        lies below mesh resolution, so in practice the tip climbs through only the last
        ``1 - weight`` of the range and the visible point is measurably shorter than the exponent
        implies.
        """
        if scale <= 0:
            raise ValidationError(f"log_mixture requires scale > 0; got {scale}")
        if boost <= 0:
            raise ValidationError(f"log_mixture requires boost > 0; got {boost}")
        if not 0.0 <= weight <= 1.0:
            raise ValidationError(f"log_mixture requires weight in [0, 1]; got {weight}")

        moduli = np.asarray(moduli, dtype=float)
        with np.errstate(all="ignore"):
            log_modulus = np.log(np.where(moduli > 0, moduli, 1e-300))
            # A pole sampled exactly gives +inf; carry it as a large finite value so the logistic
            # saturates at 1 instead of producing nan.
            log_modulus = np.where(np.isfinite(log_modulus), log_modulus, 700.0)

        def logistic(k: float) -> np.ndarray:
            return 1.0 / (1.0 + np.exp(-np.clip(log_modulus / k, -700.0, 700.0)))

        height = weight * logistic(scale / boost) + (1.0 - weight) * logistic(scale)
        return r_min + (r_max - r_min) * height

    @staticmethod
    def linear_clamp(
        moduli: np.ndarray, m_max: float = 10, r_min: float = 0.5, r_max: float = 1.5
    ) -> np.ndarray:
        """Linear scaling with clamping.

        Linear up to m_max, then constant.

        Parameters
        ----------
        moduli : np.ndarray
            Modulus values |f(z)|.
        m_max : float, default=10
            Maximum modulus value before clamping.
        r_min : float, default=0.5
            Minimum radius value.
        r_max : float, default=1.5
            Maximum radius value.

        Returns
        -------
        np.ndarray
            Linearly scaled and clamped radius values.
        """
        # Clamp moduli to [0, m_max]
        clamped = np.minimum(moduli, m_max)
        # Normalize to [0, 1]
        normalized = clamped / m_max
        # Map to [r_min, r_max]
        return r_min + (r_max - r_min) * normalized

    @staticmethod
    def power(
        moduli: np.ndarray, exponent: float = 0.5, r_min: float = 0.5, r_max: float = 1.5
    ) -> np.ndarray:
        """Power scaling: r = r_min + (r_max - r_min) * (|f|/|f|_max)^exponent.

        Exponent < 1 compresses large values, > 1 expands them.

        Parameters
        ----------
        moduli : np.ndarray
            Modulus values |f(z)|.
        exponent : float, default=0.5
            Power exponent.
        r_min : float, default=0.5
            Minimum radius value.
        r_max : float, default=1.5
            Maximum radius value.

        Returns
        -------
        np.ndarray
            Power-scaled radius values.
        """
        # Normalize by maximum modulus
        max_mod = np.max(moduli)
        if max_mod > 0:
            normalized = (moduli / max_mod) ** exponent
        else:
            normalized = np.zeros_like(moduli)
        # Map to [r_min, r_max]
        return r_min + (r_max - r_min) * normalized

    @staticmethod
    def custom(
        moduli: np.ndarray,
        scaling_func: Callable[[np.ndarray], np.ndarray],
        r_min: float = 0.5,
        r_max: float = 1.5,
    ) -> np.ndarray:
        """Custom scaling function.

        Parameters
        ----------
        moduli : np.ndarray
            Modulus values |f(z)|.
        scaling_func : callable
            User-defined function that maps moduli to [0, 1].
        r_min : float, default=0.5
            Minimum radius value.
        r_max : float, default=1.5
            Maximum radius value.

        Returns
        -------
        np.ndarray
            Custom-scaled radius values.
        """
        # Apply custom function and clip to [0, 1]
        normalized = np.clip(scaling_func(moduli), 0, 1)
        # Map to [r_min, r_max]
        return r_min + (r_max - r_min) * normalized

    @staticmethod
    def sigmoid(
        moduli: np.ndarray,
        steepness: float = 2.0,
        center: float = 1.0,
        r_min: float = 0.5,
        r_max: float = 1.5,
    ) -> np.ndarray:
        """Sigmoid (S-curve) scaling.

        Provides smooth transition with adjustable steepness and center.
        Good general-purpose scaling for most functions.

        Parameters
        ----------
        moduli : np.ndarray
            Modulus values |f(z)|.
        steepness : float, default=2.0
            Controls transition sharpness (higher = steeper).
        center : float, default=1.0
            Center of transition (where r ≈ (r_min + r_max) / 2).
        r_min : float, default=0.5
            Minimum radius value.
        r_max : float, default=1.5
            Maximum radius value.

        Returns
        -------
        np.ndarray
            Sigmoid-scaled radius values.
        """
        # Sigmoid function
        normalized = 1 / (1 + np.exp(-steepness * (moduli - center)))
        # Map to [r_min, r_max]
        return r_min + (r_max - r_min) * normalized

    @staticmethod
    def adaptive(
        moduli: np.ndarray,
        low_percentile: float = 10,
        high_percentile: float = 90,
        r_min: float = 0.5,
        r_max: float = 1.5,
    ) -> np.ndarray:
        """Adaptive percentile-based scaling.

        Automatically adjusts to data range, ignoring outliers.
        Excellent for unknown functions or those with extreme values.

        Parameters
        ----------
        moduli : np.ndarray
            Modulus values |f(z)|.
        low_percentile : float, default=10
            Lower percentile for mapping to r_min.
        high_percentile : float, default=90
            Upper percentile for mapping to r_max.
        r_min : float, default=0.5
            Minimum radius value.
        r_max : float, default=1.5
            Maximum radius value.

        Returns
        -------
        np.ndarray
            Adaptively scaled radius values.
        """
        # Get finite values only
        finite_moduli = moduli[np.isfinite(moduli)]
        if len(finite_moduli) == 0:
            return np.full_like(moduli, r_min)

        # Calculate percentiles
        p_low = np.percentile(finite_moduli, low_percentile)
        p_high = np.percentile(finite_moduli, high_percentile)

        # Handle edge case where all values are similar
        if p_high <= p_low:
            return np.full_like(moduli, (r_min + r_max) / 2)

        # Normalize to [0, 1] based on percentiles
        normalized = np.clip((moduli - p_low) / (p_high - p_low), 0, 1)

        # Map to [r_min, r_max]
        return r_min + (r_max - r_min) * normalized

    @staticmethod
    def hybrid(
        moduli: np.ndarray, transition: float = 1.0, r_min: float = 0.5, r_max: float = 1.5
    ) -> np.ndarray:
        """Hybrid linear-logarithmic scaling.

        Linear for |f| < transition, logarithmic for larger values.
        Ideal for functions with detailed behavior near zero.

        Parameters
        ----------
        moduli : np.ndarray
            Modulus values |f(z)|.
        transition : float, default=1.0
            Transition point between linear and logarithmic.
        r_min : float, default=0.5
            Minimum radius value.
        r_max : float, default=1.5
            Maximum radius value.

        Returns
        -------
        np.ndarray
            Hybrid-scaled radius values.
        """
        normalized = np.zeros_like(moduli)

        # Linear part: [0, transition] -> [0, 0.5]
        small_mask = moduli <= transition
        if np.any(small_mask):
            normalized[small_mask] = 0.5 * moduli[small_mask] / transition

        # Logarithmic part: (transition, ∞) -> (0.5, 1]
        large_mask = ~small_mask
        if np.any(large_mask):
            with np.errstate(divide="ignore", invalid="ignore"):
                log_values = np.log(moduli[large_mask] / transition)
                # Use tanh to compress to (0.5, 1]
                normalized[large_mask] = 0.5 + 0.5 * np.tanh(log_values)

        # Map to [r_min, r_max]
        return r_min + (r_max - r_min) * normalized


# Scaling presets for common use cases
SCALING_PRESETS = {
    "balanced": {
        "method": "sigmoid",
        "params": {"steepness": 2.0, "center": 1.0, "r_min": 0.2, "r_max": 1.0},
        "description": "General purpose sigmoid scaling for balanced visualization",
    },
    "detail_near_zero": {
        "method": "hybrid",
        "params": {"transition": 0.5, "r_min": 0.2, "r_max": 1.0},
        "description": "Emphasizes small values with hybrid linear-log scaling",
    },
    "auto": {
        "method": "adaptive",
        "params": {"low_percentile": 5, "high_percentile": 95, "r_min": 0.2, "r_max": 1.0},
        "description": "Adaptive scaling that automatically adjusts to function range",
    },
    "high_contrast": {
        "method": "sigmoid",
        "params": {"steepness": 5.0, "center": 1.0, "r_min": 0.1, "r_max": 1.0},
        "description": "High contrast sigmoid with steep transition",
    },
    "poles_emphasis": {
        "method": "power",
        "params": {"exponent": 0.3, "r_min": 0.2, "r_max": 1.0},
        "description": "Emphasizes pole behavior with power scaling",
    },
}


def get_scaling_preset(name: str) -> dict:
    """Get a predefined scaling configuration.

    Parameters
    ----------
    name : str
        Name of the preset. Available presets:
        - 'balanced': General purpose sigmoid scaling
        - 'detail_near_zero': Emphasizes small values
        - 'auto': Adaptive scaling for unknown functions
        - 'high_contrast': High contrast with steep transition
        - 'poles_emphasis': Emphasizes pole behavior

    Returns
    -------
    dict
        Dictionary with 'method' and 'params' keys.

    Raises
    ------
    ValidationError
        If preset name is not recognized.
    """
    if name not in SCALING_PRESETS:
        available = ", ".join(SCALING_PRESETS.keys())
        raise ValidationError(f"Unknown preset: {name}. Available presets: {available}")

    preset = SCALING_PRESETS[name]
    return {"method": preset["method"], "params": preset["params"].copy()}


def normalization_constant(
    zeros: Sequence[complex] | np.ndarray,
    poles: Sequence[complex] | np.ndarray,
    gain: complex = 1.0,
) -> float:
    """Closed-form constant placing the geometric mean of ``|c * f|`` over the sphere at 1.

    For a rational ``f(z) = gain * prod(z - z_j) / prod(z - p_k)`` over its **finite** zeros and
    poles, the constant is ``1 / |gain|`` times the product over the poles of
    ``sqrt(1 + |p_k|^2)``, divided by the product over the zeros of ``sqrt(1 + |z_j|^2)``. That
    closed form exists because the log of the chordal distance to a fixed point integrates to
    ``-1/2`` over the sphere, whatever that point is.

    Normalizing by this constant makes the relief independent of the arbitrary scalar in front of
    ``f``. Sea level sits at ``|f| = 1`` for every self-dual transfer, so that scalar otherwise
    changes the ornament's shape rather than only its labels.

    The constant is self-dual: exchanging the zeros and the poles returns its reciprocal, so the
    relief of ``1/f`` is the relief of ``f`` turned inside out.

    Parameters
    ----------
    zeros : sequence of complex
        The **finite** zeros of ``f``, with multiplicity. A zero at infinity is not listed.
    poles : sequence of complex
        The **finite** poles of ``f``, with multiplicity.
    gain : complex, default=1.0
        The leading coefficient. Only its magnitude matters.

    Returns
    -------
    float
        The normalization constant.

    Raises
    ------
    ValidationError
        If ``gain`` is zero, or a zero or pole is not finite.

    Warnings
    --------
    This is exact only for a rational function given by a **complete** divisor: every finite zero
    and every finite pole, with multiplicity. It does not apply to a transcendental function, nor to
    a partial list of singularities, and it fails silently in both cases -- it returns a confidently
    wrong number. Fed the three listed zeros of ``sin(z)``, which has infinitely many, it returns
    0.092 against a true 0.853. In particular, do not assemble a divisor from a function preset's
    ``singularities`` field, which is illustrative rather than complete. When the divisor is not
    known to be complete, estimate the constant from samples instead, with
    :func:`sampled_normalization_constant`, which has no such failure mode.

    Examples
    --------
    >>> import numpy as np
    >>> round(normalization_constant([0.0], np.exp(2j * np.pi * np.arange(10) / 10)), 6)
    32.0
    >>> round(normalization_constant([], [0.0], gain=3.0), 6)
    0.333333
    """
    if gain == 0:
        raise ValidationError("normalization_constant requires a non-zero gain")

    zeros = np.asarray(zeros, dtype=complex).ravel()
    poles = np.asarray(poles, dtype=complex).ravel()
    for name, divisor in (("zeros", zeros), ("poles", poles)):
        if divisor.size and not np.all(np.isfinite(divisor)):
            raise ValidationError(
                f"normalization_constant requires finite {name}; a zero or pole at infinity is "
                "not part of the finite divisor"
            )

    # Accumulated in log space: a degree-20 divisor would otherwise multiply forty square roots.
    log_c = -np.log(abs(gain))
    log_c += 0.5 * float(np.sum(np.log1p(np.abs(poles) ** 2)))
    log_c -= 0.5 * float(np.sum(np.log1p(np.abs(zeros) ** 2)))
    return float(np.exp(log_c))


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """The value at which the cumulative weight first reaches half the total."""
    order = np.argsort(values)
    values = values[order]
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, 0.5 * cumulative[-1], side="left"))
    return float(values[min(index, values.size - 1)])


def sampled_normalization_constant(
    modulus: np.ndarray,
    sphere_z: np.ndarray,
    *,
    statistic: str = "geometric",
) -> float:
    """Estimate the normalization constant from moduli sampled on a Riemann-sphere grid.

    Takes the samples a relief is already built from, so no second pass over ``f`` is needed. Unlike
    :func:`normalization_constant` it needs no divisor, and so has no silent failure mode on a
    transcendental function or an incomplete list of singularities.

    Both grid corrections below are matters of correctness rather than precision:

    - **Area weighting.** A latitude/longitude grid crowds the poles, so each sample is weighted by
      ``sin(theta) = sqrt(1 - z^2)``. Unweighted, the constant for ``z/(z^10-1)`` -- whose feature
      of order ten sits at infinity, at a pole of the grid -- comes out about ten times wrong.
    - **The duplicated seam.** ``sphere_coordinates`` builds ``phi`` as
      ``linspace(0, 2*pi, resolution)``, whose first and last entries are the same meridian, so that
      meridian is sampled twice. It lies along the positive real axis, exactly where a function with
      real coefficients puts its features: counted twice it biases ``(z-1)/(z+1)`` by about 1%,
      against 0.01% with it dropped.

    Samples where ``log|f|`` is not finite -- the zeros and the poles themselves -- are excluded.

    Parameters
    ----------
    modulus : np.ndarray
        ``|f|`` on a ``sample_sphere`` grid, shaped ``(n_phi, n_theta)``: axis 0 is longitude, whose
        first and last rows are the duplicated seam meridian.
    sphere_z : np.ndarray
        The ``z`` coordinate of each sample, the same shape, i.e. ``field.sphere_xyz[..., 2]``.
    statistic : {'geometric', 'median'}, default='geometric'
        Which location statistic of ``log|f|`` to centre on. ``'geometric'`` gives the area-weighted
        geometric mean of ``|f|``; ``'median'`` gives the area-weighted median, which is more robust
        when a high-order feature at infinity covers enough of the sphere to drag sea level away
        from the structure worth seeing. Both are self-dual.

    Returns
    -------
    float
        The estimated constant, or 1.0 when no usable sample remains.

    Raises
    ------
    ValidationError
        If ``statistic`` is not recognized, or the two arrays disagree in shape.

    Notes
    -----
    Accuracy is limited by ``sample_sphere``'s ``avoid_poles`` clamp, which is fixed rather than
    shrinking with resolution, so the estimate converges to a slightly biased value rather than to
    the exact one. The error stays far below what a printer can resolve -- under 0.1% on the
    functions measured -- but a test comparing this against the closed form must use a loose
    tolerance and must not tighten it as resolution grows.
    """
    if statistic not in ("geometric", "median"):
        raise ValidationError(f"Unknown statistic: {statistic}. Available: geometric, median")

    modulus = np.asarray(modulus, dtype=float)
    sphere_z = np.asarray(sphere_z, dtype=float)
    if modulus.shape != sphere_z.shape:
        raise ValidationError(
            f"modulus and sphere_z must have the same shape; got {modulus.shape} and "
            f"{sphere_z.shape}"
        )

    # Drop the duplicated seam meridian: axis 0 is longitude, and its endpoints coincide.
    if modulus.ndim >= 1 and modulus.shape[0] > 1:
        modulus = modulus[:-1]
        sphere_z = sphere_z[:-1]

    with np.errstate(all="ignore"):
        log_modulus = np.log(modulus).ravel()
    weight = np.sqrt(np.clip(1.0 - sphere_z.ravel() ** 2, 0.0, None))

    good = np.isfinite(log_modulus) & (weight > 0)
    if not np.any(good):
        return 1.0
    log_modulus = log_modulus[good]
    weight = weight[good]

    # One code path: both statistics are exp(-centre) for a location statistic of log|f|.
    if statistic == "geometric":
        centre = float(np.average(log_modulus, weights=weight))
    else:
        centre = _weighted_median(log_modulus, weight)
    return float(np.exp(-centre))
