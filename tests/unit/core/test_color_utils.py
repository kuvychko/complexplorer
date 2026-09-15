"""Colour-space helpers ported from 2.0.0 (reconcile-with-2-0-release, task 3.2)."""

import numpy as np
import pytest

from complexplorer.core import color_utils as cu
from complexplorer.exceptions import ValidationError

SWEEP_L = np.linspace(0.25, 0.95, 8)
SWEEP_C = np.linspace(0.0, 0.35, 8)
SWEEP_H = np.linspace(0.0, 2 * np.pi, 8)


class TestOklchToSrgb:
    @pytest.mark.parametrize("clip_method", ["adaptive_chroma", "simple"])
    def test_output_is_in_gamut_and_shaped_like_the_input(self, clip_method):
        L, C = np.meshgrid(SWEEP_L, SWEEP_C)
        H = np.broadcast_to(SWEEP_H, L.shape)
        rgb = cu.oklch_to_srgb(L, C, H, clip_method=clip_method)
        for channel in rgb:
            assert channel.shape == L.shape
            assert np.all(np.isfinite(channel))
            assert np.all((channel >= 0.0) & (channel <= 1.0))

    def test_zero_chroma_is_neutral(self):
        r, g, b = cu.oklch_to_srgb(np.array([0.6]), np.array([0.0]), np.array([1.2]))
        assert r == pytest.approx(g, abs=1e-6)
        assert g == pytest.approx(b, abs=1e-6)

    def test_unknown_clip_method_is_rejected(self):
        with pytest.raises(ValidationError, match="clip_method"):
            cu.oklch_to_srgb(np.array([0.5]), np.array([0.1]), np.array([0.0]), clip_method="nope")


class TestClipToGamut:
    @pytest.mark.parametrize("preserve", ["hue", "lightness"])
    def test_out_of_range_values_are_brought_into_gamut(self, preserve):
        R = np.array([1.6, -0.3, 0.5])
        G = np.array([0.4, 0.5, 0.5])
        B = np.array([0.2, 1.9, 0.5])
        for channel in cu.clip_to_gamut(R, G, B, preserve=preserve):
            assert np.all((channel >= 0.0) & (channel <= 1.0))
            assert np.all(np.isfinite(channel))

    def test_in_gamut_values_are_untouched(self):
        R, G, B = (np.array([0.2, 0.7]), np.array([0.3, 0.6]), np.array([0.4, 0.5]))
        r, g, b = cu.clip_to_gamut(R.copy(), G.copy(), B.copy())
        assert np.allclose(r, R) and np.allclose(g, G) and np.allclose(b, B)

    def test_unknown_preserve_mode_is_rejected(self):
        with pytest.raises(ValidationError, match="preserve"):
            cu.clip_to_gamut(np.array([0.5]), np.array([0.5]), np.array([0.5]), preserve="nope")


class TestHslToRgb:
    @pytest.mark.parametrize(
        "hue,expected",
        [(0.0, (1.0, 0.0, 0.0)), (1 / 3, (0.0, 1.0, 0.0)), (2 / 3, (0.0, 0.0, 1.0))],
    )
    def test_primaries(self, hue, expected):
        rgb = cu.hsl_to_rgb(np.array([hue]), np.array([1.0]), np.array([0.5]))
        assert np.allclose([float(c[0]) for c in rgb], expected, atol=1e-6)

    @pytest.mark.parametrize("lightness,expected", [(0.0, 0.0), (1.0, 1.0)])
    def test_lightness_extremes_are_black_and_white(self, lightness, expected):
        rgb = cu.hsl_to_rgb(np.array([0.25]), np.array([1.0]), np.array([lightness]))
        assert np.allclose([float(c[0]) for c in rgb], [expected] * 3, atol=1e-6)

    def test_zero_saturation_is_grey(self):
        r, g, b = cu.hsl_to_rgb(np.array([0.8]), np.array([0.0]), np.array([0.42]))
        assert float(r[0]) == pytest.approx(0.42, abs=1e-6)
        assert float(g[0]) == pytest.approx(float(r[0]), abs=1e-9)
        assert float(b[0]) == pytest.approx(float(r[0]), abs=1e-9)


class TestCubehelix:
    def test_sweep_stays_in_gamut(self):
        h = np.linspace(0.0, 1.0, 64)
        for channel in cu.cubehelix(h):
            assert channel.shape == h.shape
            assert np.all(np.isfinite(channel))
            assert np.all((channel >= 0.0) & (channel <= 1.0))


class TestInterpolateHue:
    def test_shortest_path_wraps_across_zero(self):
        out = cu.interpolate_hue(0.9, 0.1, 0.5)
        assert float(out) == pytest.approx(0.0, abs=1e-9)

    def test_endpoints_are_preserved(self):
        assert float(cu.interpolate_hue(0.2, 0.7, 0.0)) == pytest.approx(0.2, abs=1e-9)
        assert float(cu.interpolate_hue(0.2, 0.7, 1.0)) == pytest.approx(0.7, abs=1e-9)

    @pytest.mark.parametrize("direction", ["shortest", "longest", "clockwise", "counter-clockwise"])
    def test_every_direction_stays_in_unit_range(self, direction):
        out = cu.interpolate_hue(0.15, 0.85, np.linspace(0, 1, 17), direction=direction)
        assert np.all((out >= 0.0) & (out <= 1.0))

    def test_unknown_direction_is_rejected(self):
        with pytest.raises(ValidationError, match="direction"):
            cu.interpolate_hue(0.1, 0.4, 0.5, direction="sideways")
