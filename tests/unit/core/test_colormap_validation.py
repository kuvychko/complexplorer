"""Configuration that cannot mean anything must be refused at construction.

The pattern colormaps validated their parameters from the start; the phase-portrait family did
not, so a handful of values were accepted and produced nonsense:

* ``phase_sectors=0`` divided by zero inside the constructor, surfacing as ``ZeroDivisionError``
  rather than the library's own error type;
* ``phase_sectors=-1`` and ``2.5`` were accepted and gave a meaningless sector count;
* ``r_log_base=1`` made every colour NaN, so the portrait came out blank with nothing raised.

These run against the base class, so they cover ``Phase`` and every perceptual family at once.
"""

from __future__ import annotations

import numpy as np
import pytest

import complexplorer as cp
from complexplorer.exceptions import ValidationError

PHASE_PORTRAITS = [
    cp.Phase,
    cp.OklabPhase,
    cp.PerceptualPastel,
    cp.AnalogousWedge,
    cp.DivergingWarmCool,
    cp.Isoluminant,
    cp.CubehelixPhase,
    cp.InkPaper,
    cp.EarthTopographic,
    cp.FourQuadrant,
]


class TestPhaseSectors:
    @pytest.mark.parametrize("colormap", PHASE_PORTRAITS)
    @pytest.mark.parametrize("bad", [0, -1, -12, 2.5, True])
    def test_invalid_sector_counts_are_refused(self, colormap, bad):
        """Every phase portrait inherits the check, so a tenth family cannot reintroduce the gap."""
        with pytest.raises(ValidationError, match="phase_sectors"):
            colormap(phase_sectors=bad)

    @pytest.mark.parametrize("colormap", PHASE_PORTRAITS)
    def test_valid_sector_counts_still_work(self, colormap):
        instance = colormap(phase_sectors=6)
        rgb = instance.rgb(np.array([1 + 1j, -0.5 + 0.25j]))
        assert np.isfinite(rgb).all()

    def test_numpy_integers_are_accepted(self):
        """A sector count computed with numpy is still an integer."""
        assert cp.Phase(phase_sectors=np.int64(6)).phase_sectors == 6

    def test_none_means_unsectored(self):
        assert cp.Phase(phase_sectors=None).phase_sectors is None

    def test_the_message_names_the_value_and_what_is_accepted(self):
        with pytest.raises(ValidationError) as caught:
            cp.Phase(phase_sectors=0)
        message = str(caught.value)
        assert "0" in message and "positive integer" in message

    def test_zero_no_longer_raises_a_bare_python_error(self):
        """The defect this guards: ZeroDivisionError from inside a constructor."""
        with pytest.raises(ValidationError):
            cp.Phase(phase_sectors=0)


class TestModulusParameters:
    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            ({"r_linear_step": -1}, "r_linear_step"),
            ({"r_linear_step": 0}, "r_linear_step"),
            ({"r_log_base": 1}, "r_log_base"),
            ({"r_log_base": 0.5}, "r_log_base"),
            ({"phase_sectors": 6, "auto_scale_r": True, "scale_radius": 0}, "scale_radius"),
            ({"phase_sectors": 6, "auto_scale_r": True, "scale_radius": -2}, "scale_radius"),
        ],
    )
    def test_out_of_range_modulus_parameters_are_refused(self, kwargs, expected):
        with pytest.raises(ValidationError, match=expected):
            cp.Phase(**kwargs)

    def test_a_log_base_of_one_would_have_produced_nan(self):
        """Before the check, this configuration rendered every pixel as NaN and raised nothing."""
        with pytest.raises(ValidationError):
            cp.Phase(r_log_base=1)

        # The valid neighbour still renders finite colour, so the check is not over-tight.
        rgb = cp.Phase(r_log_base=10).rgb(np.array([0.5 + 0.5j, 2 + 1j]))
        assert np.isfinite(rgb).all()


class TestThePublicSurfaceRaisesItsOwnErrors:
    """One ``except`` clause should catch everything the documented API can raise."""

    @pytest.mark.parametrize(
        "call",
        [
            lambda: cp.Rectangle(0, 4),
            lambda: cp.Disk(0),
            lambda: cp.Annulus(-1, 2),
            lambda: cp.Phase(v_base=-0.5),
            lambda: cp.Phase(phase_sectors=0),
            lambda: cp.Phase(r_log_base=1),
            lambda: cp.Chessboard(spacing=-1),
            lambda: cp.catalog.get("no-such-preset"),
            lambda: cp.get_scaling_preset("no-such-preset"),
            lambda: cp.quick_plot(lambda z: z, mode="sideways"),
            lambda: cp.riemann_surface_pv("no-such-family"),
            lambda: cp.ee.TransferFunction([], []),
        ],
    )
    def test_invalid_use_raises_a_complexplorer_error(self, call):
        with pytest.raises(cp.ComplexplorerError):
            call()
