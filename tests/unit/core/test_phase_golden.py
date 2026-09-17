"""The `Phase` colormap must render exactly as it did before the BasePhasePortrait rebase.

`reconcile-with-2-0-release` rebases `Phase` onto the ported 2.0 enhanced-phase base. The
committed gallery portraits depend on its output being unchanged, so the fixture recorded
before that refactor (tests/data/phase_golden.npz) is compared here.

The comparison allows half an 8-bit level. The fixture was recorded on one platform and CI
runs on another, where the last bits of the trigonometric and logarithmic work differ; a
difference below half a level cannot change a pixel of a saved PNG, which is what "the
portraits are unchanged" actually means. Anything visible still fails.

Regenerate with tests/data/generate_phase_golden.py ONLY if a Phase output change is
intended and the gallery is re-rendered with it.
"""

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from complexplorer import Phase

# The fixture is recorded in the post-rename naming. Until the rebase (task 3.3) lands, `Phase`
# still takes `phase_sectors` and the comparison cannot run; this guard disappears on its own once
# `phase_sectors` exists.
pytestmark = pytest.mark.skipif(
    "phase_sectors" not in inspect.signature(Phase.__init__).parameters,
    reason="Phase still takes n_phi; the BasePhasePortrait rebase has not landed yet",
)

DATA = Path(__file__).resolve().parents[3] / "tests" / "data"
CONFIGS = json.loads((DATA / "phase_golden_configs.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "index,config",
    list(enumerate(CONFIGS)),
    ids=["-".join(f"{k}={v}" for k, v in c.items()) or "defaults" for c in CONFIGS],
)
def test_phase_output_matches_pre_rebase_reference(index, config):
    with np.load(DATA / "phase_golden.npz") as fixture:
        z = fixture["z"]
        expected = fixture[f"rgb_{index}"]
    actual = Phase(**config).rgb(z)
    assert actual.shape == expected.shape
    assert np.all(np.isfinite(actual)), f"Phase({config}) produced non-finite RGB"

    difference = float(np.max(np.abs(actual - expected)))
    half_a_level = 1 / 512
    assert difference < half_a_level, (
        f"Phase({config}) output changed by {difference:.2e} "
        f"(more than half an 8-bit level, {half_a_level:.2e}); "
        "the committed gallery portraits would change too"
    )
