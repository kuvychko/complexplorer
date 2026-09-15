"""The `Phase` colormap must render exactly as it did before the BasePhasePortrait rebase.

`reconcile-with-2-0-release` rebases `Phase` onto the ported 2.0 enhanced-phase base. The
committed gallery portraits depend on its output being unchanged, so the fixture recorded
before that refactor (tests/data/phase_golden.npz) is compared bit-for-bit here.

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
    assert np.array_equal(actual, expected), (
        f"Phase({config}) output changed; the committed gallery portraits would change too"
    )
