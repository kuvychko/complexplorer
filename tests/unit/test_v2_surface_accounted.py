"""No 2.0.0 public name may vanish without being documented.

complexplorer 2.0.0 (PyPI, 2025-10-19) is the release users upgrade from. `tests/data/v2_0_0_all.txt`
freezes its `__all__`; every name in it must either still exist in 3.0 or have a row in the
migration inventory, so a removal cannot ship undocumented.
"""

import re
from pathlib import Path

import pytest

import complexplorer as cp

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "tests" / "data" / "v2_0_0_all.txt"
INVENTORY = ROOT / "docs" / "migration-3.0.md"


def _frozen_names():
    return [
        line.strip()
        for line in FROZEN.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def _documented_names():
    rows = re.findall(r"^\| `([^`]+)` \|", INVENTORY.read_text(encoding="utf-8"), re.M)
    return set(rows)


def test_the_frozen_surface_is_not_empty():
    assert len(_frozen_names()) > 50


@pytest.mark.parametrize("name", _frozen_names())
def test_every_2_0_name_is_present_or_documented(name):
    if name in cp.__all__:
        return
    assert name in _documented_names(), (
        f"{name!r} was public in 2.0.0, is gone in 3.0, and has no row in docs/migration-3.0.md"
    )


def test_the_inventory_lists_no_name_that_still_exists():
    stale = sorted(n for n in _documented_names() if n in cp.__all__)
    assert not stale, f"inventory lists names that 3.0 still exports: {stale}"
