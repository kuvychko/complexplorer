"""Tests for FunctionPreset.answer_key_stats (enrich-answer-key-stats)."""

import json
import math

from complexplorer.core.presets import FunctionPreset, catalog, singularity


def test_multi_singularity_stats():
    """pole_flower_10: 1 zero + 10 poles; closest pair = adjacent roots of unity."""
    stats = catalog.get("pole_flower_10").answer_key_stats()
    assert stats["count"] == 11
    assert stats["count_by_type"] == {"pole": 10, "zero": 1}
    # adjacent unit-circle poles are 2*sin(pi/10) apart, closer than zero(0)->pole(1)=1.0
    assert math.isclose(stats["min_separation"], 2 * math.sin(math.pi / 10), rel_tol=1e-9)


def test_single_singularity_separation_is_none():
    stats = catalog.get("identity").answer_key_stats()
    assert stats["count"] == 1
    assert stats["min_separation"] is None


def test_zero_singularity_preset():
    """exp is entire (its only singularity is the unrepresented essential at infinity)."""
    stats = catalog.get("exp").answer_key_stats()
    assert stats == {"count": 0, "count_by_type": {}, "min_separation": None}


def test_duplicate_location_gives_zero_separation():
    p = FunctionPreset(
        id="dup",
        title="dup",
        expression="z",
        func=lambda z: z,
        singularities=(singularity("zero", 0, 1), singularity("pole", 0, 1)),
    )
    assert p.answer_key_stats()["min_separation"] == 0.0


def test_count_by_type_is_sorted():
    stats = catalog.get("pole_flower_10").answer_key_stats()
    assert list(stats["count_by_type"]) == sorted(stats["count_by_type"])


def test_to_dict_includes_stats_and_is_json_serializable():
    rec = catalog.get("pole_flower_10").to_dict()
    assert "answer_key_stats" in rec
    assert rec["answer_key_stats"] == catalog.get("pole_flower_10").answer_key_stats()
    json.dumps(rec)  # must not raise
