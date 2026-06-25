# Tasks — enrich-answer-key-stats

## 1. Derived stats on FunctionPreset
- [x] 1.1 Add `FunctionPreset.answer_key_stats()` (`core/presets.py`): `count`,
      `count_by_type` (sorted by type), `min_separation` (global min pairwise Euclidean
      distance over singularity `at` locations, or `None` if < 2). Stdlib only
      (`math.hypot`, `itertools.combinations`); PyVista-free.
- [x] 1.2 Include `"answer_key_stats": self.answer_key_stats()` in `to_dict()`.

## 2. Gallery schema bump
- [x] 2.1 Bump `gallery.SCHEMA_VERSION` `1 → 2` (the card shape changed).
- [x] 2.2 Update the gallery test asserting `schema_version == 1` → `2`.

## 3. Tests
- [x] 3.1 `answer_key_stats` for a multi-singularity preset (`pole_flower_10`): `count`,
      `count_by_type` tally, and `min_separation` == the hand-checked closest pair.
- [x] 3.2 `min_separation is None` for a single-singularity preset (`identity`); and for the
      zero-singularity preset (`exp`): `count == 0`, `count_by_type == {}`, `min_separation is None`.
- [x] 3.3 Duplicate-location case → `0.0` (direct method call on a synthetic/edge input).
- [x] 3.4 `to_dict()` includes `answer_key_stats` and is JSON-serializable.
- [x] 3.5 Gallery determinism still holds (two runs byte-identical) and `schema_version == 2`.

## 4. Close out
- [x] 4.1 Run `pytest tests/` green; `ruff` clean; `openspec validate --specs`.
- [x] 4.2 Update `openspec/ROADMAP.md` (enrich-answer-key-stats status).
