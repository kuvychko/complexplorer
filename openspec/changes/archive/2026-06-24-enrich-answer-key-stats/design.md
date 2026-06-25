# Design — enrich answer-key stats

## The method

```python
# FunctionPreset
def answer_key_stats(self) -> dict:
    points = [s["at"] for s in self.singularities]      # each [re, im]
    by_type = {}
    for s in self.singularities:
        by_type[s["type"]] = by_type.get(s["type"], 0) + 1
    return {
        "count": len(self.singularities),
        "count_by_type": dict(sorted(by_type.items())),  # stable key order
        "min_separation": _min_pairwise(points),         # None if < 2 points
    }

def _min_pairwise(points) -> float | None:
    if len(points) < 2:
        return None
    return min(
        math.hypot(a[0]-b[0], a[1]-b[1])
        for a, b in itertools.combinations(points, 2)
    )
```

`to_dict()` gains one line: `"answer_key_stats": self.answer_key_stats()`.

## Decisions

| Question | Decision | Why |
|---|---|---|
| Separation across types? | **Global** — min over *all* singularity locations regardless of type | A consumer distinguishing features by position cares about the closest pair, period; a zero next to a pole is as un-resolvable as two poles. |
| `< 2` singularities | `min_separation = null` | No pair exists; `null` is the honest "not applicable" (consumers branch on it). |
| `0` singularities | `count = 0`, `count_by_type = {}`, `min_separation = null` | Real case: `exp` is entire (its only singularity is the unrepresented essential at ∞). Same `< 2` guard covers it. |
| Duplicate location | `0.0` (a real, degenerate value) | Two records at the same point genuinely have zero separation; don't hide it. |
| Rounding | **None** — full precision | Consistent with the existing `at` coords (already full-precision irrationals); consumers round if they want. |
| `count_by_type` order | sorted by type name | Deterministic record even without `sort_keys`. |
| Stdlib only | `math.hypot` + `itertools.combinations` | No numpy needed; `presets.py` stays light and PyVista-free. n≤~10 per preset, so O(n²) is nothing. |

## Gallery interaction

`card.json == to_dict() + {schema_version, files}`. Since `to_dict()` grows
`answer_key_stats`, the card grows it automatically — no gallery *spec* change (the card is
still "to_dict plus files"). Two concrete edits:
- bump `gallery.SCHEMA_VERSION` `1 → 2` (honest signal that the card shape changed),
- update the one test asserting `schema_version == 1` → `2`.

Determinism is unaffected: `answer_key_stats` is a pure function of authored floats, so two
gallery runs still produce byte-identical manifests (the existing run-twice-diff test covers
it).

## Why has_infinity is deferred (not forgotten)

`min_separation` and `count_by_type` are **derivations** — zero new authored data. `has_infinity`
is not derivable: the answer key has no ∞ representation (`at` is finite `[re, im]`). Shipping
it would mean authoring ∞ singularities into the curated catalog (e.g. a `pole at ∞` for
polynomials, `essential at ∞` for `exp`), which is a content/answer-key decision deserving
its own change. Kept out so this stays a clean, pure-derivation enrichment.

Concretely, the `exp` preset reports `count = 0` today — its essential-at-∞ is unrepresented.
That is the visible edge of the deferred gap, not a bug in this change: the stats faithfully
summarize the authored (finite) answer key.

## Tests

- `answer_key_stats` shape for a multi-singularity preset (e.g. `pole_flower_10`): `count`
  matches, `count_by_type` tallies, `min_separation` equals the hand-checked closest pair.
- `min_separation is None` for a single-singularity preset (e.g. `identity`).
- Duplicate-location case → `0.0` (a small synthetic preset or direct method call).
- `to_dict()` includes `answer_key_stats` and stays JSON-serializable.
- Gallery: still byte-identical across two runs; `schema_version == 2`.
