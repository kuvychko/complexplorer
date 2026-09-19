# Enrich answer-key stats

## Why

The catalog's `singularities` are exact, hand-authored answer keys — but every consumer
(the gallery manifest, any downstream tool, Godot game prototyping) has to recompute the
same derived geometry to *use* them: how many of each type, and how close the closest pair
is. The separation in particular matters: it's what tells a consumer whether two features are
distinguishable at a given resolution/tolerance. complexplorer knows the geometry, so it
should compute it once and ship it — this is the one piece of "games-adjacent" value that is
pure math and squarely in a numerical library's domain (per the roadmap's math/game
boundary; everything game-side — task, scoring, difficulty — stays in Godot).

This is the only surviving piece of the dissolved Phase 3 games scope.

## What changes

- **`FunctionPreset.answer_key_stats()`** — a method returning derived geometry computed from
  the (authored) singularity list, with no new dependencies:
  - `count` — total number of singularity records,
  - `count_by_type` — `{zero: n, pole: n, …}`,
  - `min_separation` — the smallest Euclidean distance (in the z-plane) between any two
    singularity locations, or `null` when there are fewer than two.
- **`to_dict()` includes `answer_key_stats`** — so the derived geometry flows automatically
  into the gallery manifest (`card.json` / `index.json`) and any other serialized consumer.
- The gallery card shape changes (one new field), so the gallery's `SCHEMA_VERSION` bumps
  `1 → 2` and the one test asserting `schema_version == 1` updates.

## Non-goals

- **`has_infinity` / behavior-at-∞ is deferred.** The answer key stores `at` as finite
  `[re, im]` pairs only — there is no representation of a pole/essential *at* ∞. Adding it
  would be a *catalog-authoring* decision (author ∞ into the curated singularities), not a
  derivation. Out of scope here; revisit as its own question.
- No task, scoring, tolerance, or difficulty — those are game design and live in Godot.
- No numerical singularity *detection* — stats are derived from the hand-authored answer
  key, never from analyzing `func`.

## Impact

- Touched: `complexplorer/core/presets.py` (`FunctionPreset`), `complexplorer/gallery.py`
  (`SCHEMA_VERSION` bump), one gallery test assertion.
- Affected specs: `function-presets` — a new "Derived answer-key statistics" requirement;
  the `to_dict()` record gains `answer_key_stats`.
- Determinism preserved: stats are a pure function of authored floats, so the gallery's
  byte-stable manifest contract still holds (a test already diffs two runs).
- Risk: low; additive, single-capability.
