# Function Presets

## ADDED Requirements

### Requirement: Derived answer-key statistics

A preset SHALL expose derived geometry computed from its hand-authored `singularities`,
via an `answer_key_stats()` method, and SHALL include the same record in `to_dict()`. The
statistics are a pure function of the authored answer key (never computed by analyzing the
callable) and SHALL contain:

- `count` — the total number of singularity records,
- `count_by_type` — a mapping of singularity type to count,
- `min_separation` — the smallest Euclidean distance, in the z-plane, between any two
  singularity locations, or `null` when the preset has fewer than two singularities.

#### Scenario: Stats summarize the answer key

- **WHEN** `preset.answer_key_stats()` is called for a preset with multiple singularities
- **THEN** `count` equals the number of records, `count_by_type` tallies the records by `type`, and `min_separation` is the distance between the closest pair of singularity locations

#### Scenario: Separation is null without a pair

- **WHEN** a preset has zero or one singularity
- **THEN** `min_separation` is `null`

#### Scenario: Stats are part of the serialized record

- **WHEN** `preset.to_dict()` is called
- **THEN** the result includes an `answer_key_stats` record (count, count_by_type, min_separation) and remains JSON-serializable
