# Catalog design notes (internal)

Working notes kept out of the published site. The user-facing pages are
`docs/guide/presets-and-catalog.md` and `docs/function-presets.md`.

## The games boundary

The catalog's serialization design exists so that a native reimplementation (originally Godot
game prototyping) can rebuild the same mathematics and be validated against the same exact
answer keys. Nothing in the shipped library depends on that consumer, which is why it is
recorded here rather than in the documentation.

## Parametrized families

The base `FunctionPreset` is **static**. Parametrized "playgrounds" (Möbius, Julia,
`z^(1/n)`, resonators) are a separate, later `FunctionFamily` whose `bind(**params)` emits an
ordinary static preset — so the registry shape here is the snapshot families produce.
