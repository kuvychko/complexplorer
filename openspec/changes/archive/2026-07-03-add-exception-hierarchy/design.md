# Design — add-exception-hierarchy

## Context

The documented `ComplexplorerError` hierarchy never existed; `ValidationError(ValueError)` in
`utils/validation.py` is the de-facto error type (used at ~60 sites), with 7 bare
`ValueError` stragglers. 3.0.0 ships imminently.

## Goals / Non-Goals

**Goals**: make `except ComplexplorerError` catch every deliberate library error without
breaking any existing `except ValueError` / import path.

**Non-Goals**: a deep taxonomy (DomainError, ColormapError, …) — no consumer needs it and
speculative subclasses violate the project's "avoid fake generality" principle. Converting
`ImportError` (genuine import failures) or CLI `SystemExit`.

## Decisions

1. **Definition lives in `complexplorer/exceptions.py`**; `utils/validation.py` re-imports.
   Rationale: matches the documented layout (CLAUDE.md, config.yaml) and gives exceptions a
   home that doesn't import numpy. Alternative (define in utils, alias in exceptions)
   inverts the documented dependency direction.
2. **Dual inheritance `ValidationError(ComplexplorerError, ValueError)`** rather than moving
   off `ValueError`: zero-risk for existing handlers; `isinstance(e, ValueError)` stays True.
3. **All 7 bare `ValueError`s become `ValidationError`** rather than introducing new
   subclasses — including the two "No mesh generated yet" state errors in
   `OrnamentGenerator`, because `ValidationError` is the only `ValueError`-compatible type
   in the hierarchy and those sites previously raised `ValueError`.

## Risks / Trade-offs

- [MRO surprise for code doing exact-type checks `type(e) is ValueError`] → Such checks were
  already false for most library errors (`ValidationError` sites); no observed usage.

## Migration Plan

Single commit; rollback = revert.

## Open Questions

_None._
