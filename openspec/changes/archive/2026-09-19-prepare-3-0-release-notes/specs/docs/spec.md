## ADDED Requirements

### Requirement: A migration guide covers the upgrade from the published release

The documentation SHALL include a migration guide written for the upgrade from the latest
published version, with a short appendix for older ones. It SHALL state the few things a reader
must do first, and then give old-to-new mappings for every removed or renamed part of the public
surface, so that a reader can resolve an `ImportError` or a `ValidationError` by looking up the
name they used.

Every entry SHALL name a replacement, or say plainly that the capability was removed and why.

#### Scenario: A removed name can be looked up

- **WHEN** a reader searches the migration guide for a name that 3.0 removed or renamed
- **THEN** they find it, with its replacement or the reason it is gone

#### Scenario: The guide's examples run

- **WHEN** the code in the migration guide's before-and-after example is executed against the
  released package
- **THEN** the "after" form runs

### Requirement: Published claims about the project are supportable

Documentation and package description SHALL NOT state performance figures, priority claims, or
quality superlatives that the project cannot support with evidence. A specific, checkable
statement is preferred to an impressive one; where a claim has no measurement behind it, it SHALL
be removed rather than restated more cautiously.

#### Scenario: An unmeasured performance figure is not published

- **WHEN** the README, the documentation or the package description is reviewed before a release
- **THEN** it contains no speed multiplier, "first library" claim, or similar superlative that is
  not backed by a measurement or a citation in the repository
