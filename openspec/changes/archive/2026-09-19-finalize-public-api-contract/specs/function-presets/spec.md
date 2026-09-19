## ADDED Requirements

### Requirement: The interchange shapes are declared

`domain_spec`, `cmap_spec` and `scaling_spec`, and the singularity records, SHALL have declared
shapes that a type checker can verify, rather than being bare `dict`. The declarations SHALL
describe the same structure the manifest serializes, so the interchange record and the in-memory
record cannot drift apart silently.

#### Scenario: A malformed spec is visible to a type checker

- **WHEN** a preset is constructed with a spec dictionary whose keys do not match the declared
  shape
- **THEN** a type checker reports it
