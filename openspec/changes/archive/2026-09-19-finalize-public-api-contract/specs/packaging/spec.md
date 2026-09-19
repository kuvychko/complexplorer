## ADDED Requirements

### Requirement: The public typing surface is gated in CI

CI SHALL type-check the public surface and a small downstream program that imports complexplorer
the way a user would. The gate SHALL cover what users can see rather than the whole package: a
whole-package gate is dominated by numpy scalar-versus-array unions and third-party stub
imprecision, which produces failures that are not defects and teaches contributors to add ignores.

#### Scenario: A wrong public annotation fails the build

- **WHEN** a public entry point is annotated in a way that rejects correct user code, or a public
  callable loses its return type
- **THEN** the typing job fails
