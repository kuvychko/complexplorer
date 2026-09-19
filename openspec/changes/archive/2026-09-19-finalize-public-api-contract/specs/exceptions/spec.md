## ADDED Requirements

### Requirement: The public API raises only the library's own exceptions

Every error reachable by using the public API as documented SHALL be a `ComplexplorerError`
subclass, so that one `except` clause catches all of them. Bare Python exceptions raised from
inside an operation — `ZeroDivisionError`, `IndexError`, `AttributeError` — are defects, not part
of the contract.

Messages SHALL name the offending value and the accepted values, and where a name replaced a 2.x
name, SHALL name the replacement.

#### Scenario: An invalid argument is reported by the library, not by Python

- **WHEN** a public constructor or entry point is given an invalid argument
- **THEN** the error raised is a `ComplexplorerError` subclass whose message names the value
  received and what would be accepted
