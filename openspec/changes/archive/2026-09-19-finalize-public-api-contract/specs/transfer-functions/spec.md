## ADDED Requirements

### Requirement: Calling a transfer function is specified for scalars and arrays

`TransferFunction.__call__` SHALL document and declare its behaviour for both a scalar complex
argument and an array of them, including the type it returns in each case, so that its use as a
plain callable by every renderer is a stated contract rather than something discovered by
experiment.

#### Scenario: Both call forms are documented and typed

- **WHEN** a transfer function is called with a scalar, and with an array
- **THEN** each form's return type is declared and documented, and both work as described
