## ADDED Requirements

### Requirement: Status output is encodable on any console

The progress and validation messages printed by STL export, mesh repair and printability
validation SHALL be ASCII. These messages are emitted by the library itself during
`generate_and_save`, so a character the console cannot encode aborts the export with
`UnicodeEncodeError` after the mesh has been built and before the file is written. Status markers
SHALL therefore be written as ASCII markers rather than as symbols.

#### Scenario: A verbose export on a legacy code page

- **WHEN** an ornament is generated and saved with verbose output on a console using a legacy code
  page
- **THEN** the repair and validation report is printed in full and the STL file is written

#### Scenario: Markers carry the same meaning

- **WHEN** the repair or validation report reports success, a warning or a failure
- **THEN** each is marked distinguishably in ASCII, so the report stays readable
