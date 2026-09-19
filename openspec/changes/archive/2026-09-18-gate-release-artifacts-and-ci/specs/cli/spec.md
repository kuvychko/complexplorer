## ADDED Requirements

### Requirement: Console output survives a legacy console encoding

The CLI SHALL produce output on every console it can be run from, including a Windows console using
a legacy code page (cp437 in a stock `cmd.exe`, cp1252 for a redirected stream under many locales).
Preset titles and descriptions are mathematical text and SHALL keep their notation; where the
console cannot represent a character, the CLI SHALL degrade that character and continue, and SHALL
NOT fail with `UnicodeEncodeError`.

#### Scenario: Listing presets on a legacy code page

- **WHEN** `complexplorer list` is run on a console whose encoding cannot represent a preset title
  such as `z³ - z`
- **THEN** the command prints the full listing and exits 0, with only the unrepresentable
  characters degraded

#### Scenario: Notation is not stripped from the catalog

- **WHEN** preset titles and descriptions are authored
- **THEN** they may use mathematical notation, because the degradation happens at the console
  rather than in the catalog
