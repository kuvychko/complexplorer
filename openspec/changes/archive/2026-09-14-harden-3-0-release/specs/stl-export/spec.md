## REMOVED Requirements

### Requirement: Optional-dependency gating

**Reason**: PyVista became a required core dependency at 3.0 (it is the sole 3D backend and
powers STL export). There is no longer a configuration in which the library imports but PyVista
is absent, so gating STL export behind a runtime availability check — and the `HAS_PYVISTA` /
`check_pyvista_available()` machinery that implemented it — is dead code that contradicts the
shipped dependency contract.

**Migration**: PyVista is installed automatically with `complexplorer`; STL export is always
available. Code that imported the internal `HAS_PYVISTA` flag or called
`check_pyvista_available()` should simply drop the guard and use the export functions directly.

## ADDED Requirements

### Requirement: STL export is always available

STL export SHALL be available whenever `complexplorer` is importable, because PyVista is a
required dependency. The export modules SHALL import PyVista unconditionally and SHALL NOT define
or expose any PyVista-availability flag or gating check.

#### Scenario: Ornament export works without any availability guard

- **WHEN** an `OrnamentGenerator` is constructed and used in a normal installation
- **THEN** it generates and saves an STL mesh without consulting any capability flag, and no `HAS_PYVISTA` / `check_pyvista_available` symbol is importable from the library
