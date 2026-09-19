# Packaging

## ADDED Requirements

### Requirement: 3D backend dependency strategy

The package SHALL provide PyVista as the 3D backend through an optional extra in the 2.x
line, exposed under both `pyvista` and a `3d` alias, and SHALL document that PyVista
becomes a required dependency at 3.0.

#### Scenario: PyVista installable via either extra name

- **WHEN** a user installs `complexplorer[3d]` or `complexplorer[pyvista]`
- **THEN** PyVista is installed and the same set of 3D features is enabled

#### Scenario: 3D backend policy is discoverable

- **WHEN** a user reads the installation/backend documentation
- **THEN** it states that matplotlib serves 2D and PyVista serves 3D, that new 3D features
  are PyVista-only, and that PyVista will be required starting at 3.0
