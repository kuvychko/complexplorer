# 3D Plotting (PyVista)

## REMOVED Requirements

### Requirement: Optional-dependency gating

**Reason:** PyVista is a **required** core dependency in 3.0, so it is always present. The
"missing PyVista raises a clear error", "library imports without PyVista", and capability-flag
scenarios describe a configuration that can no longer occur. PyVista-backed plotting is simply
available; the `HAS_PYVISTA` flag is removed.
