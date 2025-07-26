# STL Export Module

This module provides functionality for exporting complex function visualizations as STL files for 3D printing.

## Architecture

The module consists of four main components:

### 1. `ornament_generator.py`
Main STL generation functionality:
- `OrnamentGenerator` class - Creates 3D-printable ornaments from complex functions
- `create_ornament()` - Convenience function for one-step STL creation
- Uses the same mesh distortion logic as Riemann sphere visualization
- Applies ModulusScaling to map function moduli to physical radii

### 2. `utils.py`
General utilities for STL processing:
- `validate_printability()` - Check if mesh is suitable for 3D printing
- `scale_to_size()` - Scale mesh to target size in millimeters
- `center_mesh()` - Center mesh at origin
- `check_pyvista_available()` - Verify PyVista installation

### 3. `mesh_repair.py`
Simple mesh repair functions:
- `repair_mesh_simple()` - Basic mesh cleanup and optional hole filling
- `close_mesh_holes()` - Attempt to fill gaps (note: Riemann sphere naturally has small polar gaps)
- `ensure_consistent_normals()` - Ensure face normals point outward

### 4. `__init__.py`
Module interface exposing all public functions

## Design Philosophy

1. **Simplicity**: Direct use of modulus-scaled meshes without complex healing
2. **Consistency**: Same mesh distortion as used in visualization
3. **Practicality**: Accept that rectangular Riemann sphere parameterization has small polar gaps
4. **Flexibility**: Optional simple repairs, but not required

## Usage Example

```python
from complexplorer.export.stl import create_ornament

# Define complex function
func = lambda z: (z**2 - 1) / (z**2 + 1)

# Create STL file
create_ornament(
    func,
    "my_ornament.stl",
    size_mm=50,
    resolution=150,
    scaling='arctan'
)
```

## Technical Notes

- The rectangular (UV) parameterization of the Riemann sphere naturally has small gaps at the poles
- These gaps are typically not problematic for 3D printing
- ModulusScaling parameters can be adjusted for printability (tighter r_min/r_max bounds)
- Simple mesh repair is available but often unnecessary