# Export API Reference

Export complex function visualizations as 3D-printable STL files.

## Overview

The export module allows you to create physical ornaments from complex functions by generating STL files suitable for 3D printing. The ornaments are based on the Riemann sphere visualization with modulus-scaled relief.

**Requirements:** PyVista must be installed (`pip install "complexplorer[pyvista]"`)

## STL Export Classes

::: complexplorer.export.stl.ornament_generator
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3
      members:
        - OrnamentGenerator
        - create_ornament

### OrnamentGenerator

Main class for generating 3D-printable ornaments from complex functions.

**Example:**
```python
import complexplorer as cp
from complexplorer.export.stl import OrnamentGenerator

# Define function
f = lambda z: (z**2 - 1) / (z**2 + 1)

# Create generator
gen = OrnamentGenerator(
    func=f,
    resolution=150,
    modulus_mode='arctan',
    cmap=cp.Phase(phase_sectors=6, auto_scale_r=True)
)

# Generate and save
gen.generate_and_save('ornament.stl', size_mm=80)
```

**Parameters:**
- `func`: Complex function f(z) to visualize
- `resolution`: Mesh resolution (default: 150)
  - 100-150: Fast generation, lower quality
  - 150-200: Good quality (recommended)
  - 200-300: High quality (slower)
- `modulus_mode`: Modulus scaling method (default: 'arctan')
  - `'arctan'`: Smooth bounded relief (recommended)
  - `'logarithmic'`: Emphasizes poles and zeros
  - `'adaptive'`: Auto-adjusts to function behavior
  - `'sigmoid'`: S-curve mapping
  - `'linear'`: Direct mapping (can be unstable)
  - `'constant'`: No relief (flat sphere)
- `modulus_params`: Optional dict of scaling parameters
- `cmap`: Colormap for visualization (default: Phase with 6 sectors)
- `domain`: Optional domain restriction to avoid numerical issues

**Note:** The old parameter names `scaling` and `scaling_params` are deprecated but still supported for backwards compatibility.

**Methods:**

#### generate_ornament()

Generate the ornament mesh.

```python
mesh = gen.generate_ornament(verbose=True)
# Returns: PyVista PolyData mesh with RGB colors
```

**Parameters:**
- `verbose`: Print progress information (default: False)

**Returns:** PyVista PolyData mesh with color information

#### validate_mesh()

Validate mesh for 3D printing.

```python
results = gen.validate_mesh(size_mm=50, verbose=True)

print(f"Watertight: {results['is_watertight']}")
print(f"Manifold: {results['is_manifold']}")
print(f"Vertices: {results['n_vertices']}")
print(f"Faces: {results['n_faces']}")
```

**Parameters:**
- `size_mm`: Target size in millimeters (default: 50)
- `verbose`: Print validation results (default: True)

**Returns:** dict with validation results:
- `is_watertight`: Whether mesh has no holes
- `is_manifold`: Whether mesh is manifold
- `n_vertices`: Number of vertices
- `n_faces`: Number of faces
- `dimensions_mm`: Bounding box dimensions
- `volume_mm3`: Mesh volume

#### save_stl()

Save the ornament as an STL file.

```python
filename = gen.save_stl(
    'ornament.stl',
    size_mm=80,
    center=True,
    repair=True,
    binary=True,
    validate=True,
    verbose=True
)
```

**Parameters:**
- `filename`: Output filename (should end with .stl)
- `size_mm`: Scale mesh to this size in mm (default: 50)
- `center`: Center the mesh at origin (default: True)
- `repair`: Apply simple mesh repair (default: True)
- `binary`: Save as binary STL (default: True, smaller file)
- `validate`: Validate before saving (default: True)
- `verbose`: Print progress information (default: True)

**Returns:** Path to saved file

#### generate_and_save()

Generate ornament and save as STL in one step.

```python
filename = gen.generate_and_save(
    'ornament.stl',
    size_mm=80,
    center=True,
    repair=True,
    binary=True,
    validate=True,
    verbose=True
)
```

**Parameters:** Same as save_stl() plus generate_ornament()

**Returns:** Path to saved file

### create_ornament()

Convenience function for creating STL files from complex functions in one line.

**Example:**
```python
from complexplorer.export.stl import create_ornament

# Simple ornament
create_ornament(
    lambda z: z**3 - 2*z + 1,
    'polynomial.stl',
    size_mm=60
)

# With domain restriction
f = lambda z: 1 / (z**2 + 1)
domain = cp.Annulus(inner_radius=0.1, outer_radius=5)

create_ornament(
    f,
    'rational.stl',
    size_mm=80,
    resolution=200,
    modulus_mode='logarithmic',
    domain=domain
)
```

**Parameters:**
- `func`: Complex function to visualize
- `filename`: Output STL filename
- `size_mm`: Size in millimeters (default: 50)
- `resolution`: Mesh resolution (default: 150)
- `modulus_mode`: Modulus scaling method (default: 'arctan')
- `modulus_params`: Optional scaling parameters
- `cmap`: Colormap (default: Phase)
- `domain`: Optional domain restriction
- `verbose`: Print progress (default: True)

**Note:** The old parameter names `scaling` and `scaling_params` are deprecated.

**Returns:** Path to saved STL file

## STL Utilities

::: complexplorer.export.stl.utils
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3
      members:
        - validate_printability
        - scale_to_size
        - center_mesh

### validate_printability()

Check if mesh is suitable for 3D printing.

```python
from complexplorer.export.stl.utils import validate_printability

results = validate_printability(mesh, size_mm=50, verbose=True)
```

**Checks:**
- Watertight (no holes)
- Manifold (proper edge connectivity)
- Vertex and face counts
- Physical dimensions
- Mesh volume

### scale_to_size()

Scale mesh to target size.

```python
from complexplorer.export.stl.utils import scale_to_size

scaled_mesh = scale_to_size(mesh, size_mm=80, axis='max')
```

**Parameters:**
- `mesh`: PyVista mesh
- `size_mm`: Target size in millimeters
- `axis`: Scaling axis ('x', 'y', 'z', 'max', 'min', 'average')

### center_mesh()

Center mesh at origin.

```python
from complexplorer.export.stl.utils import center_mesh

centered_mesh = center_mesh(mesh)
```

## Mesh Repair

::: complexplorer.export.stl.mesh_repair
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3
      members:
        - repair_mesh_simple

### repair_mesh_simple()

Apply simple mesh repair operations.

```python
from complexplorer.export.stl.mesh_repair import repair_mesh_simple

repaired = repair_mesh_simple(
    mesh,
    fill_holes=True,
    remove_duplicates=True,
    verbose=True
)
```

**Operations:**
- Fill holes to make watertight
- Remove duplicate vertices
- Clean degenerate faces

**Note:** For complex repair, consider external tools like MeshLab or Meshmixer.

## Complete Examples

### Basic Ornament

```python
import complexplorer as cp
from complexplorer.export.stl import create_ornament

# Create simple polynomial ornament
f = lambda z: z**3 - z

create_ornament(
    f,
    'polynomial_ornament.stl',
    size_mm=60,
    resolution=150
)
```

### High-Quality Ornament

```python
from complexplorer.export.stl import OrnamentGenerator

f = lambda z: (z**2 - 1) / (z**2 + 1)
cmap = cp.PerceptualPastel(phase_sectors=6, auto_scale_r=True)

gen = OrnamentGenerator(
    func=f,
    resolution=200,
    modulus_mode='arctan',
    modulus_params={'r_min': 0.3, 'r_max': 0.9},
    cmap=cmap
)

gen.generate_ornament(verbose=True)
gen.validate_mesh(size_mm=80)
gen.save_stl('high_quality_ornament.stl', size_mm=80)
```

### Ornament with Domain Restriction

```python
# Function with pole at origin
f = lambda z: 1 / z

# Exclude problematic region
domain = cp.Annulus(inner_radius=0.2, outer_radius=5)

create_ornament(
    f,
    'restricted_ornament.stl',
    size_mm=70,
    resolution=180,
    modulus_mode='logarithmic',
    domain=domain
)
```

### Ornament with Custom Scaling

```python
gen = OrnamentGenerator(
    func=lambda z: z**2 / (z**2 + 1),
    resolution=180,
    modulus_mode='arctan',
    modulus_params={
        'r_min': 0.25,  # Minimum sphere radius
        'r_max': 0.95   # Maximum sphere radius
    }
)

gen.generate_and_save('custom_scaling_ornament.stl', size_mm=75)
```

### Batch Processing

```python
import numpy as np
from complexplorer.export.stl import create_ornament

functions = {
    'polynomial': lambda z: z**4 - 2*z**2 + 1,
    'rational': lambda z: (z**2 - 1) / (z**2 + 1),
    'trigonometric': lambda z: np.sin(z),
    'mobius': lambda z: (z - 0.5) / (1 - 0.5*np.conj(z))
}

for name, f in functions.items():
    filename = f'ornament_{name}.stl'
    create_ornament(
        f,
        filename,
        size_mm=65,
        resolution=160,
        verbose=True
    )
    print(f"Generated {filename}")
```

## Scaling Modes for STL

Different scaling modes create different relief patterns:

| Mode | Description | Best For |
|------|-------------|----------|
| `arctan` | Smooth bounded relief | Most functions (default) |
| `logarithmic` | Emphasizes poles/zeros | Rational functions |
| `adaptive` | Auto-adjusts | Unknown behavior |
| `sigmoid` | S-curve mapping | Smooth transitions |
| `linear` | Direct mapping | Simple polynomials |

**Recommended Settings:**
```python
# General purpose (recommended)
modulus_mode='arctan'
modulus_params={'r_min': 0.3, 'r_max': 0.9}

# Emphasize features
modulus_mode='logarithmic'
modulus_params={'base': np.e, 'r_min': 0.3, 'r_max': 0.9}

# Auto-adjust
modulus_mode='adaptive'
modulus_params={'low_percentile': 5, 'high_percentile': 95}
```

## Resolution Guidelines

Balance between quality and generation time:

| Resolution | Quality | Generation Time | File Size | Use Case |
|-----------|---------|-----------------|-----------|----------|
| 100-120 | Low | ~10-20 sec | ~2-4 MB | Testing |
| 150-180 | Good | ~30-60 sec | ~5-10 MB | Standard prints |
| 200-250 | High | ~2-4 min | ~15-30 MB | Quality prints |
| 300+ | Very High | ~5-10 min | ~50+ MB | Publication |

**Recommendation:** Start with 150, increase to 200 for final prints.

## Printer Compatibility

### File Format
- **Binary STL**: Smaller files (default)
- **ASCII STL**: Human-readable, larger files

### Size Recommendations
- **Small prints**: 40-60mm (desk ornaments)
- **Medium prints**: 60-80mm (display pieces)
- **Large prints**: 80-100mm (statement pieces)

### Material Considerations
- **PLA**: Easy to print, rigid
- **PETG**: More flexible, durable
- **Resin**: Highest detail, brittle
- **TPU**: Flexible, experimental

### Printer Settings
- **Layer height**: 0.1-0.2mm
- **Wall thickness**: 2-3 perimeters
- **Infill**: 15-20% (ornaments are hollow spheres)
- **Supports**: Usually not needed (sphere is self-supporting)

## Troubleshooting

### Mesh Not Watertight

```python
# Enable mesh repair
gen.save_stl('ornament.stl', repair=True)

# Or use stronger repair
from complexplorer.export.stl.mesh_repair import repair_mesh_simple
mesh = gen.generate_ornament()
repaired = repair_mesh_simple(mesh, fill_holes=True, verbose=True)
```

### Numerical Artifacts

```python
# Restrict domain to stable region
domain = cp.Disk(radius=5)

gen = OrnamentGenerator(
    func=f,
    domain=domain,
    modulus_mode='adaptive'  # Auto-adjusts to function
)
```

### Too Spiky or Flat

```python
# Adjust scaling parameters
gen = OrnamentGenerator(
    func=f,
    modulus_mode='arctan',
    modulus_params={
        'r_min': 0.2,  # Lower for more relief
        'r_max': 0.95  # Higher for more variation
    }
)
```

### Large File Size

```python
# Reduce resolution
gen = OrnamentGenerator(func=f, resolution=120)

# Or use binary STL
gen.save_stl('ornament.stl', binary=True)
```

### Slicer Errors

```python
# Validate before printing
results = gen.validate_mesh(size_mm=70, verbose=True)

if not results['is_watertight']:
    # Apply repair
    gen.save_stl('ornament.stl', repair=True)
```

## Advanced: Post-Processing

For complex mesh issues, use external tools:

1. **MeshLab**: Free, powerful mesh editing
2. **Meshmixer**: Autodesk's free mesh repair tool
3. **Blender**: Full 3D modeling suite

**Workflow:**
```python
# 1. Generate mesh
gen.generate_and_save('raw_ornament.stl', repair=False)

# 2. Open in MeshLab/Meshmixer
# 3. Apply filters: remove duplicates, fill holes, remesh
# 4. Export as repaired STL

# 5. Slice and print
```

## Tips for Best Results

1. **Start simple**: Test with low resolution first
2. **Domain restriction**: Exclude singularities for cleaner meshes
3. **Scaling choice**: Use arctan for most functions
4. **Mesh repair**: Enable by default unless you know it's clean
5. **Validation**: Always validate before printing
6. **Size appropriately**: 60-80mm is ideal for desk ornaments
7. **Binary STL**: Smaller files, faster slicing
8. **Test print**: Do a small test before full-size print
9. **Material choice**: PLA is easiest for beginners
10. **Post-processing**: Sand and paint for best appearance

## See Also

- **[Core API](core.md)** - Domains and colormaps for ornaments
- **[Plotting API](plotting.md)** - Visualize before exporting
- **[Riemann Sphere Guide](../user-guide/riemann.md)** - Understanding the geometry
- **[Gallery](../examples/gallery.md)** - Visual examples
