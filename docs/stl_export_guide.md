# STL Export Guide for 3D Printing

This guide explains how to generate 3D-printable STL files from complex function visualizations using complexplorer's STL export functionality.

## Overview

The STL export module allows you to transform complex function visualizations on the Riemann sphere into physical 3D objects. This is perfect for creating educational models, mathematical art, or unique decorative items.

## Key Features

- **Flat Bisection Planes**: Ensures perfectly flat cutting surfaces for optimal 3D printing
- **Automatic Mesh Healing**: Fixes common mesh defects like holes and non-manifold edges
- **Multiple Scaling Methods**: Control how the modulus affects the 3D shape
- **Colormap Support**: All complexplorer colormaps work with STL generation
- **Print-Ready Output**: Generates watertight meshes with flat bottoms for bed adhesion
- **Empty Layer Prevention**: Advanced healing algorithm specifically targets gaps between layers
- **Spike Elimination**: Intelligent smoothing removes artifacts while preserving detail

## Basic Usage

```python
import complexplorer as cp
from complexplorer.stl_export import OrnamentGeneratorV2

# Define your complex function
func = lambda z: (z - 1) / (z**2 + z + 1)

# Create ornament generator
ornament = OrnamentGeneratorV2(
    func=func,
    resolution=150,  # Higher = more detail
    scaling='arctan',  # Modulus scaling method
    cmap=cp.Phase(n_phi=12, auto_scale_r=True)
)

# Generate STL files
top_file, bottom_file = ornament.generate_ornament(
    cut_mode='real',  # Cut along real axis
    size_mm=80,       # Diameter in millimeters
    smooth=True,      # Apply smoothing
    output_prefix='my_ornament'
)
```

## Scaling Methods

The `scaling` parameter controls how the modulus (magnitude) of complex values affects the 3D shape:

### 1. Constant Scaling
```python
scaling='constant'
scaling_params={'radius': 1.0}
```
All points are projected to a sphere of fixed radius. Good for visualizing phase only.

### 2. Arctan Scaling
```python
scaling='arctan'
scaling_params={'r_min': 0.2, 'r_max': 1.0}
```
Uses arctangent to smoothly map all modulus values to a bounded range. Great for functions with poles.

### 3. Logarithmic Scaling
```python
scaling='logarithmic'
scaling_params={'base': np.e, 'r_min': 0.2, 'r_max': 1.0}
```
Emphasizes differences in small modulus values. Useful for functions with zeros.

### 4. Linear Clamp Scaling
```python
scaling='linear_clamp'
scaling_params={'m_max': 10, 'r_min': 0.2, 'r_max': 1.0}
```
Linear scaling up to m_max, then clamped. Good for controlled visualization range.

## Cutting Modes

The ornament is cut into two halves for easier printing:

- `'real'`: Cut along the real axis (y=0)
- `'imaginary'`: Cut along the imaginary axis (x=0)
- `'angle:degrees'`: Cut at a specified angle (e.g., `'angle:45'`)

## Advanced Options

### Custom Resolution
```python
# Higher resolution for detailed functions
ornament = OrnamentGenerator(func=func, resolution=300)

# Lower resolution for faster generation
ornament = OrnamentGenerator(func=func, resolution=80)
```

### Smoothing Control
```python
# Generate sphere and heal mesh separately
sphere = ornament.generate_sphere(verbose=True)
healed = ornament.heal_mesh(
    smooth=True,
    smooth_iterations=30,  # More iterations = smoother
    verbose=True
)
```

### Manual Processing
```python
# Step-by-step generation for full control
ornament = OrnamentGenerator(func, resolution=200)

# 1. Generate Riemann sphere
sphere = ornament.generate_sphere()

# 2. Heal mesh defects
healed = ornament.heal_mesh(smooth=True)

# 3. Cut into halves
top, bottom = ornament.cut(mode='real')

# 4. Validate printability
ornament.validate_printability(top)
ornament.validate_printability(bottom)

# 5. Export with custom size
ornament.export('top.stl', top, size_mm=100)
ornament.export('bottom.stl', bottom, size_mm=100)
```

## Colormap Examples

### Enhanced Phase Portrait
```python
cmap = cp.Phase(n_phi=12, auto_scale_r=True)
```

### Chessboard Pattern
```python
cmap = cp.Chessboard(spacing=1.0)
```

### Polar Chessboard
```python
cmap = cp.PolarChessboard(n_phi=8, spacing=0.5)
```

### Logarithmic Rings
```python
cmap = cp.LogRings(base=2)
```

## 3D Printing Tips

1. **No Supports Needed**: The flat bottom ensures good bed adhesion
2. **Recommended Settings**:
   - Layer height: 0.15-0.2mm
   - Infill: 20-30% for decorative use
   - Print speed: Standard (50-60 mm/s)
   
3. **Assembly**: Print both halves and glue together with cyanoacrylate or epoxy
4. **Finishing**: Sand the seam and optionally paint to highlight features

## Troubleshooting

### Gaps in Mesh
If your slicer reports gaps:
- Increase resolution (200-300)
- Enable smoothing
- Use mesh repair in your slicer

### Large File Size
- Reduce resolution (100-150)
- Use binary STL format (default)

### Poor Detail
- Increase resolution
- Adjust scaling parameters
- Choose appropriate colormap

## Example Functions

### Rational Functions
```python
# Simple poles and zeros
f1 = lambda z: (z - 1) / (z + 1)
f2 = lambda z: 1 / (z**2 + 1)
f3 = lambda z: (z**2 - 1) / (z**2 + z + 1)
```

### Transcendental Functions  
```python
# Interesting periodic behavior
f4 = lambda z: np.sin(z)
f5 = lambda z: np.exp(z)
f6 = lambda z: np.log(z + 1)
```

### Polynomials
```python
# Multiple roots create symmetry
f7 = lambda z: z**3 - 1
f8 = lambda z: z**4 + z**2 + 1
```

## Complete Example

See `examples/stl_ornament_examples.py` for a comprehensive set of examples generating various ornaments with different functions and settings.