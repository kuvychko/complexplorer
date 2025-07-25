# STL Export Guide - Complete Meshes (No Cutting!)

This guide explains how to generate 3D-printable STL files from complex function visualizations.

## Overview

The `OrnamentGenerator` creates complete, watertight Riemann sphere meshes. Users can orient and slice the mesh however they want in their 3D printing slicer software.

### Advantages:
- **No "peg" artifacts** from incorrect pole detection
- **Unlimited cutting angles** - slice at any orientation in your slicer
- **Simpler code** - no complex bisection algorithms
- **Better for asymmetric functions** - no forced symmetry
- **Hollow ornaments** - use your slicer's infill settings

## Basic Usage

```python
from complexplorer import Phase, OrnamentGenerator

# Define your complex function
func = lambda z: (z - 1) / (z**2 + z + 1)

# Create generator
generator = OrnamentGenerator(
    func=func,
    resolution=120,
    scaling='arctan',
    scaling_params={'r_min': 0.3, 'r_max': 1.2},
    cmap=Phase(n_phi=12)
)

# Generate complete watertight mesh
filename = generator.generate_ornament(
    output_file='my_ornament.stl',
    size_mm=80,
    smooth=True
)
```

That's it! You get a single STL file that you can:
- Import into any slicer (PrusaSlicer, Cura, etc.)
- Orient however you like
- Cut at any angle or position
- Make hollow with infill settings

## How It Works

### Spherical Shell Clipping
Instead of trying to heal complex, jagged tears from singularities:

1. **Clip to spherical shell** - Creates clean circular boundaries
2. **Cap the boundaries** - Simple radial triangulation
3. **Result** - Watertight mesh ready for printing

```python
# The healing happens automatically with spherical shell clipping
generator.heal_mesh(smooth=True, smooth_iterations=20)
```

## Scaling Methods

Same as v1, but now applied to the complete sphere:

### Arctan Scaling (Recommended)
```python
scaling='arctan'
scaling_params={'r_min': 0.3, 'r_max': 1.0}
```
Smoothly maps all values to a bounded range. Perfect for functions with poles.

### Linear Clamp
```python
scaling='linear_clamp'
scaling_params={'m_max': 10, 'r_min': 0.3, 'r_max': 1.2}
```
Linear up to m_max, then clamped. Good for controlled visualization.

### Logarithmic
```python
scaling='logarithmic'
scaling_params={'base': np.e, 'r_min': 0.3, 'r_max': 1.0}
```
Emphasizes small modulus values. Great for functions with zeros.

### Constant
```python
scaling='constant'
scaling_params={'radius': 1.0}
```
Perfect sphere - shows only phase information.

## Domain Restrictions

Still supported for focusing on specific regions:

```python
from complexplorer import Disk, Annulus

# Avoid infinity
domain = Disk(radius=5)
generator = OrnamentGenerator(func=func, domain=domain)

# Exclude origin for 1/z type functions
domain = Annulus(radius_inner=0.1, radius_outer=3)
generator = OrnamentGenerator(func=lambda z: 1/z, domain=domain)
```

## Step-by-Step Control

For advanced users who want more control:

```python
# 1. Create generator
generator = OrnamentGenerator(func, resolution=150)

# 2. Generate sphere
sphere = generator.generate_sphere(verbose=True)

# 3. Heal with spherical clipping
healed = generator.heal_mesh(smooth=True, verbose=True)

# 4. Validate
generator.validate_printability(verbose=True)

# 5. Export with custom size
generator.export('output.stl', size_mm=100)
```

## Migration from v1

If you have code using the old `OrnamentGenerator`:

```python
# Old way (v1)
from complexplorer import OrnamentGenerator
ornament = OrnamentGenerator(func, resolution=150)
top, bottom = ornament.generate_ornament(cut_mode='real')

# New way (v2)
from complexplorer import OrnamentGenerator
ornament = OrnamentGeneratorV2(func, resolution=150)
complete_file = ornament.generate_ornament()
```

## Slicing in Your 3D Printer Software

### PrusaSlicer
1. Import the STL file
2. Use "Cut" tool (C key)
3. Position cut plane wherever you want
4. Export both parts

### Cura
1. Import the STL
2. Use "Mesh Tools" → "Split model into parts"
3. Position as desired

### Benefits of Slicer Cutting
- Visual preview of cut position
- Multiple cut angles in one session
- Automatic support generation
- Hollow ornaments with vase mode
- Different infill patterns for each half

## Examples

### Simple Pole Function
```python
func = lambda z: 1/z
generator = OrnamentGenerator(
    func=func,
    scaling='arctan',
    scaling_params={'r_min': 0.3, 'r_max': 1.2}
)
generator.generate_ornament('simple_pole.stl')
```

### Multiple Singularities
```python
func = lambda z: 1/(z**4 - 1)  # Singularities at ±1, ±i
generator = OrnamentGenerator(
    func=func,
    scaling='logarithmic',
    scaling_params={'base': 2, 'r_min': 0.25, 'r_max': 1.1}
)
generator.generate_ornament('four_poles.stl')
```

### Identity Function (Previously Problematic)
```python
func = lambda z: z
generator = OrnamentGenerator(
    func=func,
    scaling='arctan',
    scaling_params={'r_min': 0.3, 'r_max': 1.0}
)
generator.generate_ornament('identity_clean.stl')
```

## Tips for Best Results

1. **Resolution**: 100-150 for most functions, 200+ for high detail
2. **Smoothing**: 20-30 iterations usually sufficient
3. **Size**: 60-100mm diameter works well for desktop display
4. **Scaling**: Use arctan for functions with poles, logarithmic for zeros

## Troubleshooting

### File Too Large
- Reduce resolution to 80-120
- Binary STL format is default (smaller files)

### Slicer Reports Non-Manifold
- Usually fine for printing
- Use slicer's repair function if needed

### Want Traditional Halves?
- Use your slicer's cut tool
- Or use the original `OrnamentGenerator` (still available)

## Complete Example Script

```python
#!/usr/bin/env python3
"""Generate a collection of mathematical ornaments."""

from complexplorer import Phase, OrnamentGenerator
import numpy as np

# Collection of interesting functions
functions = [
    ("mobius", lambda z: (z - 1) / (z + 1)),
    ("quadratic", lambda z: z**2),
    ("sine", lambda z: np.sin(z)),
    ("rational", lambda z: (z**2 - 1) / (z**2 + z + 1)),
]

for name, func in functions:
    print(f"Generating {name}...")
    
    generator = OrnamentGenerator(
        func=func,
        resolution=120,
        scaling='arctan',
        scaling_params={'r_min': 0.3, 'r_max': 1.0},
        cmap=Phase(n_phi=12)
    )
    
    generator.generate_ornament(
        output_file=f'{name}_ornament.stl',
        size_mm=80,
        smooth=True
    )

print("All ornaments generated!")
print("Import into your slicer and cut at any angle you like!")
```