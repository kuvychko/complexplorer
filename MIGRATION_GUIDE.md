# Migration Guide: Complexplorer v1.x to v2.0

This guide helps you migrate from the legacy API (v1.x) to the new modular API (v2.0).

## Overview

Complexplorer v2.0 introduces a cleaner, more modular API structure. While most functions remain similar, the internal organization has been improved for better maintainability and extensibility.

## Key Changes

### 1. Import Structure

The main import remains the same:
```python
import complexplorer as cp
```

All public API functions are available directly from the `cp` namespace.

### 2. Domain Parameter Names

Some domain constructors have updated parameter names for consistency:

**Annulus**:
```python
# Old
domain = cp.Annulus(radius_inner=0.5, radius_outer=2.0)

# New
domain = cp.Annulus(inner_radius=0.5, outer_radius=2.0)
```

**Rectangle** parameters remain the same:
```python
# Both old and new
domain = cp.Rectangle(re_length=4, im_length=3)
# or positional
domain = cp.Rectangle(4, 3)
```

### 3. Plotting Functions

Most plotting functions have the same interface. Key parameters:

**2D Plots**:
```python
# Resolution parameter is 'n', not 'resolution'
cp.plot(domain, func, n=400)  # Not resolution=400
```

**3D PyVista Plots**:
```python
# For high quality in Jupyter, always use notebook=False
cp.plot_landscape_pv(domain, func, notebook=False, show=True)

# Parameter name changes:
# scaling → modulus_scaling (for riemann_pv)
# n → resolution (for PyVista functions)
```

### 4. STL Export

The STL export API has been simplified:

```python
# Import
from complexplorer.export.stl import OrnamentGenerator

# Old: multiple output files with cut_mode
# New: single output file
generator = OrnamentGenerator(func, resolution=150)
stl_file = generator.generate_ornament(
    output_file='ornament.stl',
    size_mm=80
)
```

### 5. Removed/Deprecated Features

- Legacy `Cmap` base class (use specific colormaps like `Phase`, `Chessboard`)
- `plot_domain_coloring` (use `plot`)
- `pair_plot_domain_coloring` (use `pair_plot`)
- Multiple STL file outputs (now single file)

## Common Migration Patterns

### Basic Visualization
```python
# Old and new are the same
domain = cp.Rectangle(4, 4)
cp.plot(domain, lambda z: z**2)
```

### Enhanced Phase Portrait
```python
# Old and new are the same
cmap = cp.Phase(n_phi=6, auto_scale_r=True)
cp.plot(domain, func, cmap=cmap)
```

### PyVista 3D (High Quality)
```python
# Old
cp.plot_landscape_pv(domain, func, show_orientation=True)

# New (for quality in Jupyter)
cp.plot_landscape_pv(domain, func, notebook=False, show=True)
```

### Domain Composition
```python
# Old and new are the same
union = disk1 | disk2
intersection = rect & disk
difference = rect - hole
```

## Troubleshooting

### Import Errors

If you get import errors, ensure you're using the new API:
```python
# Correct
import complexplorer as cp

# Avoid deep imports unless necessary
# from complexplorer.core.domain import Rectangle  # Usually not needed
```

### Parameter Errors

Check for updated parameter names:
- `radius_inner` → `inner_radius`
- `radius_outer` → `outer_radius`
- `scaling` → `modulus_scaling` (in riemann_pv)

### Quality Issues in Jupyter

Always use `notebook=False` for PyVista functions:
```python
# High quality external window
cp.plot_landscape_pv(domain, func, notebook=False, show=True)
```

## Getting Help

- See `examples/getting_started.ipynb` for basic usage
- See `examples/api_cookbook.ipynb` for common patterns
- Check function docstrings with `help(cp.function_name)`

## Future Compatibility

The v2.0 API is designed to be stable. Future updates will maintain backward compatibility with the v2.0 API.