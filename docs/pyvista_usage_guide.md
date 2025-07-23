# PyVista Usage Guide for Complexplorer

## Overview

Complexplorer includes high-performance 3D visualization functions powered by PyVista. These functions provide 15-30x faster rendering than matplotlib with superior visual quality.

## ⚠️ Critical Quality Note

**For production-quality visualizations, always use PyVista via command-line scripts, not Jupyter notebooks.**

The Jupyter trame backend has severe aliasing and rendering issues that cannot be fixed by increasing resolution. The difference in quality between CLI and Jupyter is dramatic and unacceptable for any serious visualization work.

## Installation

```bash
# Install complexplorer with PyVista support
pip install "complexplorer[pyvista]"

# For interactive matplotlib plots in CLI scripts
pip install "complexplorer[qt]"

# Or install everything
pip install "complexplorer[all]"

# Or install PyVista separately
pip install pyvista>=0.45.0
```

## Available Functions

### 1. `plot_landscape_pv()`
High-performance 3D landscape visualization.

```python
import complexplorer as cp

domain = cp.Rectangle(4, 4)
func = lambda z: (z - 1) / (z**2 + z + 1)

# High-quality CLI usage
cp.plot_landscape_pv(
    domain, func,
    n=250,                    # Resolution
    z_max=10,                 # Control Z-axis scaling
    show_orientation=True,    # Show Re/Im/Z axes
    interactive=True,         # Interactive 3D navigation
    title="My Function"
)
```

### 2. `pair_plot_landscape_pv()`
Side-by-side domain and codomain visualization.

```python
cp.pair_plot_landscape_pv(
    domain, func,
    n=200,
    z_max=10,
    show_orientation=True,
    cmap=cp.Phase(n_phi=12, auto_scale_r=True)  # Auto-scaled enhanced phase
)
```

### 3. `riemann_pv()`
Riemann sphere visualization with modulus scaling.

```python
cp.riemann_pv(
    func,
    n_theta=200,              # Latitude resolution
    n_phi=200,                # Longitude resolution
    scaling='arctan',         # Modulus scaling method
    show_orientation=True,
    show_grid=True           # Latitude/longitude grid
)
```

## Recommended Workflow

### Best Practice: Use the Interactive CLI Demo

```bash
python examples/interactive_demo.py
```

This provides:
- Menu-driven selection of functions, domains, and color schemes
- Full-quality interactive 3D visualization
- Orientation axes (Re/Im/Z) for spatial awareness
- Z-axis scaling control
- No quality degradation

### Creating Custom Scripts

For production work, create standalone Python scripts:

```python
#!/usr/bin/env python3
import complexplorer as cp
import numpy as np

# Define your function
def my_complex_function(z):
    return np.sin(z) / z

# Set up domain
domain = cp.Rectangle(6, 6)

# Create high-quality visualization
cp.plot_landscape_pv(
    domain,
    my_complex_function,
    n=300,
    z_max=20,
    cmap=cp.Phase(n_phi=12, auto_scale_r=True),
    show_orientation=True,
    title="sin(z)/z",
    interactive=True
)
```

Save as `my_visualization.py` and run:
```bash
python my_visualization.py
```

## Export Options

PyVista supports multiple export formats:

```python
# Generate visualization with export
cp.plot_landscape_pv(
    domain, func,
    filename="output.png",    # Static image
    interactive=False
)

# For 3D printing, access the mesh directly
from complexplorer.plots_3d_pyvista import _create_complex_surface
grid, _ = _create_complex_surface(domain, func, n=200)
grid.save("model.stl")  # STL for 3D printing
```

## Modulus Scaling Options (Riemann Sphere)

1. **Constant** (traditional): `scaling='constant'`
2. **Arctan** (smooth compression): `scaling='arctan'`
3. **Logarithmic** (for exponential growth): `scaling='logarithmic'`
4. **Linear clamp** (focus on range): `scaling='linear_clamp'`

## Troubleshooting

### Black screens or shader errors in Jupyter
- Switch to static backend: `pv.set_jupyter_backend('static')`
- Better: Use command-line scripts instead

### Poor quality in Jupyter
- This is a known limitation of the trame backend
- **Solution**: Use CLI scripts for production work

### Performance issues
- Reduce resolution: use `n=100` instead of `n=300`
- Disable anti-aliasing: `anti_aliasing=False`
- Close other 3D applications

## Why CLI Over Jupyter?

1. **Quality**: Native OpenGL rendering without web browser limitations
2. **Performance**: Direct GPU access without WebGL overhead
3. **Features**: All PyVista features work correctly
4. **Reliability**: No shader compatibility issues
5. **Export**: Better quality exports and screenshots

## Conclusion

PyVista provides exceptional 3D visualization capabilities for complex functions, but only when used correctly. Always prioritize command-line usage for any serious visualization work. The `examples/interactive_demo.py` script provides an excellent starting point for exploring these capabilities with full quality.