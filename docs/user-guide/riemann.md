# Riemann Sphere Visualization

Visualize complex functions on the Riemann sphere, revealing behavior at infinity and providing a complete view of the extended complex plane.

## Understanding the Riemann Sphere

The Riemann sphere is a way to compactify the complex plane by adding a "point at infinity". This allows visualization of:

- **Behavior at infinity**: How functions behave as |z| -> 
- **Complete topology**: The entire extended complex plane  * {}
- **Poles and zeros**: Their relationship to infinity
- **Global structure**: Overall function behavior in one view

The complex plane is mapped to a sphere via stereographic projection:

- **South pole**: z = 0 (origin)
- **Equator**: |z| = 1 (unit circle)
- **North pole**: z =  (point at infinity)

## Matplotlib Riemann Plots

### riemann()

Create a Riemann sphere visualization using matplotlib 3D.

```python
import complexplorer as cp
import numpy as np

f = lambda z: z**2 / (z**2 + 1)

# Basic Riemann sphere
cp.riemann(f)

# With custom colormap
cmap = cp.Phase(phase_sectors=6, auto_scale_r=True)
cp.riemann(f, cmap=cmap)

# High resolution
cp.riemann(f, cmap=cmap, resolution=400)

# Adjust view angle
cp.riemann(f, cmap=cmap, elevation=20, azimuth=45)
```

**Parameters:**

- `func`: Complex function
- `cmap`: Colormap (default: Phase)
- `resolution`: Points per dimension (default: 300)
- `n_theta`: Latitudinal divisions (default: 50)
- `phase_sectors`: Longitudinal divisions (default: 50)
- `figsize`: Figure size (default: (10, 10))
- `elevation`, `azimuth`: View angles
- `title`: Plot title
- `filename`: Save to file

**Returns:** matplotlib 3D axes

## PyVista Riemann Relief Maps

PyVista provides stunning Riemann relief maps where modulus creates topographic "relief".

### riemann_pv()

High-quality interactive Riemann sphere with modulus scaling options.

```python
# Basic Riemann relief map
cp.riemann_pv(f)

# With custom colormap
cmap = cp.Phase(phase_sectors=6, auto_scale_r=True)
cp.riemann_pv(f, cmap=cmap)

# Smooth bounded relief (recommended)
cp.riemann_pv(f, modulus_mode='arctan')

# High resolution
cp.riemann_pv(f, cmap=cmap, modulus_mode='arctan',
             n_theta=100, phase_sectors=60, resolution=600)
```

**Parameters:**

- `func`: Complex function
- `cmap`: Colormap
- `modulus_mode`: Height scaling mode (default: 'arctan')
- `modulus_params`: Parameters for modulus scaling
- `n_theta`: Latitudinal resolution (default: 50)
- `phase_sectors`: Longitudinal resolution (default: 50)
- `resolution`: Function sampling resolution (default: 400)
- `window_size`: Window dimensions (default: (1024, 768))
- `title`: Plot title
- `filename`: Save screenshot
- `show`: Display window (default: True)
- `project_from_north`: Projection pole (default: True)

**Returns:** PyVista plotter (if show=False)

!!! tip "Cinema Quality"
    For best quality Riemann relief maps, use PyVista via command-line scripts (not Jupyter). The CLI offers superior antialiasing and interactivity.

## Modulus Scaling for Relief Maps

The key to beautiful Riemann relief maps is modulus scaling, which controls how the magnitude creates topography.

### Scaling Modes

```python
# Flat (phase only, no relief)
cp.riemann_pv(f, modulus_mode='constant')

# Smooth bounded relief (recommended)
cp.riemann_pv(f, modulus_mode='arctan')

# Emphasize poles and zeros
cp.riemann_pv(f, modulus_mode='logarithmic')

# Auto-adjust to function
cp.riemann_pv(f, modulus_mode='adaptive')

# Linear (can be unstable)
cp.riemann_pv(f, modulus_mode='linear')
```

### Recommended Modes

| Mode | Description | Best For |
|------|-------------|----------|
| `arctan` | Smooth, bounded | Most functions (recommended) |
| `logarithmic` | Emphasize poles/zeros | Rational functions |
| `adaptive` | Auto-adjust | Unknown behavior |
| `constant` | Flat sphere | Pure phase structure |
| `sigmoid` | S-curve mapping | Smooth transitions |

### Custom Relief Parameters

```python
# Arctan with custom scaling
cp.riemann_pv(f,
             modulus_mode='arctan',
             modulus_params={'r_min': 0.3, 'r_max': 0.9})

# Logarithmic with custom base
cp.riemann_pv(f,
             modulus_mode='logarithmic',
             modulus_params={'log_base': 2.0})

# Adaptive with custom percentiles
cp.riemann_pv(f,
             modulus_mode='adaptive',
             modulus_params={'percentile_low': 5, 'percentile_high': 95})
```

## Domain Restrictions

Restrict the visualization to specific regions of the complex plane:

```python
# Avoid infinity at large distances
domain = cp.Disk(radius=5, center=0)
cp.riemann_pv(f, domain=domain, modulus_mode='arctan')

# Exclude origin for functions with poles
domain = cp.Annulus(inner_radius=0.1, outer_radius=10, center=0)
cp.riemann_pv(f, domain=domain, modulus_mode='logarithmic')

# Focus on specific region
domain = cp.Rectangle(4, 4, center=2+1j)
cp.riemann_pv(f, domain=domain, modulus_mode='arctan')
```

**Use Cases:**

- Functions with essential singularities
- Improving numerical stability
- Focusing on interesting regions
- Creating cleaner visualizations

## Resolution Guidelines

### Sphere Resolution

```python
# Low res for exploration
cp.riemann_pv(f, n_theta=30, phase_sectors=30)

# Medium res for general use
cp.riemann_pv(f, n_theta=50, phase_sectors=50)

# High res for quality
cp.riemann_pv(f, n_theta=100, phase_sectors=60)

# Very high res for publication
cp.riemann_pv(f, n_theta=150, phase_sectors=90, resolution=800)
```

**Parameters:**

- `n_theta`: Latitudinal divisions (bands from pole to pole)
- `phase_sectors`: Longitudinal divisions (sectors around equator)
- `resolution`: Function evaluation grid

**Guidelines:**

- **n_theta 30-50**: Exploration (fast)
- **n_theta 60-80**: Presentation quality
- **n_theta 100+**: Publication quality
- **phase_sectors**: Usually 50-90 sufficient
- **resolution**: 400-800 for function sampling

## Projection Direction

Choose whether to project from north or south pole:

```python
# Standard (from north pole)
cp.riemann_pv(f, project_from_north=True)

# From south pole
cp.riemann_pv(f, project_from_north=False)
```

**Effect:**

- **From north**: South pole = origin, north pole = infinity
- **From south**: North pole = origin, south pole = infinity

Usually north projection is preferred (standard convention).

## Common Patterns

### Exploring a Function

```python
f = lambda z: (z**2 - 1) / (z**2 + 1)

# Quick matplotlib view
cp.riemann(f, resolution=200)

# High-quality PyVista relief map
cmap = cp.Phase(phase_sectors=6, auto_scale_r=True)
cp.riemann_pv(f, cmap=cmap, modulus_mode='arctan',
             n_theta=80, phase_sectors=60)
```

### Publication Figure

```python
# High-quality Riemann relief map
cmap = cp.PerceptualPastel(phase_sectors=6, auto_scale_r=True)

cp.riemann_pv(
    f,
    cmap=cmap,
    modulus_mode='arctan',
    n_theta=120,
    phase_sectors=80,
    resolution=800,
    window_size=(1920, 1080),
    filename='riemann_relief.png',
    show=False
)
```

### Comparing Modulus Modes

```python
modes = ['constant', 'arctan', 'logarithmic', 'adaptive']

for mode in modes:
    cp.riemann_pv(
        f,
        modulus_mode=mode,
        title=f"Riemann Relief: {mode}",
        filename=f'riemann_{mode}.png',
        show=False
    )
```

## Examples by Function Type

### Polynomial

```python
f = lambda z: z**3 - z
cmap = cp.Phase(phase_sectors=6, auto_scale_r=True)

cp.riemann_pv(f, cmap=cmap, modulus_mode='arctan')
```

### Rational (poles and zeros)

```python
f = lambda z: (z**2 - 1) / (z**2 + 1)
cmap = cp.Phase(phase_sectors=6, auto_scale_r=True)

# Logarithmic mode emphasizes poles/zeros
cp.riemann_pv(f, cmap=cmap, modulus_mode='logarithmic')
```

### Trigonometric

```python
f = lambda z: np.sin(z)
cmap = cp.Phase(phase_sectors=8, auto_scale_r=True)

# Adaptive handles periodic behavior
cp.riemann_pv(f, cmap=cmap, modulus_mode='adaptive')
```

### Mobius Transformation

```python
a = 0.5 + 0.3j
f = lambda z: (z - a) / (1 - np.conj(a) * z)
cmap = cp.Phase(phase_sectors=6, auto_scale_r=True)

# Arctan for smooth bounded relief
cp.riemann_pv(f, cmap=cmap, modulus_mode='arctan')
```

### Essential Singularity

```python
def exp_1_over_z(z):
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.exp(1/z)
    return np.where(np.isfinite(result), result, 0)

# Use domain to exclude problematic region
domain = cp.Annulus(0.1, 5)
cp.riemann_pv(exp_1_over_z, domain=domain, modulus_mode='logarithmic')
```

## Tips and Best Practices

1. **Use PyVista for Riemann spheres** - Much better quality than matplotlib
2. **Start with arctan mode** - Works well for most functions
3. **Use logarithmic for rational functions** - Emphasizes poles/zeros
4. **Restrict domain for singularities** - Improves stability and appearance
5. **Run from command line** - Better quality than notebooks
6. **Use high resolution** - Riemann spheres benefit from detail
7. **Try different view angles** - Rotate interactively to explore
8. **Save high-res screenshots** - Press 's' in interactive window
9. **Use adaptive mode** - When function behavior is unknown
10. **Match colormap to purpose** - PerceptualPastel for presentations

## Understanding Riemann Relief Maps

A Riemann relief map is like a topographic map of your complex function:

- **Peaks (mountains)**: Poles (where function -> )
- **Valleys**: Zeros (where function = 0)
- **Contour lines**: Phase sectors and modulus rings
- **Colors**: Phase (argument) of the function
- **Height**: Modulus (magnitude) scaled appropriately

The relief reveals the "landscape" of your function's behavior across the entire extended complex plane.

## Troubleshooting

### Too Spiky or Flat

```python
# Try different modulus modes
cp.riemann_pv(f, modulus_mode='arctan')  # Smooth
cp.riemann_pv(f, modulus_mode='logarithmic')  # More variation

# Adjust parameters
cp.riemann_pv(f, modulus_mode='arctan',
             modulus_params={'r_min': 0.2, 'r_max': 0.95})
```

### Numerical Artifacts

```python
# Restrict domain to stable region
domain = cp.Disk(radius=10)
cp.riemann_pv(f, domain=domain)

# Use adaptive mode
cp.riemann_pv(f, modulus_mode='adaptive')
```

### Low Quality Display

```python
# Increase resolution
cp.riemann_pv(f, n_theta=100, phase_sectors=80, resolution=600)

# Run from command line (not notebook)
# Save script as riemann_viewer.py and run: python riemann_viewer.py

# Ensure notebook=False (default)
cp.riemann_pv(f, notebook=False)
```

### PyVista Not Working

```python
# Install PyVista
# pip install "complexplorer[pyvista]"

# Fall back to matplotlib
cp.riemann(f)
```

## Interactive Workflow

```python
# 1. Quick 2D view
cp.plot(cp.Rectangle(4, 4), f)

# 2. Check Riemann sphere with matplotlib
cp.riemann(f, resolution=200)

# 3. High-quality PyVista relief map
cmap = cp.Phase(phase_sectors=6, auto_scale_r=True)
cp.riemann_pv(f, cmap=cmap, modulus_mode='arctan')

# 4. Try different modes
for mode in ['arctan', 'logarithmic', 'adaptive']:
    cp.riemann_pv(f, cmap=cmap, modulus_mode=mode)

# 5. Save best version
cp.riemann_pv(f, cmap=cmap, modulus_mode='arctan',
             n_theta=120, phase_sectors=80, resolution=800,
             filename='riemann_final.png', show=False)
```

## Mathematical Background

The stereographic projection maps the complex plane to a sphere:

**From plane to sphere (north pole projection):**
```
x = 2*Re(z) / (|z|^2 + 1)
y = 2*Im(z) / (|z|^2 + 1)
z_coord = (|z|^2 - 1) / (|z|^2 + 1)
```

**From sphere to plane:**
```
w = (x + iy) / (1 - z_coord)
```

This provides a bijection between  * {} and the sphere S^2.

## Next Steps

- **[3D Plotting](plotting-3d.md)** - Create 3D analytic landscapes
- **[2D Plotting](plotting-2d.md)** - Start with 2D phase portraits
- **[Colormaps](colormaps.md)** - Choose visualization colors
- **[Domains](domains.md)** - Domain restrictions for Riemann sphere
- **[Quick Start](../getting-started/quickstart.md)** - Basic examples
