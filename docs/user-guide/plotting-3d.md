# 3D Plotting

Create three-dimensional analytic landscapes to visualize complex functions as surfaces where height represents modulus or other scaling.

## Overview

Complexplorer offers two backends for 3D visualization:

- **Matplotlib**: Good for quick visualizations, works everywhere
- **PyVista**: 15-30x faster, cinema-quality rendering, better for high resolution

!!! tip "PyVista Recommended"
    For best quality, use PyVista via command-line scripts (not Jupyter notebooks). The CLI offers superior antialiasing and interactivity.

## Matplotlib 3D Plotting

### plot_landscape()

Create a 3D surface where the complex plane forms the base and height represents function modulus.

```python
import complexplorer as cp
import numpy as np

f = lambda z: (z**2 - 1) / (z**2 + 1)
domain = cp.Rectangle(4, 4)

# Basic 3D landscape
cp.plot_landscape(domain, f)

# With custom colormap
cmap = cp.Phase(phase_sectors=6, auto_scale_r=True)
cp.plot_landscape(domain, f, cmap=cmap)

# Adjust vertical scaling
cp.plot_landscape(domain, f, cmap=cmap, z_scale=0.5)

# High resolution
cp.plot_landscape(domain, f, cmap=cmap, resolution=400)
```

**Parameters:**

- `domain`: Domain object
- `func`: Complex function
- `cmap`: Colormap (default: Phase)
- `resolution`: Points per axis (default: 300)
- `z_scale`: Vertical scale factor (default: 1.0)
- `figsize`: Figure size (default: (10, 8))
- `elevation`: View elevation angle (default: 30)
- `azimuth`: View azimuth angle (default: -60)
- `title`: Plot title
- `filename`: Save to file
- `show`: Display plot (default: True)

**Returns:** matplotlib 3D axes

### pair_plot_landscape()

Side-by-side 3D landscapes of domain and codomain.

```python
# Compare input and output
cp.pair_plot_landscape(domain, f, cmap=cmap)

# Custom view angles
cp.pair_plot_landscape(domain, f, elevation=20, azimuth=-45)

# Adjust vertical scale
cp.pair_plot_landscape(domain, f, z_scale=0.3)
```

**Use Cases:**

- Understanding function transformations in 3D
- Visualizing how complex plane warps
- Teaching complex analysis
- Creating presentation graphics

## PyVista 3D Plotting

PyVista provides high-performance interactive 3D visualization with much better quality than matplotlib.

### plot_landscape_pv()

High-quality 3D landscape with interactive controls.

```python
# Basic PyVista landscape
cp.plot_landscape_pv(domain, f)

# With custom colormap and resolution
cmap = cp.Phase(phase_sectors=6, auto_scale_r=True)
cp.plot_landscape_pv(domain, f, cmap=cmap, resolution=400)

# Save screenshot
cp.plot_landscape_pv(domain, f, cmap=cmap,
                      filename='landscape.png', show=False)

# Interactive mode
cp.plot_landscape_pv(domain, f, cmap=cmap, window_size=(1200, 900))
```

**Parameters:**

- `domain`: Domain object
- `func`: Complex function
- `cmap`: Colormap
- `resolution`: Points per axis (default: 300)
- `window_size`: Window dimensions (default: (1024, 768))
- `title`: Plot title
- `filename`: Save screenshot
- `show`: Display interactive window (default: True)
- `notebook`: Use notebook backend (default: False)

**Returns:** PyVista plotter object (if show=False)

!!! warning "Notebook Backend"
    The Jupyter notebook backend has severe aliasing issues. For best quality, run PyVista in command-line scripts with `notebook=False` (default).

### pair_plot_landscape_pv()

Side-by-side PyVista landscapes.

```python
# High-quality comparison
cp.pair_plot_landscape_pv(domain, f, cmap=cmap)

# Wide window for side-by-side
cp.pair_plot_landscape_pv(domain, f, window_size=(1600, 800))

# Custom labels
cp.pair_plot_landscape_pv(domain, f, labels=["Input", "Output"])
```

**Parameters:**

- `domain`, `func`, `cmap`: Standard parameters
- `resolution`: Points per axis (default: 300)
- `window_size`: Window size (default: (1400, 700))
- `labels`: List of 2 strings for subplot titles
- `filename`: Save screenshot

## Modulus Scaling Modes

Control how the magnitude creates the topography of your landscape:

```python
# Flat (phase only, no height variation)
cp.plot_landscape(domain, f, modulus_mode='constant')

# Linear scaling (can be unstable near poles)
cp.plot_landscape(domain, f, modulus_mode='linear')

# Arctan (smooth, bounded - recommended)
cp.plot_landscape(domain, f, modulus_mode='arctan')

# Logarithmic (emphasize poles/zeros)
cp.plot_landscape(domain, f, modulus_mode='logarithmic')

# Adaptive (auto-adjust to data range)
cp.plot_landscape(domain, f, modulus_mode='adaptive')
```

### Available Modes

| Mode | Description | Best For |
|------|-------------|----------|
| `constant` | Flat surface (no height) | Phase-only visualization |
| `linear` | Direct |z| mapping | Simple functions |
| `arctan` | Smooth bounded scaling | Most functions (recommended) |
| `logarithmic` | log(1 + |z|) | Functions with poles/zeros |
| `adaptive` | Auto-adjust to range | Unknown function behavior |
| `linear_clamp` | Linear with max cap | Controlled height |
| `power` | |z|^p mapping | Emphasize/de-emphasize features |
| `sigmoid` | S-curve mapping | Smooth transitions |
| `hybrid` | Combination modes | Complex landscapes |

### Custom Modulus Parameters

```python
# Arctan with custom parameters
cp.plot_landscape(domain, f,
                 modulus_mode='arctan',
                 modulus_params={'scale': 2.0})

# Power scaling with custom exponent
cp.plot_landscape(domain, f,
                 modulus_mode='power',
                 modulus_params={'exponent': 0.5})

# Linear with clamping
cp.plot_landscape(domain, f,
                 modulus_mode='linear_clamp',
                 modulus_params={'max_height': 2.0})
```

## Resolution and Performance

### Resolution Guidelines

```python
# Low res for exploration (fast)
cp.plot_landscape(domain, f, resolution=200)

# Medium res for general use
cp.plot_landscape(domain, f, resolution=300)

# High res for quality (slower on matplotlib)
cp.plot_landscape(domain, f, resolution=500)

# Very high res (PyVista recommended)
cp.plot_landscape_pv(domain, f, resolution=800)
```

**Performance Comparison:**

- **Matplotlib 300x300**: ~2-5 seconds
- **Matplotlib 500x500**: ~8-15 seconds
- **PyVista 300x300**: ~0.5-1 second
- **PyVista 800x800**: ~2-4 seconds

!!! tip "Use PyVista for High Resolution"
    PyVista is 15-30x faster and produces better quality. Use it for resolutions above 400x400.

## View Angles and Camera

### Matplotlib View Control

```python
# Adjust view angles
cp.plot_landscape(domain, f, elevation=45, azimuth=-30)

# Top-down view
cp.plot_landscape(domain, f, elevation=90, azimuth=0)

# Side view
cp.plot_landscape(domain, f, elevation=0, azimuth=0)
```

### PyVista Interactive Controls

PyVista provides interactive camera controls:

- **Left mouse**: Rotate view
- **Middle mouse**: Pan
- **Right mouse**: Zoom
- **Scroll wheel**: Zoom in/out
- **Shift + drag**: Pan
- **'r'**: Reset camera
- **'s'**: Save screenshot
- **'q'**: Quit

## Customization

### Vertical Scaling

```python
# Flatten landscape (default z_scale=1.0)
cp.plot_landscape(domain, f, z_scale=0.3)

# Exaggerate height
cp.plot_landscape(domain, f, z_scale=2.0)

# Very flat (almost 2D)
cp.plot_landscape(domain, f, z_scale=0.1)
```

### Figure Size and Window Size

```python
# Matplotlib figure size
cp.plot_landscape(domain, f, figsize=(12, 10))

# PyVista window size
cp.plot_landscape_pv(domain, f, window_size=(1920, 1080))
```

### Titles and Labels

```python
# With title
cp.plot_landscape(domain, f, title="f(z) = z^2 / (z^2 + 1)")

# Pair plot with labels
cp.pair_plot_landscape_pv(domain, f, labels=["Domain", "Codomain"])
```

## Saving 3D Plots

### Matplotlib Saving

```python
# Save as PNG
cp.plot_landscape(domain, f, filename='landscape.png', show=False)

# Save as PDF (vector)
cp.plot_landscape(domain, f, filename='landscape.pdf', show=False)

# High DPI
cp.plot_landscape(domain, f, filename='hires.png',
                 show=False, dpi=300)
```

### PyVista Saving

```python
# Screenshot
cp.plot_landscape_pv(domain, f, filename='landscape.png', show=False)

# High resolution screenshot
cp.plot_landscape_pv(domain, f, resolution=800,
                     filename='hires.png', show=False)

# From interactive window: press 's' to save
```

## Common Patterns

### Exploring Function Landscape

```python
f = lambda z: z**3 - 2*z
domain = cp.Rectangle(3, 3)
cmap = cp.Phase(phase_sectors=6, auto_scale_r=True)

# Quick matplotlib view
cp.plot_landscape(domain, f, cmap=cmap, resolution=200)

# High-quality PyVista
cp.plot_landscape_pv(domain, f, cmap=cmap, resolution=400)
```

### Publication Figure

```python
# PyVista with specific view for publication
cmap = cp.PerceptualPastel(phase_sectors=6, auto_scale_r=True)

cp.plot_landscape_pv(
    domain, f,
    cmap=cmap,
    resolution=600,
    window_size=(1200, 1000),
    filename='publication.png',
    show=False
)
```

### Comparing Modulus Modes

```python
modes = ['constant', 'arctan', 'logarithmic', 'adaptive']

for mode in modes:
    cp.plot_landscape_pv(
        domain, f,
        modulus_mode=mode,
        title=f"Modulus: {mode}",
        filename=f'landscape_{mode}.png',
        show=False
    )
```

## Examples by Function Type

### Polynomial

```python
f = lambda z: z**4 - 1
domain = cp.Disk(radius=1.5)
cmap = cp.Phase(phase_sectors=6, auto_scale_r=True)

cp.plot_landscape_pv(domain, f, cmap=cmap, modulus_mode='arctan')
```

### Rational (with poles)

```python
f = lambda z: 1 / (z**2 + 1)
domain = cp.Rectangle(4, 4)
cmap = cp.Phase(phase_sectors=6, auto_scale_r=True)

# Use logarithmic to handle poles
cp.plot_landscape_pv(domain, f, cmap=cmap, modulus_mode='logarithmic')
```

### Trigonometric

```python
f = lambda z: np.sin(z)
domain = cp.Rectangle(6, 6)
cmap = cp.Phase(phase_sectors=8, auto_scale_r=True)

cp.plot_landscape_pv(domain, f, cmap=cmap, modulus_mode='adaptive')
```

## Tips and Best Practices

1. **Use PyVista for final output** - Much better quality and performance
2. **Start with matplotlib** for quick exploration if PyVista not installed
3. **Use arctan modulus mode** as default - handles most functions well
4. **Adjust z_scale** to flatten or exaggerate features
5. **Use logarithmic mode** for functions with poles
6. **Run PyVista from command line** not notebooks for best quality
7. **Use pair plots** to understand transformations
8. **Save high-res screenshots** for presentations
9. **Experiment with view angles** to find best perspective
10. **Use adaptive mode** when function behavior is unknown

## Troubleshooting

### Matplotlib 3D Too Slow

```python
# Reduce resolution
cp.plot_landscape(domain, f, resolution=200)

# Or switch to PyVista
cp.plot_landscape_pv(domain, f, resolution=300)
```

### Surface Too Spiky

```python
# Use arctan or adaptive mode
cp.plot_landscape(domain, f, modulus_mode='arctan')

# Or flatten with z_scale
cp.plot_landscape(domain, f, z_scale=0.3)
```

### PyVista Not Displaying

```python
# Make sure show=True (default)
cp.plot_landscape_pv(domain, f, show=True)

# Avoid notebook backend
cp.plot_landscape_pv(domain, f, notebook=False)

# Check PyVista installation
# pip install "complexplorer[pyvista]"
```

### Surface Too Flat

```python
# Increase z_scale
cp.plot_landscape(domain, f, z_scale=2.0)

# Use different modulus mode
cp.plot_landscape(domain, f, modulus_mode='linear')
```

## Interactive Workflow

### Exploration Workflow

```python
# 1. Quick 2D view first
cp.plot(domain, f, resolution=300)

# 2. Check landscape with matplotlib
cp.plot_landscape(domain, f, resolution=200)

# 3. Fine-tune with PyVista
cmap = cp.Phase(phase_sectors=6, auto_scale_r=True)
cp.plot_landscape_pv(domain, f, cmap=cmap, resolution=400)

# 4. Try different modulus modes
for mode in ['arctan', 'logarithmic', 'adaptive']:
    cp.plot_landscape_pv(domain, f, cmap=cmap, modulus_mode=mode)

# 5. Save best view
cp.plot_landscape_pv(domain, f, cmap=cmap, modulus_mode='arctan',
                     resolution=600, filename='final.png', show=False)
```

## Advanced: Custom Interactive Scripts

For maximum control, create a Python script:

```python
# landscape_viewer.py
import complexplorer as cp
import numpy as np

f = lambda z: (z**2 - 1) / (z**2 + 1)
domain = cp.Rectangle(4, 4)
cmap = cp.Phase(phase_sectors=6, auto_scale_r=True)

# High-quality interactive visualization
cp.plot_landscape_pv(
    domain, f,
    cmap=cmap,
    resolution=600,
    modulus_mode='arctan',
    window_size=(1920, 1080),
    title="Interactive Complex Landscape"
)
```

Run with: `python landscape_viewer.py`

## Next Steps

- **[Riemann Sphere](riemann.md)** - Visualize on the sphere
- **[2D Plotting](plotting-2d.md)** - Start with 2D phase portraits
- **[Colormaps](colormaps.md)** - Choose visualization colors
- **[Domains](domains.md)** - Select appropriate domains
- **[Quick Start](../getting-started/quickstart.md)** - Basic examples
