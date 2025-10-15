# 2D Plotting

Create two-dimensional phase portraits and visualizations of complex functions using matplotlib.

## Basic 2D Plotting

### plot()

The fundamental 2D visualization function. Creates a phase portrait of your complex function.

```python
import complexplorer as cp
import numpy as np

# Define function and domain
f = lambda z: (z**2 - 1) / (z**2 + 1)
domain = cp.Rectangle(4, 4)

# Basic plot
cp.plot(domain, f)

# With custom colormap
cmap = cp.Phase(phase_sectors=6, auto_scale_r=True)
cp.plot(domain, f, cmap=cmap)

# High resolution
cp.plot(domain, f, cmap=cmap, resolution=800)

# Custom figure size
cp.plot(domain, f, cmap=cmap, figsize=(10, 10))
```

**Parameters:**

- `domain`: Domain object (Rectangle, Disk, or Annulus)
- `func`: Complex function z -> f(z)
- `cmap`: Colormap object (default: Phase with 6 sectors)
- `resolution`: Number of points per axis (default: 500)
- `figsize`: Figure size in inches (default: (8, 8))
- `dpi`: Dots per inch for figure (default: 100)
- `title`: Plot title (optional)
- `filename`: Save to file instead of displaying (optional)
- `show`: Whether to display the plot (default: True)

**Returns:** matplotlib axes object

## Side-by-Side Comparisons

### pair_plot()

Visualize both the domain (input) and codomain (output) side-by-side.

```python
f = lambda z: z**2
domain = cp.Rectangle(4, 4)

# Basic pair plot
cp.pair_plot(domain, f)

# With custom colormap
cmap = cp.Phase(phase_sectors=6, auto_scale_r=True)
cp.pair_plot(domain, f, cmap=cmap)

# Custom figure size for wide display
cp.pair_plot(domain, f, cmap=cmap, figsize=(16, 8))

# With labels
cp.pair_plot(domain, f, labels=["Domain", "Codomain"])
```

**Parameters:**

- `domain`: Domain object
- `func`: Complex function
- `cmap`: Colormap (default: Phase with 6 sectors)
- `resolution`: Points per axis (default: 500)
- `figsize`: Figure size (default: (14, 7))
- `labels`: List of 2 strings for subplot titles
- `filename`: Save to file (optional)

**Use Cases:**

- Understanding how functions transform the complex plane
- Seeing input/output relationship visually
- Teaching complex function behavior
- Presentations and lectures

## Resolution and Quality

### Choosing Resolution

```python
# Low res for quick exploration (fast)
cp.plot(domain, f, resolution=200)

# Medium res for general use (default)
cp.plot(domain, f, resolution=500)

# High res for quality visualization
cp.plot(domain, f, resolution=800)

# Very high res for publication
cp.plot(domain, f, resolution=1200)
```

**Resolution Guidelines:**

- **200-300**: Quick exploration, interactive work
- **400-600**: General visualization, presentations
- **800-1000**: High quality, detailed analysis
- **1200+**: Publication quality, large prints

!!! tip "Performance"
    Higher resolution = longer computation time. Start with 400-500 for exploration, then increase for final output.

### DPI for Publication

```python
# Screen display (default)
cp.plot(domain, f, dpi=100)

# High quality screen
cp.plot(domain, f, dpi=150)

# Print quality
cp.plot(domain, f, resolution=800, dpi=300, figsize=(10, 10))
```

## Saving Plots

### Save to File

```python
# PNG (default, good for web)
cp.plot(domain, f, cmap=cmap, filename='output.png')

# High-res PNG
cp.plot(domain, f, cmap=cmap, resolution=1000, dpi=300,
       filename='high_res.png')

# PDF (vector, good for publications)
cp.plot(domain, f, cmap=cmap, filename='output.pdf')

# SVG (vector, good for editing)
cp.plot(domain, f, cmap=cmap, filename='output.svg')

# Save without displaying
cp.plot(domain, f, filename='output.png', show=False)
```

**Format Support:**

- PNG: Raster, web-friendly, transparency support
- PDF: Vector, publication-ready, scalable
- SVG: Vector, editable in Inkscape/Illustrator
- JPG: Raster, compressed (use PNG instead)

## Customization

### Figure Size and Aspect

```python
# Square (default)
cp.plot(domain, f, figsize=(8, 8))

# Wide
cp.plot(domain, f, figsize=(12, 6))

# Tall
cp.plot(domain, f, figsize=(6, 10))

# Publication size (column width)
cp.plot(domain, f, figsize=(3.5, 3.5), dpi=300)
```

### Titles and Labels

```python
# With title
cp.plot(domain, f, title="f(z) = (z^2 - 1)/(z^2 + 1)")

# Pair plot with custom labels
cp.pair_plot(domain, f, labels=["Input Domain", "Output Codomain"])

# No title
cp.plot(domain, f, title="")
```

## Common Patterns

### Exploring a Function

```python
f = lambda z: z**3 - 2*z + 1
domain = cp.Rectangle(4, 4)

# Quick look
cp.plot(domain, f, resolution=300)

# Detailed view with enhanced phase
cmap = cp.Phase(phase_sectors=6, auto_scale_r=True)
cp.plot(domain, f, cmap=cmap, resolution=600)

# Side-by-side comparison
cp.pair_plot(domain, f, cmap=cmap)
```

### Multiple Colormaps

```python
# Compare different colormaps
cmaps = [
    ('Phase', cp.Phase(phase_sectors=6, auto_scale_r=True)),
    ('Pastel', cp.PerceptualPastel(phase_sectors=6, auto_scale_r=True)),
    ('Chessboard', cp.Chessboard(spacing=0.5))
]

for name, cmap in cmaps:
    cp.plot(domain, f, cmap=cmap, title=f"f(z) - {name}",
           filename=f'function_{name}.png', show=False)
```

### Publication Figure

```python
# High-quality publication figure
cmap = cp.PerceptualPastel(phase_sectors=6, auto_scale_r=True)

cp.plot(
    domain, f,
    cmap=cmap,
    resolution=1000,
    figsize=(4, 4),  # Single column width
    dpi=300,
    title="",  # Add in LaTeX document instead
    filename='publication_figure.pdf'
)
```

## Examples by Function Type

### Polynomial

```python
f = lambda z: z**4 - 2*z**2 + 1
domain = cp.Disk(radius=2)
cmap = cp.Phase(phase_sectors=6, auto_scale_r=True)

cp.plot(domain, f, cmap=cmap)
```

### Rational

```python
f = lambda z: (z - 1) / (z + 1)
domain = cp.Disk(radius=2)
cmap = cp.Phase(phase_sectors=6, auto_scale_r=True)

cp.pair_plot(domain, f, cmap=cmap)
```

### Trigonometric

```python
f = lambda z: np.sin(z)
domain = cp.Rectangle(6, 6)
cmap = cp.Phase(phase_sectors=6, auto_scale_r=True)

cp.plot(domain, f, cmap=cmap, resolution=600)
```

### Exponential

```python
f = lambda z: np.exp(z)
domain = cp.Rectangle(4, 4)
cmap = cp.Phase(phase_sectors=8, auto_scale_r=True)

cp.plot(domain, f, cmap=cmap)
```

## Tips and Best Practices

1. **Start with defaults** - `cp.plot(domain, f)` works great for exploration
2. **Use pair_plot()** to understand function behavior visually
3. **Increase resolution gradually** - Start at 300, go up as needed
4. **Match colormap to purpose**:
   - Exploration: `Phase` with auto_scale_r
   - Print: `PerceptualPastel`
   - Scientific: `CubehelixPhase`
   - Grid visualization: `Chessboard` or `PolarChessboard`
5. **Save intermediate results** - Use `filename=` to save without displaying
6. **Use PDF/SVG for publications** - Vector formats scale perfectly
7. **Consider aspect ratio** - Use `figsize` to match your medium
8. **Disable display for batch processing** - Set `show=False`

## Troubleshooting

### Plot Too Slow

```python
# Reduce resolution
cp.plot(domain, f, resolution=300)

# Use smaller domain
domain = cp.Rectangle(2, 2)
```

### Not Enough Detail

```python
# Increase resolution
cp.plot(domain, f, resolution=800)

# Zoom in
domain = cp.Rectangle(1, 1, center=1+1j)
```

### Colors Look Wrong

```python
# Try different colormaps
cp.plot(domain, f, cmap=cp.OklabPhase())
cp.plot(domain, f, cmap=cp.PerceptualPastel())
```

### Saved Image Quality Low

```python
# Increase DPI
cp.plot(domain, f, dpi=300, filename='high_quality.png')

# Use vector format
cp.plot(domain, f, filename='scalable.pdf')
```

## Next Steps

- **[3D Plotting](plotting-3d.md)** - Create 3D analytic landscapes
- **[Riemann Sphere](riemann.md)** - Visualize on the Riemann sphere
- **[Colormaps](colormaps.md)** - Explore all colormap options
- **[Domains](domains.md)** - Choose appropriate domains
- **[Quick Start](../getting-started/quickstart.md)** - Basic examples
