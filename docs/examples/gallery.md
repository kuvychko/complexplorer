# Gallery

A visual showcase of Complexplorer's capabilities with code examples.

## Classic Phase Portraits

### Basic Phase Portrait

```python
import complexplorer as cp

f = lambda z: (z**2 - 1) / (z**2 + 1)
domain = cp.Rectangle(4, 4)

cp.plot(domain, f)
```

**Function**: Rational function with zeros at z = +/-1 and poles at z = +/-i

### Enhanced Phase Portrait

```python
cmap = cp.Phase(phase_sectors=6, auto_scale_r=True, scale_radius=0.8)
cp.plot(domain, f, cmap=cmap, resolution=600)
```

**Features**: 6 phase sectors with auto-scaled square cells

### With Modulus Contours

```python
cmap = cp.Phase(phase_sectors=6, r_linear_step=0.5)
cp.plot(domain, f, cmap=cmap, resolution=600)
```

**Features**: Phase sectors + linear modulus rings

## Perceptually Uniform Colormaps

### OklabPhase

```python
f = lambda z: z**3 - 2*z + 1
domain = cp.Rectangle(4, 4)

cmap = cp.OklabPhase(phase_sectors=6, auto_scale_r=True, enhanced=True)
cp.plot(domain, f, cmap=cmap, resolution=600)
```

**Function**: Cubic polynomial with 3 zeros
**Colormap**: Perceptually uniform OKLAB color space

### PerceptualPastel

```python
cmap = cp.PerceptualPastel(phase_sectors=6, auto_scale_r=True)
cp.plot(domain, f, cmap=cmap, resolution=600)
```

**Features**: Elegant pastels, ideal for print and presentations

### Isoluminant

```python
f = lambda z: (z - 1) / (z + 1)
domain = cp.Disk(radius=2)

cmap = cp.Isoluminant(L=0.65, show_contours=True)
cp.plot(domain, f, cmap=cmap, resolution=600)
```

**Function**: Mobius transformation
**Features**: Constant brightness, phase encoded in hue only

### CubehelixPhase

```python
cmap = cp.CubehelixPhase(phase_sectors=6, auto_scale_r=True)
cp.plot(domain, f, cmap=cmap, resolution=600)
```

**Features**: Grayscale-safe, ideal for scientific publications

## Artistic Colormaps

### AnalogousWedge (Ocean)

```python
import numpy as np

f = lambda z: np.sin(z)
domain = cp.Rectangle(6, 6)

cmap = cp.AnalogousWedge(H_center=0.55, H_wedge=0.2, phase_sectors=8)
cp.plot(domain, f, cmap=cmap, resolution=600)
```

**Function**: Sine function
**Colors**: Blue-cyan harmonious range

### AnalogousWedge (Sunset)

```python
cmap = cp.AnalogousWedge(H_center=0.05, H_wedge=0.15, phase_sectors=6)
cp.plot(domain, f, cmap=cmap, resolution=600)
```

**Colors**: Red-orange-yellow warm tones

### DivergingWarmCool

```python
f = lambda z: (z**2 - 1) / (z**2 + 1)
domain = cp.Rectangle(4, 4)

cmap = cp.DivergingWarmCool(phase_sectors=6)
cp.plot(domain, f, cmap=cmap, resolution=600)
```

**Features**: Warm amber for positive phase, cool indigo for negative

### InkPaper

```python
cmap = cp.InkPaper()
cp.plot(domain, f, cmap=cmap, resolution=600)
```

**Features**: Minimalist near-monochrome with subtle phase tints

### EarthTopographic

```python
cmap = cp.EarthTopographic(phase_sectors=8)
cp.plot(domain, f, cmap=cmap, resolution=600)
```

**Features**: Terrain-inspired with earth tones and hillshade effect

### FourQuadrant

```python
cmap = cp.FourQuadrant(phase_sectors=6)
cp.plot(domain, f, cmap=cmap, resolution=600)
```

**Features**: Bauhaus-inspired with 4 color anchors (red, green, cyan, magenta)

## Grid Visualizations

### Chessboard

```python
f = lambda z: z**2
domain = cp.Rectangle(4, 4)

cmap = cp.Chessboard(spacing=0.5)
cp.plot(domain, f, cmap=cmap, resolution=600)
```

**Purpose**: Visualize how the function distorts the Cartesian grid

### PolarChessboard

```python
cmap = cp.PolarChessboard(phase_sectors=12, spacing=0.5)
cp.plot(domain, f, cmap=cmap, resolution=600)
```

**Purpose**: Visualize polar coordinate structure

### LogRings

```python
f = lambda z: (z - 1) / (z**2 + 1)
domain = cp.Disk(radius=3)

cmap = cp.LogRings(log_spacing=0.3)
cp.plot(domain, f, cmap=cmap, resolution=600)
```

**Purpose**: Focus on magnitude structure with logarithmic rings

## 3D Analytic Landscapes

### Matplotlib 3D Landscape

```python
f = lambda z: (z**2 - 1) / (z**2 + 1)
domain = cp.Rectangle(4, 4)
cmap = cp.Phase(phase_sectors=6, auto_scale_r=True)

cp.plot_landscape(domain, f, cmap=cmap, resolution=300, z_scale=0.5)
```

**Features**: Height represents modulus, color represents phase

### PyVista High-Quality Landscape

```python
cp.plot_landscape_pv(domain, f, cmap=cmap, resolution=400,
                      modulus_mode='arctan')
```

**Performance**: 15-30x faster than matplotlib
**Quality**: Superior antialiasing and rendering

### Side-by-Side Comparison

```python
cp.pair_plot_landscape_pv(domain, f, cmap=cmap, resolution=300,
                           labels=["Domain", "Codomain"])
```

**Purpose**: See transformation from input to output in 3D

## Riemann Sphere Visualizations

### Matplotlib Riemann Sphere

```python
f = lambda z: z**2 / (z**2 + 1)
cmap = cp.Phase(phase_sectors=6, auto_scale_r=True)

cp.riemann(f, cmap=cmap, resolution=300)
```

**Features**: Complete view of extended complex plane including infinity

### PyVista Riemann Relief Map

```python
cp.riemann_pv(f, cmap=cmap, modulus_mode='arctan',
              n_theta=80, phase_sectors=60, resolution=600)
```

**Features**:
- Smooth bounded relief (arctan scaling)
- High-quality interactive visualization
- Peaks = poles, valleys = zeros

### Riemann Sphere with Logarithmic Relief

```python
f = lambda z: (z**2 - 1) / (z**2 + 1)
cmap = cp.Phase(phase_sectors=6, auto_scale_r=True)

cp.riemann_pv(f, cmap=cmap, modulus_mode='logarithmic',
              n_theta=100, phase_sectors=80)
```

**Features**: Logarithmic scaling emphasizes poles and zeros

## Function Examples

### Polynomial: Cubic

```python
f = lambda z: z**3 - z
domain = cp.Disk(radius=2)
cmap = cp.Phase(phase_sectors=6, auto_scale_r=True)

cp.plot(domain, f, cmap=cmap, resolution=600)
```

**Properties**: 3 zeros at z = 0, +/-1; zero at infinity

### Polynomial: Quartic

```python
f = lambda z: z**4 - 2*z**2 + 1
domain = cp.Disk(radius=2)
cmap = cp.PerceptualPastel(phase_sectors=6, auto_scale_r=True)

cp.plot(domain, f, cmap=cmap, resolution=600)
```

**Properties**: 4 zeros; symmetric structure

### Rational: Simple Pole

```python
f = lambda z: 1 / (z - 1)
domain = cp.Annulus(inner_radius=0.3, outer_radius=3, center=1)
cmap = cp.Phase(phase_sectors=6, auto_scale_r=True)

cp.plot(domain, f, cmap=cmap, resolution=600)
```

**Properties**: Pole at z = 1 (excluded by annulus)

### Rational: Multiple Poles

```python
f = lambda z: 1 / (z**2 - 1)
domain = cp.Rectangle(4, 4)
cmap = cp.Phase(phase_sectors=6, r_log_base=2.0)

cp.plot(domain, f, cmap=cmap, resolution=600)
```

**Properties**: Poles at z = +/-1; logarithmic contours show magnitude

### Trigonometric: Sine

```python
f = lambda z: np.sin(z)
domain = cp.Rectangle(6, 6)
cmap = cp.Phase(phase_sectors=8, auto_scale_r=True)

cp.plot(domain, f, cmap=cmap, resolution=600)
```

**Properties**: Periodic with period 2*pi; zeros at n*pi

### Trigonometric: Tangent

```python
f = lambda z: np.tan(z)
domain = cp.Rectangle(6, 6)
cmap = cp.Phase(phase_sectors=6, r_log_base=2.0)

cp.plot(domain, f, cmap=cmap, resolution=600)
```

**Properties**: Poles at (n + 1/2)*pi; periodic structure

### Exponential

```python
f = lambda z: np.exp(z)
domain = cp.Rectangle(4, 4)
cmap = cp.Phase(phase_sectors=8, auto_scale_r=True)

cp.plot(domain, f, cmap=cmap, resolution=600)
```

**Properties**: Exponential growth; periodic in imaginary direction

### Mobius Transformation

```python
a = 0.5 + 0.3j
f = lambda z: (z - a) / (1 - np.conj(a) * z)
domain = cp.Disk(radius=1)
cmap = cp.Phase(phase_sectors=6, auto_scale_r=True)

cp.plot(domain, f, cmap=cmap, resolution=600)
```

**Properties**: Maps unit disk to itself; conformal

## Comparison Galleries

### Colormap Comparison

```python
f = lambda z: (z**2 - 1) / (z**2 + 1)
domain = cp.Rectangle(4, 4)

cmaps = [
    ('Phase', cp.Phase(phase_sectors=6, auto_scale_r=True)),
    ('OklabPhase', cp.OklabPhase(phase_sectors=6, auto_scale_r=True)),
    ('PerceptualPastel', cp.PerceptualPastel(phase_sectors=6, auto_scale_r=True)),
    ('Isoluminant', cp.Isoluminant(show_contours=True)),
]

for name, cmap in cmaps:
    cp.plot(domain, f, cmap=cmap, title=f"f(z) - {name}",
           filename=f'comparison_{name}.png', show=False)
```

### Modulus Mode Comparison (3D)

```python
modes = ['constant', 'arctan', 'logarithmic', 'adaptive']

for mode in modes:
    cp.plot_landscape_pv(domain, f, cmap=cmap,
                         modulus_mode=mode,
                         title=f"Landscape: {mode}",
                         filename=f'landscape_{mode}.png',
                         show=False)
```

### Domain Comparison

```python
f = lambda z: z**2
cmap = cp.Phase(phase_sectors=6, auto_scale_r=True)

domains = [
    ('Rectangle', cp.Rectangle(4, 4)),
    ('Disk', cp.Disk(radius=2)),
    ('Annulus', cp.Annulus(0.5, 2)),
]

for name, domain in domains:
    cp.plot(domain, f, cmap=cmap, title=f"f(z) = z^2 on {name}",
           filename=f'domain_{name}.png', show=False)
```

## Creating Gallery Images

### High-Resolution for Web

```python
cp.plot(domain, f, cmap=cmap,
       resolution=800,
       dpi=150,
       figsize=(8, 8),
       filename='gallery_image.png',
       show=False)
```

### Publication Quality

```python
cmap = cp.PerceptualPastel(phase_sectors=6, auto_scale_r=True)

cp.plot(domain, f, cmap=cmap,
       resolution=1200,
       dpi=300,
       figsize=(4, 4),
       title="",
       filename='publication.pdf',
       show=False)
```

### PyVista Screenshot

```python
cp.riemann_pv(f, cmap=cmap,
              modulus_mode='arctan',
              n_theta=120,
              phase_sectors=80,
              resolution=800,
              window_size=(1920, 1080),
              filename='riemann_hires.png',
              show=False)
```

## Batch Processing

Create multiple visualizations programmatically:

```python
import complexplorer as cp
import numpy as np

# Define functions
functions = {
    'polynomial': lambda z: z**3 - 2*z + 1,
    'rational': lambda z: (z**2 - 1) / (z**2 + 1),
    'trigonometric': lambda z: np.sin(z),
    'exponential': lambda z: np.exp(z),
}

# Standard settings
domain = cp.Rectangle(4, 4)
cmap = cp.PerceptualPastel(phase_sectors=6, auto_scale_r=True)

# Generate gallery
for name, f in functions.items():
    # 2D plot
    cp.plot(domain, f, cmap=cmap, resolution=800,
           filename=f'gallery_2d_{name}.png', show=False)

    # 3D landscape
    cp.plot_landscape_pv(domain, f, cmap=cmap,
                         resolution=400,
                         filename=f'gallery_3d_{name}.png',
                         show=False)

    # Riemann sphere
    cp.riemann_pv(f, cmap=cmap, modulus_mode='arctan',
                  filename=f'gallery_riemann_{name}.png',
                  show=False)

    print(f"Generated gallery for {name}")
```

## Tips for Gallery Images

1. **Resolution**: Use 600-800 for web, 1000-1200 for print
2. **DPI**: 150 for web, 300 for print
3. **Colormap**: PerceptualPastel for elegant appearance
4. **Format**: PNG for web, PDF for print
5. **Aspect ratio**: Square (8x8 or 4x4) works best
6. **Phase sectors**: 6 or 8 create clean patterns
7. **Auto-scale**: Use `auto_scale_r=True` for square cells
8. **PyVista**: Run from CLI scripts for best quality
9. **Batch mode**: Set `show=False` to save without display
10. **Consistency**: Use same settings across comparison galleries

## Next Steps

- **[2D Plotting](../user-guide/plotting-2d.md)** - Learn 2D visualization
- **[3D Plotting](../user-guide/plotting-3d.md)** - Create 3D landscapes
- **[Riemann Sphere](../user-guide/riemann.md)** - Explore Riemann visualization
- **[Colormaps](../user-guide/colormaps.md)** - Choose perfect colors
- **[Notebooks](notebooks.md)** - Interactive Jupyter examples
