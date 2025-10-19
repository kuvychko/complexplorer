# Quick Start Guide

Get started with Complexplorer in just a few minutes! This guide will walk you through creating your first complex function visualizations.

## Your First Visualization

Let's start with the simplest possible example:

```python
import complexplorer as cp

# Define a complex function
f = lambda z: z**2

# Visualize it using the show() function
cp.show(f)
```

This will display an interactive window showing the phase portrait of f(z) = z^2. The colors represent the phase (argument) of the complex output, while brightness variations show the modulus.

## Understanding the Visualization

In the phase portrait:

- **Hue (color)**: Represents the argument/phase of f(z)
  - Red: arg(f) = 0 (positive real axis)
  - Yellow: arg(f) = π/3
  - Green: arg(f) = 2π/3
  - Cyan: arg(f) = π (negative real axis)
  - Blue: arg(f) = 4π/3
  - Magenta: arg(f) = 5π/3

- **Brightness**: Represents the modulus |f(z)|
  - Bright regions: large magnitude
  - Dark regions: small magnitude

- **Black points**: Zeros of the function
- **White points**: Poles (infinities)

## Customizing the Domain

By default, `show()` uses a rectangular domain from -2 to 2 in both real and imaginary directions. You can customize this:

```python
# Larger domain with custom resolution
cp.show(lambda z: z**2, x_range=(-5, 5, 800), y_range=(-5, 5, 800))

# Different x and y ranges
cp.show(lambda z: z**3 - z, x_range=(-3, 3), y_range=(-2, 2))
```

## Using Explicit Domains

For more control, use explicit domain objects:

```python
# Rectangular domain
domain = cp.Rectangle(re_length=4, im_length=4, center=0+0j)
f = lambda z: (z - 1) / (z + 1)
cp.plot(domain, f)

# Disk domain
domain = cp.Disk(radius=2, center=0)
cp.plot(domain, f)

# Annular domain (ring)
domain = cp.Annulus(inner_radius=0.5, outer_radius=2, center=1j)
cp.plot(domain, f)
```

## Exploring Colormaps

Complexplorer offers many colormaps beyond the basic phase portrait:

### Enhanced Phase Portraits

Automatically create square cells in the complex plane:

```python
from complexplorer import Phase

f = lambda z: (z**2 - 1) / (z**2 + 1)
domain = cp.Rectangle(4, 4)

# Enhanced phase with 6 sectors and auto-scaled modulus rings
cmap = Phase(phase_sectors=6, auto_scale_r=True, scale_radius=0.8)
cp.plot(domain, f, cmap=cmap)
```

### Chessboard Patterns

Visualize both real/imaginary or modulus/phase using grid patterns:

```python
# Cartesian chessboard (rectangular grid)
cmap = cp.Chessboard(spacing=1.0)
cp.plot(domain, f, cmap=cmap)

# Polar chessboard (circular grid)
cmap = cp.PolarChessboard(phase_sectors=12, r_linear_step=0.5)
cp.plot(domain, f, cmap=cmap)
```

### Perceptually Optimized Colormaps

Use modern color science for better print quality:

```python
# Elegant pastels with uniform brightness
cmap = cp.PerceptualPastel(L_center=0.55, C=0.1)
cp.plot(domain, f, cmap=cmap)

# Constant luminance for pure phase visualization
cmap = cp.Isoluminant(L=0.6, C=0.15)
cp.plot(domain, f, cmap=cmap)

# Scientific coloring with grayscale compatibility
cmap = cp.CubehelixPhase(start=0.5, rotations=1.5)
cp.plot(domain, f, cmap=cmap)
```

## 3D Visualizations

Complexplorer can create stunning 3D analytic landscapes:

```python
# Create a 3D landscape using matplotlib
f = lambda z: (z - 1) / (z**2 + z + 1)
domain = cp.Rectangle(4, 4)
cmap = Phase(phase_sectors=6, auto_scale_r=True)

cp.plot_landscape(domain, f, cmap=cmap)
```

### High-Performance PyVista 3D

For much better quality and performance (15-30x faster):

```python
# Requires: pip install "complexplorer[pyvista]"

# Interactive 3D landscape
cp.plot_landscape_pv(domain, f, cmap=cmap, resolution=400)

# Side-by-side domain and codomain
cp.pair_plot_landscape_pv(domain, f, cmap=cmap)
```

!!! tip "PyVista in Scripts, Not Notebooks"
    For best quality, use PyVista in command-line scripts rather than Jupyter notebooks. The notebook backend has aliasing issues. Try `python examples/interactive_showcase.py` for the optimal experience!

## Riemann Sphere Visualizations

Visualize your function on the Riemann sphere to see behavior at infinity:

```python
# Matplotlib-based Riemann sphere
f = lambda z: z**2 / (z**2 + 1)
cp.riemann(f, resolution=400, cmap=cmap)

# High-quality PyVista Riemann relief map (recommended)
cp.riemann_pv(f,
             modulus_mode='arctan',  # Creates beautiful relief
             n_theta=100,
             phase_sectors=60,
             resolution=400)
```

### Modulus Scaling for Relief Maps

Control how the magnitude creates topography:

```python
# Different scaling modes
cp.riemann_pv(f, modulus_mode='constant')      # Flat (phase only)
cp.riemann_pv(f, modulus_mode='arctan')        # Smooth bounded (recommended)
cp.riemann_pv(f, modulus_mode='logarithmic')   # Emphasize poles/zeros
cp.riemann_pv(f, modulus_mode='adaptive')      # Auto-adjust to data
```

## Creating Mathematical Ornaments (STL Export)

Transform your functions into 3D-printable art:

```python
from complexplorer.export.stl import OrnamentGenerator

f = lambda z: (z**2 - 1) / (z**2 + 1)

# Create ornament generator
ornament = OrnamentGenerator(
    func=f,
    resolution=150,
    scaling='arctan',  # Creates smooth relief
    cmap=Phase(phase_sectors=12, auto_scale_r=True)
)

# Generate STL file
ornament.generate_and_save('my_ornament.stl', size_mm=80)
```

The resulting STL file can be directly 3D printed!

## Common Patterns

### Preset Configurations

Use built-in presets for common scenarios:

```python
# High-resolution publication quality
settings = cp.publication_preset()
cp.plot(domain, f, **settings)

# Fast interactive exploration
settings = cp.interactive_preset()
cp.plot(domain, f, **settings)

# Maximum contrast for presentations
settings = cp.high_contrast_preset()
cp.plot(domain, f, **settings)
```

### Exploring Interesting Functions

Here are some fascinating functions to try:

```python
# Rational function with symmetry
f = lambda z: (z**4 - 1) / (z**4 + 1)

# Trigonometric function
f = lambda z: np.sin(z)

# Multi-valued function (principal branch)
f = lambda z: np.sqrt(z)

# Mobius transformation
f = lambda z: (z - 1) / (z + 1)

# Blaschke product
f = lambda z: (z - 0.5) / (1 - 0.5*np.conj(0.5)*z)

# Weierstrass function (elliptic)
# See examples/advanced_features.ipynb
```

## Saving Your Visualizations

Save plots to files instead of displaying them:

```python
# Matplotlib plots
cp.plot(domain, f, cmap=cmap, filename='my_plot.png')

# PyVista plots
cp.plot_landscape_pv(domain, f, cmap=cmap,
                      filename='landscape.png',
                      show=False)

# High-resolution for publications
cp.plot(domain, f, cmap=cmap,
       resolution=1000,
       figsize=(10, 10),
       dpi=300,
       filename='publication.png')
```

## Next Steps

Now that you understand the basics, explore:

- **[User Guide](../user-guide/domains.md)** - Deep dive into domains, colormaps, and plotting options
- **[Examples Gallery](../examples/gallery.md)** - Stunning visualizations with code
- **[API Reference](../api/core.md)** - Complete function documentation
- **Example Notebooks** - Interactive tutorials:
  - `examples/getting_started.ipynb`
  - `examples/advanced_features.ipynb`
  - `examples/stl_export_demo.ipynb`
  - `examples/api_cookbook.ipynb`

## Tips for Success

1. **Start simple**: Use `cp.show()` for quick exploration
2. **Use PyVista for 3D**: Much better quality than matplotlib
3. **Try different colormaps**: Each reveals different features
4. **Experiment with domains**: Disk and annular domains can be more interesting than rectangles
5. **Use presets**: They're optimized for specific use cases
6. **Explore modulus scaling**: Different functions need different scaling for best visualization
7. **Run scripts not notebooks**: For PyVista, CLI scripts give best quality

Happy exploring!
