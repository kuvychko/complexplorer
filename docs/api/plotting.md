# Plotting API Reference

Functions for visualizing complex functions in 2D, 3D, and on the Riemann sphere.

## High-Level API

::: complexplorer.api
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3
      members:
        - show
        - plot
        - pair_plot
        - plot_landscape
        - pair_plot_landscape
        - riemann
        - publication_preset
        - interactive_preset
        - high_contrast_preset

### Quick Plot Function

#### show()

Simplified plotting function for quick exploration.

**Example:**
```python
import complexplorer as cp

# Quickest way to visualize
cp.show(lambda z: z**2)

# With custom domain
cp.show(lambda z: z**2, domain_size=6)

# With colormap
cp.show(lambda z: z**2, colormap='pastel')
```

**Parameters:**
- `func`: Complex function z -> f(z)
- `domain_size`: Size of square domain (default: 4)
- `colormap`: Colormap name or object
- `resolution`: Points per axis (default: 500)
- `**kwargs`: Additional arguments passed to plot()

### 2D Plotting Functions

#### plot()

Main 2D visualization function.

**Example:**
```python
f = lambda z: (z**2 - 1) / (z**2 + 1)
domain = cp.Rectangle(4, 4)
cmap = cp.Phase(phase_sectors=6, auto_scale_r=True)

# Basic plot
cp.plot(domain, f, cmap=cmap)

# High-resolution
cp.plot(domain, f, cmap=cmap, resolution=800)

# Save to file
cp.plot(domain, f, cmap=cmap, filename='output.png', show=False)
```

**Parameters:**
- `domain`: Domain object (Rectangle, Disk, or Annulus)
- `func`: Complex function z -> f(z)
- `cmap`: Colormap object (default: Phase with 6 sectors)
- `resolution`: Points per axis (default: 500)
- `figsize`: Figure size in inches (default: (8, 8))
- `dpi`: Dots per inch (default: 100)
- `title`: Plot title (optional)
- `filename`: Save to file instead of displaying (optional)
- `show`: Whether to display the plot (default: True)

**Returns:** matplotlib axes object

#### pair_plot()

Side-by-side visualization of domain and codomain.

**Example:**
```python
# Compare input and output
cp.pair_plot(domain, f, cmap=cmap)

# Custom figure size
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
- `show`: Whether to display (default: True)

**Returns:** matplotlib figure and axes

### 3D Plotting Functions (Matplotlib)

::: complexplorer.plotting.matplotlib.plots_3d
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4
      members:
        - plot_landscape
        - pair_plot_landscape

#### plot_landscape()

Create 3D surface where height represents function modulus.

**Example:**
```python
# Basic 3D landscape
cp.plot_landscape(domain, f)

# With custom colormap
cmap = cp.Phase(phase_sectors=6, auto_scale_r=True)
cp.plot_landscape(domain, f, cmap=cmap)

# Adjust vertical scaling
cp.plot_landscape(domain, f, cmap=cmap, z_scale=0.5)

# Custom view angles
cp.plot_landscape(domain, f, elevation=30, azimuth=-60)
```

**Parameters:**
- `domain`: Domain object
- `func`: Complex function
- `cmap`: Colormap (default: Phase)
- `resolution`: Points per axis (default: 300)
- `z_scale`: Vertical scale factor (default: 1.0)
- `modulus_mode`: Height scaling mode (default: 'arctan')
- `modulus_params`: Parameters for modulus scaling
- `figsize`: Figure size (default: (10, 8))
- `elevation`: View elevation angle (default: 30)
- `azimuth`: View azimuth angle (default: -60)
- `title`: Plot title
- `filename`: Save to file
- `show`: Display plot (default: True)

**Returns:** matplotlib 3D axes

#### pair_plot_landscape()

Side-by-side 3D landscapes of domain and codomain.

**Example:**
```python
# Compare input and output in 3D
cp.pair_plot_landscape(domain, f, cmap=cmap)

# Custom view angles
cp.pair_plot_landscape(domain, f, elevation=20, azimuth=-45)
```

**Parameters:**
- Same as plot_landscape() plus:
- `labels`: List of 2 strings for subplot titles

**Returns:** matplotlib figure and axes

### 3D Plotting Functions (PyVista)

::: complexplorer.plotting.pyvista.plots_3d_pv
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4
      members:
        - plot_landscape_pv
        - pair_plot_landscape_pv

#### plot_landscape_pv()

High-quality interactive 3D landscape (15-30x faster than matplotlib).

**Example:**
```python
# Basic PyVista landscape
cp.plot_landscape_pv(domain, f)

# High resolution with custom colormap
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
- `modulus_mode`: Height scaling mode (default: 'arctan')
- `modulus_params`: Parameters for scaling
- `window_size`: Window dimensions (default: (1024, 768))
- `title`: Plot title
- `filename`: Save screenshot
- `show`: Display interactive window (default: True)
- `notebook`: Use notebook backend (default: False)

**Returns:** PyVista plotter object (if show=False)

**Interactive Controls:**
- Left mouse: Rotate view
- Middle mouse: Pan
- Right mouse: Zoom
- Scroll wheel: Zoom in/out
- 'r': Reset camera
- 's': Save screenshot
- 'q': Quit

#### pair_plot_landscape_pv()

Side-by-side PyVista landscapes.

**Example:**
```python
# High-quality comparison
cp.pair_plot_landscape_pv(domain, f, cmap=cmap)

# Wide window for side-by-side
cp.pair_plot_landscape_pv(domain, f, window_size=(1600, 800))
```

**Parameters:**
- Same as plot_landscape_pv() plus:
- `labels`: List of 2 strings for subplot titles
- `window_size`: Default (1400, 700) for side-by-side

**Returns:** PyVista plotter object

### Riemann Sphere Functions (Matplotlib)

::: complexplorer.plotting.matplotlib.riemann
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4
      members:
        - riemann

#### riemann()

Create Riemann sphere visualization using matplotlib 3D.

**Example:**
```python
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
- `modulus_mode`: Height scaling (default: 'arctan')
- `figsize`: Figure size (default: (10, 10))
- `elevation`, `azimuth`: View angles
- `title`: Plot title
- `filename`: Save to file
- `show`: Display plot (default: True)

**Returns:** matplotlib 3D axes

### Riemann Sphere Functions (PyVista)

::: complexplorer.plotting.pyvista.riemann_pv
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4
      members:
        - riemann_pv

#### riemann_pv()

High-quality interactive Riemann sphere with relief mapping.

**Example:**
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

# Domain restriction for stability
domain = cp.Annulus(inner_radius=0.1, outer_radius=10)
cp.riemann_pv(f, domain=domain, modulus_mode='logarithmic')
```

**Parameters:**
- `func`: Complex function
- `cmap`: Colormap
- `domain`: Optional domain restriction
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

**Modulus Modes for Riemann Sphere:**
- `constant`: Flat sphere (no relief)
- `arctan`: Smooth bounded relief (recommended)
- `logarithmic`: Emphasize poles/zeros
- `adaptive`: Auto-adjust to function behavior
- `sigmoid`: S-curve mapping
- `linear`: Direct mapping (can be unstable)

## Preset Configurations

#### publication_preset()

High-resolution settings for publication-quality figures.

**Example:**
```python
settings = cp.publication_preset()
# Returns: {'resolution': 1200, 'dpi': 300, 'figsize': (4, 4)}

cp.plot(domain, f, **settings, filename='publication.pdf')
```

**Returns:** dict with resolution, dpi, figsize

#### interactive_preset()

Fast settings for interactive exploration.

**Example:**
```python
settings = cp.interactive_preset()
# Returns: {'resolution': 300, 'dpi': 100, 'figsize': (8, 8)}

cp.plot(domain, f, **settings)
```

**Returns:** dict with resolution, dpi, figsize

#### high_contrast_preset()

Maximum contrast colormap settings.

**Example:**
```python
cmap = cp.high_contrast_preset()
# Returns: Phase(phase_sectors=6, auto_scale_r=True, v_base=0.3)

cp.plot(domain, f, cmap=cmap)
```

**Returns:** Phase colormap with high contrast

## Modulus Scaling Modes

All 3D and Riemann sphere functions support multiple modulus scaling modes:

| Mode | Description | Best For |
|------|-------------|----------|
| `constant` | Flat (no height) | Phase-only visualization |
| `linear` | Direct \|z\| | Simple functions |
| `arctan` | Smooth bounded | Most functions (recommended) |
| `logarithmic` | log(1 + \|z\|) | Poles and zeros |
| `adaptive` | Auto-adjust | Unknown behavior |
| `linear_clamp` | Linear with cap | Controlled height |
| `power` | \|z\|^p | Emphasize features |
| `sigmoid` | S-curve | Smooth transitions |
| `hybrid` | Combination | Complex landscapes |

**Custom Parameters Example:**
```python
# Arctan with custom scaling
cp.plot_landscape_pv(domain, f,
                     modulus_mode='arctan',
                     modulus_params={'r_min': 0.3, 'r_max': 0.9})

# Power scaling with custom exponent
cp.plot_landscape_pv(domain, f,
                     modulus_mode='power',
                     modulus_params={'exponent': 0.5})
```

## Performance Comparison

### 2D Plotting
- **Resolution 500**: ~0.5-1 second
- **Resolution 800**: ~1-2 seconds
- **Resolution 1200**: ~3-5 seconds

### 3D Plotting (Matplotlib)
- **Resolution 300**: ~2-5 seconds
- **Resolution 500**: ~8-15 seconds

### 3D Plotting (PyVista)
- **Resolution 300**: ~0.5-1 second (15-30x faster)
- **Resolution 800**: ~2-4 seconds

**Recommendation:** Use PyVista for resolutions above 400x400 or for interactive exploration.

## File Formats

### 2D Plots
- **PNG**: Raster, web-friendly, transparency support
- **PDF**: Vector, publication-ready, scalable
- **SVG**: Vector, editable in Inkscape/Illustrator
- **JPG**: Raster, compressed (use PNG instead)

### 3D Plots (PyVista)
- **PNG**: Screenshot at window resolution
- **High-res PNG**: Use large window_size parameter

**Example:**
```python
# 2D vector format
cp.plot(domain, f, filename='figure.pdf')

# 3D high-resolution screenshot
cp.plot_landscape_pv(domain, f, window_size=(1920, 1080),
                      filename='landscape.png', show=False)
```

## See Also

- **[Core API](core.md)** - Domains, colormaps, and scaling
- **[Export API](export.md)** - STL export for 3D printing
- **[User Guide](../user-guide/plotting-2d.md)** - Plotting tutorials
