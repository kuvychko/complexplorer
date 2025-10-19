# Core API Reference

The core module contains fundamental classes and functions for working with complex functions.

## Domains

::: complexplorer.core.domain
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3
      members:
        - Domain
        - Rectangle
        - Disk
        - Annulus

### Domain Base Class

All domains inherit from the `Domain` base class and provide:

- `contains(z)`: Check if points are in the domain
- `sample(resolution)`: Generate grid of points for visualization
- Proper representation and string formatting

### Rectangle

Rectangular domain in the complex plane.

**Example:**
```python
import complexplorer as cp

# Square centered at origin
domain = cp.Rectangle(4, 4)

# Rectangle centered at 1+1j
domain = cp.Rectangle(re_length=6, im_length=4, center=1+1j)
```

### Disk

Circular domain in the complex plane.

**Example:**
```python
# Unit disk
domain = cp.Disk(radius=1)

# Disk centered at 2j
domain = cp.Disk(radius=3, center=2j)
```

### Annulus

Ring-shaped domain between two circles.

**Example:**
```python
# Annulus excluding origin
domain = cp.Annulus(inner_radius=0.5, outer_radius=3)

# Off-center annulus
domain = cp.Annulus(inner_radius=1, outer_radius=2, center=1+1j)
```

## Colormaps

::: complexplorer.core.colormap
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3
      members:
        - Colormap
        - Phase
        - OklabPhase
        - PerceptualPastel
        - Isoluminant
        - CubehelixPhase
        - AnalogousWedge
        - DivergingWarmCool
        - InkPaper
        - EarthTopographic
        - FourQuadrant
        - Chessboard
        - PolarChessboard
        - LogRings

### Colormap Base Class

All colormaps inherit from `Colormap` and provide:

- `__call__(z)`: Map complex values to RGB colors
- `hsv(z)`: Map complex values to HSV colors (internal)
- `rgb(z)`: Convert HSV to RGB (internal)

### Classic Phase Portraits

#### Phase

The fundamental phase portrait colormap.

**Example:**
```python
# Basic phase portrait
cmap = cp.Phase()

# Enhanced with 6 phase sectors
cmap = cp.Phase(phase_sectors=6)

# Auto-scaled for square cells
cmap = cp.Phase(phase_sectors=6, auto_scale_r=True, scale_radius=0.8)

# With modulus contours
cmap = cp.Phase(r_linear_step=0.5)
cmap = cp.Phase(r_log_base=2.0)
```

**Parameters:**
- `phase_sectors`: Number of angular wedges (0 = none)
- `r_linear_step`: Linear modulus contour spacing (0 = none)
- `r_log_base`: Logarithmic modulus base (0 = none)
- `v_base`: Minimum brightness (default 0.5)
- `auto_scale_r`: Auto-calculate spacing for square cells
- `scale_radius`: Reference radius for auto-scaling
- `emphasize_unit_circle`: Highlight |z| = 1
- `unit_circle_strength`: Emphasis strength

#### OklabPhase

Perceptually uniform phase portrait using OKLAB color space.

**Example:**
```python
# Smooth OKLAB phase
cmap = cp.OklabPhase(L=0.7, C=0.35)

# Enhanced with phase sectors
cmap = cp.OklabPhase(phase_sectors=6, auto_scale_r=True, enhanced=True)
```

**Parameters:**
- `L`: Lightness (0 to 1, default 0.7)
- `C`: Chroma (0 to 0.5, default 0.35)
- `enhanced`: Enable sawtooth modulation
- `phase_offset`: Color rotation offset
- Plus all Phase parameters

### Perceptually Uniform Colormaps

#### PerceptualPastel

Elegant OkLCh-based pastels with uniform brightness.

**Example:**
```python
cmap = cp.PerceptualPastel()
cmap = cp.PerceptualPastel(phase_sectors=6, auto_scale_r=True)
cmap = cp.PerceptualPastel(L_center=0.60, C=0.08)
```

#### Isoluminant

Constant lightness, phase encoded purely in hue.

**Example:**
```python
cmap = cp.Isoluminant()  # With contours
cmap = cp.Isoluminant(show_contours=False)  # Pure isoluminant
```

#### CubehelixPhase

Grayscale-safe colormap for scientific publishing.

**Example:**
```python
cmap = cp.CubehelixPhase()
cmap = cp.CubehelixPhase(start=0.5, rotations=1.5)
```

### Artistic Colormaps

#### AnalogousWedge

Harmonious colors from compressed hue range.

**Example:**
```python
cmap = cp.AnalogousWedge(H_center=0.55, H_wedge=0.2)  # Ocean
cmap = cp.AnalogousWedge(H_center=0.05, H_wedge=0.15)  # Sunset
```

#### DivergingWarmCool

Warm/cool divergence for cartographic style.

**Example:**
```python
cmap = cp.DivergingWarmCool()
```

#### InkPaper

Minimalist near-monochrome with subtle phase tints.

**Example:**
```python
cmap = cp.InkPaper()
cmap = cp.InkPaper(C_min=0.01, C_max=0.04)
```

#### EarthTopographic

Terrain-inspired with earth tones and hillshade.

**Example:**
```python
cmap = cp.EarthTopographic()
cmap = cp.EarthTopographic(H_water=200, H_land=30)
```

#### FourQuadrant

Bauhaus-inspired with 4 color anchors.

**Example:**
```python
cmap = cp.FourQuadrant()
```

### Grid & Pattern Colormaps

#### Chessboard

Black/white grid aligned with Re/Im axes.

**Example:**
```python
cmap = cp.Chessboard()
cmap = cp.Chessboard(spacing=0.5)
```

#### PolarChessboard

Black/white pattern in polar coordinates.

**Example:**
```python
cmap = cp.PolarChessboard()
cmap = cp.PolarChessboard(phase_sectors=12, spacing=0.5)
cmap = cp.PolarChessboard(phase_sectors=8, r_log=2.0)
```

#### LogRings

Logarithmic concentric black/white rings.

**Example:**
```python
cmap = cp.LogRings()
cmap = cp.LogRings(log_spacing=0.3)
```

## Modulus Scaling

::: complexplorer.core.scaling
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

Modulus scaling functions control how |z| is mapped to height in 3D visualizations and Riemann sphere relief maps.

### Available Modes

- `constant`: Flat (no height variation)
- `linear`: Direct |z| mapping
- `arctan`: Smooth bounded scaling (recommended)
- `logarithmic`: log(1 + |z|) - emphasizes poles/zeros
- `adaptive`: Auto-adjust to data range
- `linear_clamp`: Linear with maximum cap
- `power`: |z|^p mapping
- `sigmoid`: S-curve mapping
- `hybrid`: Combination modes

**Example:**
```python
from complexplorer.core.scaling import apply_modulus_scaling

# Apply arctan scaling
heights = apply_modulus_scaling(z_values, mode='arctan')

# With custom parameters
heights = apply_modulus_scaling(z_values, mode='arctan',
                                params={'r_min': 0.3, 'r_max': 0.9})
```

## Core Functions

::: complexplorer.core.functions
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

Fundamental complex function utilities.

### Phase and Modulus

Functions for extracting phase (argument) and modulus (magnitude):

```python
from complexplorer.core.functions import phase, modulus

z = 1 + 1j
arg = phase(z)  # pi/4
mag = modulus(z)  # sqrt(2)
```

### Sawtooth Function

Used for enhanced phase portraits:

```python
from complexplorer.core.functions import sawtooth

# Create phase sectors
brightness = sawtooth(phase_value, n_sectors=6)
```

### Stereographic Projection

Maps complex plane to/from Riemann sphere:

```python
from complexplorer.core.functions import stereographic_to_sphere, sphere_to_stereographic

# Map to sphere
x, y, z_coord = stereographic_to_sphere(complex_values)

# Map back to plane
w = sphere_to_stereographic(x, y, z_coord)
```

## Color Utilities

::: complexplorer.core.color_utils
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

Color space conversions and utilities.

### Color Space Conversions

```python
from complexplorer.core.color_utils import hsv_to_rgb, oklab_to_rgb, oklch_to_rgb

# HSV to RGB
rgb = hsv_to_rgb(h, s, v)

# OKLAB to RGB
rgb = oklab_to_rgb(L, a, b)

# OkLCh to RGB
rgb = oklch_to_rgb(L, C, h)
```

## Constants

::: complexplorer.core.constants
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

Mathematical and visualization constants.

```python
from complexplorer.core.constants import PI, TAU, E, I

# Mathematical constants
two_pi = TAU
imaginary_unit = I
eulers_number = E
```

## See Also

- **[Plotting API](plotting.md)** - Visualization functions
- **[Export API](export.md)** - STL export for 3D printing
- **[User Guide](../user-guide/domains.md)** - Practical usage examples
