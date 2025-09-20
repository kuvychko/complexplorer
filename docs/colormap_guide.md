# Complexplorer Colormap Guide

## Overview

Complexplorer v2.0 offers **13 distinct colormap families** for visualizing complex functions. This guide helps you choose the right colormap for your needs, whether for scientific publication, artistic visualization, or educational purposes.

## Quick Selection Guide

| Use Case | Recommended Colormaps | Why |
|----------|----------------------|-----|
| **Scientific Publication** | CubehelixPhase, PerceptualPastel | CMYK-safe, grayscale-friendly |
| **Print Materials** | PerceptualPastel, InkPaper | Non-fluorescent, uniform brightness |
| **Presentations** | DivergingWarmCool, FourQuadrant | High contrast, clear structure |
| **Artistic/Gallery** | AnalogousWedge, EarthTopographic | Sophisticated aesthetics |
| **Educational** | Phase (enhanced), Chessboard | Clear mathematical structure |
| **Accessibility** | Isoluminant, CubehelixPhase | Colorblind-friendly options |
| **Web/Digital** | Phase, PerceptualPastel | Vibrant, screen-optimized |

## Traditional Colormaps

### Phase
The classic domain coloring approach mapping argument to hue.

```python
# Basic phase portrait
cp.plot(domain, f, cmap=cp.Phase())

# Enhanced with phase sectors (shows branch cuts clearly)
cp.plot(domain, f, cmap=cp.Phase(n_phi=6))

# Auto-scaled for square cells (best for seeing structure)
cp.plot(domain, f, cmap=cp.Phase(n_phi=6, auto_scale_r=True))
```

**Best for:** General purpose, educational visualization, identifying zeros and poles

### Chessboard
Cartesian grid pattern showing how rectangles are transformed.

```python
cp.plot(domain, f, cmap=cp.Chessboard(spacing=0.5))
```

**Best for:** Understanding conformal mappings, seeing distortion

### PolarChessboard
Polar grid pattern with radial and angular divisions.

```python
# Linear radial spacing
cp.plot(domain, f, cmap=cp.PolarChessboard(n_phi=8, spacing=0.5))

# Logarithmic radial spacing (emphasizes origin behavior)
cp.plot(domain, f, cmap=cp.PolarChessboard(n_phi=8, r_log=np.e))
```

**Best for:** Functions with polar symmetry, essential singularities

### LogRings
Pure modulus visualization with logarithmic black/white rings.

```python
cp.plot(domain, f, cmap=cp.LogRings(log_spacing=0.3))
```

**Best for:** Focusing on magnitude only, growth rates

## New Perceptually Uniform Colormaps (v2.0)

### PerceptualPastel
Elegant OkLCh-based pastels with uniform perceived brightness across all hues.

```python
# Default settings for print
cp.plot(domain, f, cmap=cp.PerceptualPastel())

# With enhanced phase portrait
cp.plot(domain, f, cmap=cp.PerceptualPastel(
    n_phi=6,           # Phase sectors
    auto_scale_r=True, # Square cells
    L_center=0.55,     # Lightness center
    C=0.1              # Low chroma for pastels
))
```

**Best for:** 
- Scientific publications (non-distracting colors)
- Print materials (CMYK-friendly)
- Long viewing sessions (easy on eyes)

**Color space:** OkLCh (perceptually uniform)

### Isoluminant
Constant brightness with only hue variation - pure phase information.

```python
# Flat brightness, only phase varies
cp.plot(domain, f, cmap=cp.Isoluminant(
    L=0.6,     # Fixed lightness
    C=0.15,    # Moderate chroma
    n_phi=8    # Optional phase sectors
))

# With subtle contour lines for modulus
cp.plot(domain, f, cmap=cp.Isoluminant(
    show_contours=True,
    contour_width=0.05
))
```

**Best for:**
- Focusing purely on phase structure
- Accessibility (no brightness variation)
- Overlaying additional information

**Color space:** OkLCh with fixed L

### CubehelixPhase
Scientific coloring with optimal grayscale conversion and CMYK safety.

```python
# Default cubehelix parameters
cp.plot(domain, f, cmap=cp.CubehelixPhase())

# Custom helix parameters
cp.plot(domain, f, cmap=cp.CubehelixPhase(
    start=0.5,      # Starting color
    rotations=1.5,  # Hue rotations  
    hue_range=1.0,  # Full vs partial spectrum
    gamma=1.0       # Brightness curve
))
```

**Best for:**
- Academic publications
- Ensuring grayscale readability
- Colorblind accessibility

**Color space:** Cubehelix (linear in grayscale)

## Artistic & Thematic Colormaps

### AnalogousWedge
Sophisticated palettes using compressed hue ranges instead of full spectrum.

```python
# Ocean theme (teals and blues)
cp.plot(domain, f, cmap=cp.AnalogousWedge(
    H_center=0.55,  # Blue-green center
    H_wedge=0.2,    # Narrow hue range
    S=0.35          # Moderate saturation
))

# Sunset theme (warm colors)
cp.plot(domain, f, cmap=cp.AnalogousWedge(
    H_center=0.08,  # Orange center
    H_wedge=0.25    # Warm range
))

# Violet theme
cp.plot(domain, f, cmap=cp.AnalogousWedge(
    H_center=0.75,  # Purple center
    H_wedge=0.2
))
```

**Presets:**
- Ocean: `H_center=0.55, H_wedge=0.2`
- Sunset: `H_center=0.08, H_wedge=0.25`
- Forest: `H_center=0.33, H_wedge=0.2`
- Violet: `H_center=0.75, H_wedge=0.2`

**Best for:**
- Artistic visualizations
- Matching brand colors
- Reducing "rainbow" effect

### DivergingWarmCool
Cartographic style with warm colors for positive phase, cool for negative.

```python
# Default warm/cool split
cp.plot(domain, f, cmap=cp.DivergingWarmCool())

# Custom color anchors
cp.plot(domain, f, cmap=cp.DivergingWarmCool(
    warm_hue=0.08,    # Orange for positive
    cool_hue=0.61,    # Blue for negative
    transition=0.3    # Blend zone width
))
```

**Best for:**
- Emphasizing real/imaginary axis structure
- Signed data visualization
- Geographic/cartographic aesthetics

**Color space:** OkLCh with interpolated hues

### InkPaper
Nearly monochrome with subtle phase tints - elegant etching aesthetic.

```python
# Subtle phase coloring
cp.plot(domain, f, cmap=cp.InkPaper(
    phase_strength=0.05,  # Very subtle color
    L_range=(0.35, 0.85)  # Ink to paper range
))

# With phase contours for structure
cp.plot(domain, f, cmap=cp.InkPaper(
    phase_strength=0.03,
    n_phi=8,              # Phase divisions
    contour_strength=0.1  # Visible lines
))
```

**Best for:**
- Formal presentations
- Minimalist aesthetics
- Focus on topology over color

### EarthTopographic
Terrain-inspired coloring with natural hillshade effects.

```python
# Default earth tones
cp.plot(domain, f, cmap=cp.EarthTopographic())

# Custom terrain colors
cp.plot(domain, f, cmap=cp.EarthTopographic(
    water_hue=0.55,    # Ocean blue
    land_hue=0.08,     # Sandy brown
    snow_level=3.0,    # High elevation white
    sea_level=0.1      # Water threshold
))
```

**Best for:**
- Topographic interpretation of functions
- Natural/organic aesthetics
- Intuitive peaks (poles) and valleys (zeros)

**Color space:** OkLCh with terrain mapping

### FourQuadrant
Bauhaus-inspired geometric palette with four tasteful color anchors.

```python
# Default quadrant colors
cp.plot(domain, f, cmap=cp.FourQuadrant())

# Custom quadrant setup
cp.plot(domain, f, cmap=cp.FourQuadrant(
    colors=['#E74C3C', '#3498DB', '#2ECC71', '#F39C12'],  # Custom palette
    C=0.25,        # Chroma level
    L_base=0.5,    # Base lightness
    blend=0.2      # Smoothing between quadrants
))
```

**Best for:**
- Modern/geometric aesthetics
- Reduced color palettes
- Clear quadrant identification

## Advanced Features

### Enhanced Phase Portraits
All colormaps support phase and modulus enhancements:

```python
# Any colormap can have phase sectors
cmap = cp.PerceptualPastel(n_phi=6)

# Any colormap can have modulus contours  
cmap = cp.AnalogousWedge(r_linear_step=0.5)

# Auto-scaling works with all colormaps
cmap = cp.DivergingWarmCool(n_phi=6, auto_scale_r=True)
```

### Parameter Reference

#### Common Parameters (all colormaps)
- `n_phi`: Number of phase sectors (branch cut visualization)
- `r_linear_step`: Linear modulus contour spacing
- `r_log_base`: Logarithmic contour base (e.g., `np.e`, `10`)
- `auto_scale_r`: Automatically calculate contour spacing for square cells
- `scale_radius`: Reference radius for auto-scaling

#### Colormap-Specific Parameters

**PerceptualPastel**
- `L_center`: Center lightness (0.4-0.7 recommended)
- `L_range`: Lightness variation (0.2-0.4)
- `C`: Chroma/saturation (0.05-0.15 for pastels)

**AnalogousWedge**
- `H_center`: Hue center (0-1, 0=red, 0.33=green, 0.66=blue)
- `H_wedge`: Hue range width (0.1-0.5, clamped)
- `S`: Saturation (0.2-0.5)
- `V_range`: Value/brightness range

**DivergingWarmCool**
- `warm_hue`: Warm anchor hue
- `cool_hue`: Cool anchor hue
- `transition`: Blend zone width

**InkPaper**
- `phase_strength`: How much phase affects color (0.01-0.1)
- `L_range`: Lightness range tuple
- `contour_strength`: Phase line visibility

**EarthTopographic**
- `water_hue`, `land_hue`: Color anchors
- `snow_level`: Elevation for white
- `sea_level`: Water/land threshold

## Choosing the Right Colormap

### For Scientific Work
1. **CubehelixPhase** - Best for publications requiring grayscale compatibility
2. **PerceptualPastel** - Uniform perception, print-friendly
3. **Phase** (enhanced) - Standard in complex analysis

### For Artistic Visualization
1. **AnalogousWedge** - Sophisticated, themed palettes
2. **EarthTopographic** - Natural, intuitive interpretation
3. **InkPaper** - Minimalist elegance

### For Education
1. **Phase** with enhancements - Clear mathematical structure
2. **Chessboard/PolarChessboard** - Shows transformations
3. **DivergingWarmCool** - Intuitive pos/neg distinction

### For Accessibility
1. **Isoluminant** - No brightness variation
2. **CubehelixPhase** - Colorblind-friendly
3. **InkPaper** - Minimal color dependence

## Code Examples

### Comparing Colormaps
```python
import complexplorer as cp
import matplotlib.pyplot as plt

# Test function
f = lambda z: (z - 1) / (z**2 + z + 1)
domain = cp.Rectangle(3, 3)

# Compare different colormaps
colormaps = [
    cp.Phase(n_phi=6, auto_scale_r=True),
    cp.PerceptualPastel(n_phi=6, auto_scale_r=True),
    cp.AnalogousWedge(H_center=0.55, H_wedge=0.2),
    cp.DivergingWarmCool(),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
for ax, cmap in zip(axes.flat, colormaps):
    cp.plot(domain, f, cmap=cmap, ax=ax)
    ax.set_title(cmap.__class__.__name__)
plt.tight_layout()
plt.show()
```

### Creating Custom Themes
```python
# Corporate color theme
def corporate_theme():
    return cp.AnalogousWedge(
        H_center=0.58,  # Company blue
        H_wedge=0.15,   # Narrow range
        S=0.4,          # Professional saturation
        n_phi=4,        # Quarterly divisions
        auto_scale_r=True
    )

# Use the theme
cp.plot(domain, f, cmap=corporate_theme())
```

### Publication-Ready Figures
```python
# High-quality figure for journal
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

# Perceptual pastel for main visualization
cp.plot(domain, f, 
        cmap=cp.PerceptualPastel(n_phi=6, auto_scale_r=True),
        ax=ax1, resolution=800)
ax1.set_title("Perceptual Visualization")

# Cubehelix for grayscale-compatible version
cp.plot(domain, f,
        cmap=cp.CubehelixPhase(n_phi=6, auto_scale_r=True),
        ax=ax2, resolution=800)
ax2.set_title("Grayscale-Compatible")

plt.tight_layout()
plt.savefig("figure.pdf", dpi=300, bbox_inches='tight')
```

## Tips and Best Practices

1. **Start with auto-scaling**: Use `auto_scale_r=True` to automatically get well-proportioned cells
2. **Match colormap to medium**: Use PerceptualPastel for print, Phase for screen
3. **Consider your audience**: CubehelixPhase for academics, AnalogousWedge for designers
4. **Test in grayscale**: Convert your figures to grayscale to ensure readability
5. **Use consistent colormaps**: Stick to one family within a document for coherence
6. **Leverage phase sectors**: `n_phi` parameter helps visualize branch cuts
7. **Adjust for function range**: Tune parameters based on your function's behavior

## Mathematical Background

### OkLCh Color Space
Several new colormaps use the OkLCh color space, which provides:
- **L**: Perceptually uniform lightness
- **C**: Chroma (saturation)
- **h**: Hue angle

This ensures consistent perceived brightness as hue changes, unlike HSV.

### Cubehelix
The cubehelix system generates color palettes that:
- Increase monotonically in perceived brightness
- Print correctly in grayscale
- Wrap around the color wheel

### Phase Mapping
All colormaps map the complex phase θ = arg(z) to visual properties:
- Traditional: θ → hue
- Diverging: sign(θ) → warm/cool
- Quadrant: θ → discrete colors
- Wedge: θ → compressed hue range

## Version History

- **v2.0**: Added 8 new colormap families (Perceptual, Artistic, Thematic)
- **v1.0**: Original 5 colormaps (Phase, Chessboard, PolarChessboard, LogRings)

## See Also

- [Gallery](gallery/README.md) - Visual examples of all colormaps
- [Getting Started](../examples/getting_started.ipynb) - Interactive tutorials
- [API Reference](../complexplorer/core/colormap.py) - Implementation details