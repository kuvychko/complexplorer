# Colormaps

Colormaps define how complex values are transformed into colors for visualization. Complexplorer offers a rich collection of colormaps, from classic phase portraits to modern perceptually-uniform schemes.

## Quick Overview

| Colormap Family | Best For | Key Feature |
|----------------|----------|-------------|
| **Phase** | General purpose | Classic HSV phase portrait |
| **OklabPhase** | Perceptual accuracy | OKLAB color space |
| **PerceptualPastel** | Print/presentations | Elegant pastels |
| **Isoluminant** | Phase structure | Constant brightness |
| **CubehelixPhase** | Scientific publishing | Grayscale-safe |
| **AnalogousWedge** | Aesthetic appeal | Harmonious hues |
| **DivergingWarmCool** | Cartographic style | Warm/cool divergence |
| **InkPaper** | Minimalist | Near-monochrome |
| **EarthTopographic** | Natural aesthetics | Terrain-inspired |
| **FourQuadrant** | Geometric clarity | Bauhaus-inspired |
| **Chessboard** | Grid visualization | Cartesian grid |
| **PolarChessboard** | Polar structure | Circular grid |
| **LogRings** | Magnitude focus | Logarithmic rings |

## Classic Phase Portraits

### Phase

The fundamental phase portrait colormap. Maps complex phase (argument) to hue, optionally enhanced with modulus contours.

```python
import complexplorer as cp

f = lambda z: (z**2 - 1) / (z**2 + 1)
domain = cp.Rectangle(4, 4)

# Basic phase portrait
cmap = cp.Phase()
cp.plot(domain, f, cmap=cmap)

# Enhanced with 6 phase sectors
cmap = cp.Phase(phase_sectors=6)
cp.plot(domain, f, cmap=cmap)

# Auto-scaled for square cells
cmap = cp.Phase(phase_sectors=6, auto_scale_r=True, scale_radius=0.8)
cp.plot(domain, f, cmap=cmap)

# With linear modulus rings
cmap = cp.Phase(r_linear_step=0.5)
cp.plot(domain, f, cmap=cmap)

# With logarithmic modulus rings
cmap = cp.Phase(r_log_base=2.0)
cp.plot(domain, f, cmap=cmap)
```

**Parameters:**
- `phase_sectors`: Number of phase sectors (creates sawtooth brightness patterns)
- `r_linear_step`: Linear modulus contour spacing
- `r_log_base`: Logarithmic modulus contour base
- `v_base`: Minimum brightness (0 to 1)
- `auto_scale_r`: Auto-calculate r_linear_step for square cells
- `scale_radius`: Reference radius for auto-scaling
- `emphasize_unit_circle`: Highlight |z|=1
- `unit_circle_strength`: Strength of unit circle emphasis

**Use Cases:**
- General purpose visualization
- Teaching complex analysis
- Exploring function behavior

### OklabPhase

Perceptually uniform phase portrait using the OKLAB color space. Better color consistency than standard HSV-based Phase colormap.

```python
# Smooth OKLAB phase (cplot-like)
cmap = cp.OklabPhase(L=0.7, C=0.35)
cp.plot(domain, f, cmap=cmap)

# Enhanced with phase sectors
cmap = cp.OklabPhase(phase_sectors=6, auto_scale_r=True, enhanced=True)
cp.plot(domain, f, cmap=cmap)

# Adjust lightness and chroma
cmap = cp.OklabPhase(L=0.6, C=0.25, phase_sectors=8)
cp.plot(domain, f, cmap=cmap)
```

**Parameters:**
- `L`: Lightness (0 to 1, default 0.7)
- `C`: Chroma/saturation (0 to 0.5, default 0.35)
- `enhanced`: Enable sawtooth modulation (default False)
- `phase_offset`: Color rotation offset
- Plus all Phase parameters

**Use Cases:**
- When color perception matters
- Comparing multiple visualizations
- High-quality publications

## Perceptually Uniform Colormaps

### PerceptualPastel

Elegant OkLCh-based pastels with uniform perceived brightness. Non-fluorescent colors ideal for print.

```python
# Default pastels
cmap = cp.PerceptualPastel()
cp.plot(domain, f, cmap=cmap)

# Enhanced with phase sectors
cmap = cp.PerceptualPastel(phase_sectors=6, auto_scale_r=True)
cp.plot(domain, f, cmap=cmap)

# Adjust pastel intensity
cmap = cp.PerceptualPastel(L_center=0.60, C=0.08)
cp.plot(domain, f, cmap=cmap)
```

**Key Parameters:** `L_center`, `L_range`, `C` (chroma)

### Isoluminant

Constant lightness with phase encoded purely in hue. Optional thin contour lines for modulus.

```python
cmap = cp.Isoluminant()  # With contours
cmap = cp.Isoluminant(show_contours=False)  # Pure isoluminant
cmap = cp.Isoluminant(L=0.7, C_min=0.15, C_max=0.20)
```

**Key Parameters:** `L`, `C_min`, `C_max`, `show_contours`, `contour_period`

### CubehelixPhase

Scientific coloring that prints well in grayscale.

```python
cmap = cp.CubehelixPhase()
cmap = cp.CubehelixPhase(start=0.5, rotations=1.5, saturation=0.8)
```

**Key Parameters:** `start`, `rotations`, `saturation`, `L_min`, `L_max`

## Artistic Colormaps

### AnalogousWedge

Harmonious colors from compressed hue range (20-50% of wheel).

```python
cmap = cp.AnalogousWedge(H_center=0.55, H_wedge=0.2)  # Ocean
cmap = cp.AnalogousWedge(H_center=0.05, H_wedge=0.15)  # Sunset
```

**Key Parameters:** `H_center`, `H_wedge`, `S`, `V_base`

### DivergingWarmCool

Warm for positive phase, cool for negative.

```python
cmap = cp.DivergingWarmCool()  # Amber to indigo
```

**Key Parameters:** `H_warm`, `H_cool`, `L_center`, `C_min`, `C_max`

### InkPaper

Nearly monochrome with subtle phase tints.

```python
cmap = cp.InkPaper()
cmap = cp.InkPaper(C_min=0.01, C_max=0.04)  # Very subtle
```

**Key Parameters:** `L_min`, `L_max`, `C_min`, `C_max`

### EarthTopographic

Terrain-inspired with earth tones and hillshade.

```python
cmap = cp.EarthTopographic()
cmap = cp.EarthTopographic(H_water=200, H_land=30)
```

**Key Parameters:** `H_water`, `H_land`, `add_hillshade`

### FourQuadrant

Bauhaus-inspired with 4 color anchors.

```python
cmap = cp.FourQuadrant()  # Red, green, cyan, magenta
```

**Key Parameters:** `H_anchors`, `C`, `L_min`, `L_max`

## Grid & Pattern Colormaps

### Chessboard

Black/white grid aligned with Re/Im axes.

```python
cmap = cp.Chessboard()
cmap = cp.Chessboard(spacing=0.5)  # Finer grid
```

**Use:** Visualize Cartesian grid distortion

### PolarChessboard

Black/white pattern in polar coordinates.

```python
cmap = cp.PolarChessboard()
cmap = cp.PolarChessboard(phase_sectors=12, spacing=0.5)
cmap = cp.PolarChessboard(phase_sectors=8, r_log=2.0)  # Log spacing
```

**Use:** Visualize polar structure

### LogRings

Logarithmic concentric black/white rings.

```python
cmap = cp.LogRings()
cmap = cp.LogRings(log_spacing=0.3)  # Tighter rings
```

**Use:** Focus on magnitude structure

## Enhanced Phase Portraits

Many colormaps support enhanced phase portraits:

### Phase Sectors
Create sawtooth brightness in angular wedges:
```python
cmap = cp.Phase(phase_sectors=6)
```

### Modulus Contours
Brightness variations at specific |z| values:
```python
cmap = cp.Phase(r_linear_step=0.5)  # Linear: 0.5, 1.0, 1.5, ...
cmap = cp.Phase(r_log_base=2.0)     # Log: 1, 2, 4, 8, ...
```

### Auto-Scaled Square Cells
Automatically calculate spacing for square cells:
```python
cmap = cp.Phase(phase_sectors=6, auto_scale_r=True, scale_radius=0.8)
```

## Choosing the Right Colormap

### By Purpose

**Learning/Teaching:** `Phase`, `Chessboard`, `PolarChessboard`

**Publications:** `PerceptualPastel`, `CubehelixPhase`, `Isoluminant`

**Presentations:** `PerceptualPastel`, `InkPaper`, `AnalogousWedge`

**Exploration:** `Phase` with `auto_scale_r`, `OklabPhase`, `LogRings`

**Artistic:** `EarthTopographic`, `FourQuadrant`, `DivergingWarmCool`

### By Function Type

**Rational (poles/zeros):** Enhanced `Phase`, `LogRings`, `Isoluminant`

**Periodic/Trig:** `Phase` with contours, `PolarChessboard`, `PerceptualPastel`

**Multi-valued:** `Phase` with sectors, `Chessboard`, `OklabPhase`

## Advanced Usage

### Brightness Control

```python
# Darker base for contrast (default 0.5)
cmap = cp.Phase(phase_sectors=6, v_base=0.3)

# Lighter base for pastel look
cmap = cp.Phase(phase_sectors=6, v_base=0.7)
```

### Out-of-Domain Coloring

```python
# Custom color for points outside domain (HSV)
cmap = cp.Phase(out_of_domain_hsv=(0.6, 0.1, 0.95))
```

### Combining Enhancements

```python
# Both phase sectors AND modulus rings
cmap = cp.Phase(phase_sectors=6, r_linear_step=0.5, v_base=0.4)
```

## Quick Examples

```python
import complexplorer as cp

f = lambda z: (z**2 - 1) / (z**2 + 1)
domain = cp.Rectangle(4, 4)

# Classic enhanced phase
cp.plot(domain, f, cmap=cp.Phase(phase_sectors=6, auto_scale_r=True))

# Perceptual pastel
cp.plot(domain, f, cmap=cp.PerceptualPastel(phase_sectors=6, auto_scale_r=True))

# Scientific
cp.plot(domain, f, cmap=cp.CubehelixPhase(phase_sectors=6, auto_scale_r=True))

# Minimalist
cp.plot(domain, f, cmap=cp.InkPaper())

# Grid
cp.plot(domain, f, cmap=cp.Chessboard(spacing=0.5))

# Magnitude
cp.plot(domain, f, cmap=cp.LogRings())
```

## Tips

1. Start with `Phase(phase_sectors=6, auto_scale_r=True)` for exploration
2. Use `PerceptualPastel` for presentations and print
3. Use `CubehelixPhase` for scientific publications
4. Use `Isoluminant` to focus purely on phase structure
5. Use `Chessboard`/`PolarChessboard` to understand grid distortion
6. Adjust `v_base` to control contrast (lower = more contrast)
7. Try `InkPaper` for elegant, minimalist visualizations
8. Use `OklabPhase` when color perception accuracy matters

## Next Steps

- **[2D Plotting](plotting-2d.md)** - Apply colormaps in 2D
- **[3D Plotting](plotting-3d.md)** - Use colormaps for 3D surfaces
- **[Riemann Sphere](riemann.md)** - Colormaps on the sphere
- **[Domains](domains.md)** - Domain selection
- **[Quick Start](../getting-started/quickstart.md)** - Basic examples
