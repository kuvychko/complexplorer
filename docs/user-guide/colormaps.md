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
