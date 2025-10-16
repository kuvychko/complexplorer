# Example Notebooks

Complexplorer provides comprehensive tutorial notebooks and application examples to help you master complex function visualization.

## Tutorial Notebooks

The `examples/notebooks/` directory contains 8 progressively structured tutorial notebooks (~2.5 hours total).

### 1. Getting Started (10 min)

**File**: [`examples/notebooks/01_getting_started.ipynb`](../../examples/notebooks/01_getting_started.ipynb)

Your first introduction to complexplorer:

- Installation verification
- Basic domains: Rectangle, Disk, Annulus
- Enhanced phase portraits with `auto_scale_r=True`
- 2D plotting with `plot()`
- Standard demo function: `f(z) = (z - 1) / (z**2 + z + 1)`

**Prerequisites**: None - start here!

---

### 2. Advanced Domains (15 min)

**File**: [`examples/notebooks/02_domains_advanced.ipynb`](../../examples/notebooks/02_domains_advanced.ipynb)

Master domain set operations (v2.0 feature):

- Union: `domain1 | domain2`
- Intersection: `domain1 & domain2`
- Difference: `domain1 - domain2`
- Complex compositions (flower patterns, swiss cheese)
- Custom domain design

**Prerequisites**: Notebook #1

---

### 3. PyVista 3D Visualization (20 min)

**File**: [`examples/notebooks/03_pyvista_3d.ipynb`](../../examples/notebooks/03_pyvista_3d.ipynb)

High-performance 3D visualization:

- `plot_landscape_pv()` - analytic landscapes
- Inline mode (`notebook=True`) vs interactive (`notebook=False`)
- Pair plots: `pair_plot_landscape_pv()`
- Performance comparison: 15-30x faster than matplotlib

**Key decision**: Use `notebook=True` for development (non-blocking), `notebook=False` for publication quality.

**Prerequisites**: Notebook #1

---

### 4. Colormaps Comprehensive (25 min)

**File**: [`examples/notebooks/04_colormaps_comprehensive.ipynb`](../../examples/notebooks/04_colormaps_comprehensive.ipynb)

Complete colormap reference:

**All 13 families**:
- Phase (basic + enhanced)
- Chessboard / PolarChessboard
- LogRings
- PerceptualPastel (OkLCh-based, v2.0)
- Isoluminant
- CubehelixPhase
- AnalogousWedge (v2.0)
- DivergingWarmCool (v2.0)
- InkPaper (v2.0)
- EarthTopographic (v2.0)
- FourQuadrant (v2.0)
- Sinebow
- CustomCmap

**Use case guide** included!

**Prerequisites**: Notebook #1

---

### 5. Accessibility & CVD (20 min)

**File**: [`examples/notebooks/05_accessibility_cvd.ipynb`](../../examples/notebooks/05_accessibility_cvd.ipynb)

**CRITICAL for publications!**

Color vision deficiency testing:

- CVD statistics: ~8% of males affected
- Simulation with colorspacious library
- Test protanopia, deuteranopia, tritanopia
- CVD-friendly colormaps: CubehelixPhase, PerceptualPastel
- Design guidelines for inclusive figures

**Requirement**: `pip install colorspacious`

**Prerequisites**: Notebook #4

---

### 6. Riemann Sphere (30 min)

**File**: [`examples/notebooks/06_riemann_sphere.ipynb`](../../examples/notebooks/06_riemann_sphere.ipynb)

Extended complex plane  * {}:

- Stereographic projection math
- 2D hemisphere charts with `riemann_chart()`
- 3D sphere with `riemann_pv()` (PyVista)
- Essential singularities: `exp(1/z)`

**Prerequisites**: Notebooks #1, #3

---

### 7. Modulus Scaling (25 min)

**File**: [`examples/notebooks/07_modulus_scaling.ipynb`](../../examples/notebooks/07_modulus_scaling.ipynb)

Control 3D height mapping:

**10+ scaling modes**:
- `none` - direct modulus
- `constant` - flat (phase only)
- `arctan` - smooth bounded (recommended)
- `logarithmic` - emphasize poles/zeros
- `linear_clamp`, `adaptive`, `power`, `sigmoid`, ...
- `custom` - user-defined function

**Parameter exploration** for fine control.

**Prerequisites**: Notebooks #3, #6

---

### 8. STL Export for 3D Printing (30 min)

**File**: [`examples/notebooks/08_stl_export.ipynb`](../../examples/notebooks/08_stl_export.ipynb)

Transform math into physical art:

- `OrnamentGenerator` workflow
- Modulus scaling selection
- Domain restrictions (stability)
- **Bisection workflow** (PrusaSlicer/Cura) - RECOMMENDED
- Gluing tips: cyanoacrylate vs epoxy
- Post-processing: sanding, painting

**Why bisect?** No supports, easier printing, multi-color options.

**Prerequisites**: Notebooks #6, #7

---

## Application Notebooks

The `examples/applications/` directory contains 4 real-world application examples.

### 1. FFT/DFT Matrices (15 min)

**File**: [`examples/applications/app_01_fft_matrices.ipynb`](../../examples/applications/app_01_fft_matrices.ipynb)

Discrete Fourier Transform visualization:

- DFT matrix: $W_N^{jk}$ where $W_N = e^{-2\pi i/N}$
- **2×4 grid gallery**: N = 32, 64, 128, 256, 512, 1024, 2048, 4096
- Circular symmetry from roots of unity
- Extensions: IDFT, Hadamard, Toeplitz matrices

**Mathematical context**: Linear algebra, signal processing, FFT algorithm.

**Prerequisites**: Notebook #1

---

### 2. Special Functions (25 min)

**File**: [`examples/applications/app_02_special_functions.ipynb`](../../examples/applications/app_02_special_functions.ipynb)

Classical special functions in :

- **Gamma function**: Poles at 0, -1, -2, ...
- **Riemann zeta**: Critical strip, non-trivial zeros (Riemann Hypothesis!)
- **Bessel functions**: Entire functions, wave equations
- **Elliptic functions**: Doubly periodic (Weierstrass )
- **Singularity classification**: Poles, zeros, essential singularities

**Mathematical context**: Number theory, physics, analysis.

**Prerequisites**: Notebooks #1, #6

---

### 3. Conformal Mappings (25 min)

**File**: [`examples/applications/app_03_conformal_maps.ipynb`](../../examples/applications/app_03_conformal_maps.ipynb)

Angle-preserving transformations:

- **Möbius transformations**: Linear fractional, circles ’ circles
- **Joukowsky transform**: Circles ’ airfoils (aerodynamics!)
- **Exponential/logarithm**: Strip ” plane
- **Power functions**: Angle multiplication
- **Composition**: Build complex from simple

**Uses `pair_plot()` to show domain ’ codomain transformation.**

**Applications**: Fluid dynamics, electrostatics, cartography.

**Prerequisites**: Notebooks #1, #2

---

### 4. Complex Dynamics & Fractals (35 min)

**File**: [`examples/applications/app_04_complex_dynamics.ipynb`](../../examples/applications/app_04_complex_dynamics.ipynb)

Iterative systems and chaos:

- **Newton's method**: Fractal basin boundaries (z³-1, zu-1)
- **Julia sets**: Iteration of z² + c
- **Mandelbrot set**: Parameter space catalog
- **Escape time coloring**: Fractal visualization
- **Chaos theory**: Sensitive dependence, self-similarity

**Mathematical context**: Dynamical systems, fractal geometry, holomorphic dynamics.

**Prerequisites**: Notebooks #1, #4

---

## Quick Navigation

| Goal | Notebook |
|------|----------|
| **Complete beginner** | Start with #1, follow sequence |
| **Just want colormaps** | Jump to #4 |
| **Need 3D/PyVista** | #3, then #6-7 |
| **3D printing** | #8 (requires #6-7) |
| **Check accessibility** | #5 (CRITICAL before publishing!) |
| **Real-world examples** | Applications folder |

## Running the Notebooks

### Install Dependencies

```bash
# Core dependencies (required)
uv pip install -e "."

# With all optional features (recommended)
uv pip install -e ".[all]"

# Minimal (excludes PyVista)
uv pip install -e ".[dev]"
```

### Launch Jupyter

```bash
jupyter notebook examples/notebooks/
```

Or use JupyterLab:
```bash
jupyter lab examples/notebooks/
```

### Optional: CVD Simulation

For accessibility testing (Notebook #5):
```bash
uv pip install colorspacious
```

## Tips for Learning

### Recommended Path

1. **Tutorial sequence (8 notebooks)**: Follow in order for comprehensive understanding
2. **Applications**: Pick based on interest after tutorials
3. **Interactive scripts**: Try `examples/interactive_showcase.py` for hands-on exploration

### Estimated Time

- **Tutorials**: ~2.5 hours total
- **Applications**: ~1.5 hours total
- **Combined**: ~4 hours for complete coverage

### Using the Standard Demo Function

Most notebooks use:
```python
def f(z):
    return (z - 1) / (z**2 + z + 1)
```

This function has:
- Two poles (denominators zeros)
- One zero (numerator zero)
- Interesting structure without being too complex

### PyVista Modes

**During development/learning:**
```python
cp.plot_landscape_pv(..., notebook=True, show=True)
```
- Non-blocking
- Inline display
- Good enough quality

**For publications:**
```python
cp.plot_landscape_pv(..., notebook=False, show=True)
```
- High quality
- External interactive window
- Blocks until closed

## Troubleshooting

### Import Errors

**"No module named complexplorer":**
```bash
cd ../..  # Go to repo root
uv pip install -e ".[all]"
```

**"No module named pyvista":**
```bash
uv pip install pyvista
```

**"No module named colorspacious":**
```bash
uv pip install colorspacious
```

### PyVista Issues

**Blank output in Jupyter:**
- Use `notebook=True` for inline
- Or use `notebook=False` and check external window

**Low quality inline:**
- This is expected with `notebook=True`
- Use `notebook=False` for high quality
- Or run as script instead of notebook

### Performance

**Slow 3D rendering:**
- Use PyVista functions (`*_pv`) not matplotlib
- Reduce `resolution` parameter
- Ensure PyVista is properly installed

## Contributing

Found an error or have an improvement?

1. Open an issue on GitHub
2. Submit a pull request
3. Follow the notebook style guide in CONTRIBUTING.md

## Related Documentation

- [User Guide](../user-guide/index.md) - API reference
- [Gallery](gallery.md) - Visual examples
- [Installation](../installation.md) - Setup guide

---

**Ready to start?** Open [`01_getting_started.ipynb`](../../examples/notebooks/01_getting_started.ipynb) and begin your journey! <=Ð
