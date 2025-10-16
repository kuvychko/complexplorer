# Complexplorer Examples

This directory contains comprehensive tutorials and applications for the complexplorer library.

## 📚 Learning Path

### Core Tutorial Notebooks (`notebooks/`)

**Recommended order for learning:**

1. **Getting Started** (`01_getting_started.ipynb`) - ~10 min
   - Your first complex visualization
   - Basic domains (Rectangle, Disk, Annulus)
   - Enhanced phase portraits
   - 2D plotting fundamentals
   - **START HERE if you're new!**

2. **Advanced Domains** (`02_domains_advanced.ipynb`) - ~15 min
   - Domain set operations (union, intersection, difference)
   - Complex compositions
   - Custom domain design
   - Practical examples

3. **PyVista 3D** (`03_pyvista_3d.ipynb`) - ~20 min
   - High-performance 3D visualization
   - Inline vs interactive modes
   - Pair plots (domain + codomain)
   - Performance tips

4. **Colormaps Comprehensive** (`04_colormaps_comprehensive.ipynb`) - ~25 min
   - All 13 colormap families
   - Parameters and customization
   - Use case guide
   - Comparison gallery

5. **Accessibility & CVD** (`05_accessibility_cvd.ipynb`) - ~20 min
   - Color vision deficiency simulation
   - CVD-friendly colormap selection
   - Design guidelines
   - **CRITICAL for publications!**

6. **Riemann Sphere** (`06_riemann_sphere.ipynb`) - ~30 min
   - Stereographic projection
   - Extended complex plane ℂ ∪ {∞}
   - 2D hemisphere charts
   - 3D sphere visualization

7. **Modulus Scaling** (`07_modulus_scaling.ipynb`) - ~25 min
   - 10+ scaling modes
   - Parameter exploration
   - Custom scaling functions
   - Use case guide

8. **STL Export** (`08_stl_export.ipynb`) - ~30 min
   - 3D printing workflow
   - **Bisection in slicer** (recommended!)
   - Gluing tips
   - Post-processing

**Total tutorial time: ~2.5 hours**

### Application Notebooks (`applications/`)

Real-world examples showcasing library capabilities:

1. **FFT/DFT Matrices** (`app_01_fft_matrices.ipynb`) - ~15 min
   - Discrete Fourier Transform visualization
   - 2×4 grid gallery (N = 32 to 4096)
   - Circular symmetry from roots of unity
   - Other structured matrices (Hadamard, Toeplitz)

2. **Special Functions** (`app_02_special_functions.ipynb`) - ~25 min
   - Gamma function (poles at negative integers)
   - Riemann zeta (critical strip, famous zeros)
   - Bessel functions (entire functions)
   - Elliptic functions (doubly periodic)
   - Singularity classification

3. **Conformal Mappings** (`app_03_conformal_maps.ipynb`) - ~25 min
   - Möbius transformations
   - Joukowsky airfoil transform
   - Exponential and logarithm
   - Power functions
   - Composition techniques

4. **Complex Dynamics** (`app_04_complex_dynamics.ipynb`) - ~35 min
   - Newton's method basins of attraction
   - Julia sets
   - Mandelbrot set
   - Fractals and chaos
   - Escape time algorithms

## 🖥️ Interactive Scripts

### Interactive Showcase (`interactive_showcase.py`)
Comprehensive menu-driven demo:
```bash
python interactive_showcase.py
```

Features:
- 2D phase portraits
- 3D landscapes (PyVista)
- Riemann sphere visualizations
- STL export for 3D printing
- Batch processing

### Gallery Generator (`generate_gallery.py`)
Create image gallery with code:
```bash
python generate_gallery.py [output_directory]
```

### Other Scripts
- `new_colormaps_showcase.py` - Colormap comparison
- `modulus_scaling_showcase.py` - Scaling mode comparison

## 🎯 Quick Start

| If you want to... | Start here... |
|-------------------|---------------|
| Learn the basics | `notebooks/01_getting_started.ipynb` |
| Explore interactively | `python interactive_showcase.py` |
| See all colormaps | `notebooks/04_colormaps_comprehensive.ipynb` |
| 3D printing guide | `notebooks/08_stl_export.ipynb` |
| Check accessibility | `notebooks/05_accessibility_cvd.ipynb` |
| Real-world examples | `applications/` folder |

## 💡 Important Tips

### PyVista Rendering Modes

**Inline mode** (development, non-blocking):
```python
cp.plot_landscape_pv(domain, func, notebook=True, show=True)
```
- Displays inline in Jupyter
- Non-blocking execution
- Good for development workflow

**Interactive mode** (publication quality, blocking):
```python
cp.plot_landscape_pv(domain, func, notebook=False, show=True)
```
- External window with full interactivity
- High quality, better anti-aliasing
- **Blocks execution until window is closed**

### Performance Guidelines

- **2D plots**: Use matplotlib (standard `plot()`)
- **3D plots**: Always use PyVista (`*_pv` functions)
- PyVista is **15-30x faster** than matplotlib 3D

### STL Export Best Practices

**Recommended workflow:**
1. Generate full sphere STL
2. Bisect at equator in slicer (PrusaSlicer/Cura)
3. Print hemispheres separately (no supports!)
4. Glue together

See `notebooks/08_stl_export.ipynb` for detailed guide.

### Color Blindness

~8% of males have some form of color vision deficiency!

**CVD-friendly colormaps:**
- ✅ `CubehelixPhase` (best)
- ✅ `PerceptualPastel`
- ✅ `InkPaper`

**Test your figures:**
See `notebooks/05_accessibility_cvd.ipynb` for simulation tools.

## 📁 Directory Structure

```
examples/
├── README.md                    # This file
├── notebooks/                   # Tutorial notebooks (8)
│   ├── README.md               # Learning guide
│   ├── 01_getting_started.ipynb
│   ├── 02_domains_advanced.ipynb
│   ├── 03_pyvista_3d.ipynb
│   ├── 04_colormaps_comprehensive.ipynb
│   ├── 05_accessibility_cvd.ipynb
│   ├── 06_riemann_sphere.ipynb
│   ├── 07_modulus_scaling.ipynb
│   └── 08_stl_export.ipynb
├── applications/                # Application notebooks (4)
│   ├── README.md               # Application guide
│   ├── app_01_fft_matrices.ipynb
│   ├── app_02_special_functions.ipynb
│   ├── app_03_conformal_maps.ipynb
│   └── app_04_complex_dynamics.ipynb
├── interactive_showcase.py      # Interactive demo
├── generate_gallery.py          # Gallery generator
├── new_colormaps_showcase.py    # Colormap comparison
├── modulus_scaling_showcase.py  # Scaling comparison
├── gallery/                     # Generated images
└── old/                         # Archived examples
    └── README.md               # Archive notes
```

## 🔗 Additional Resources

- [Complexplorer Documentation](../docs/)
- [Visual Complex Functions](http://www.visual.wegert.com/) - Book by Elias Wegert
- [PyVista Documentation](https://docs.pyvista.org/)
- [colorspacious](https://colorspacious.readthedocs.io/) - For CVD simulation

## 🐛 Troubleshooting

### "No module named complexplorer"
Install in development mode:
```bash
cd ..  # Go to repository root
uv pip install -e ".[all]"  # Install with all optional dependencies
```

### PyVista Issues

**Window doesn't appear:**
- Use `notebook=False` for external window
- Check display available (X11, Wayland, etc.)
- Update PyVista: `uv pip install -U pyvista`

**Low quality in Jupyter:**
- Always use `notebook=False` for high quality
- Inline backend has aliasing issues
- Or use scripts instead of notebooks

### Color Vision Deficiency Simulation

**"No module named colorspacious":**
```bash
uv pip install colorspacious
```

Required for `notebooks/05_accessibility_cvd.ipynb`

## 📝 Version 2.0 Highlights

These examples showcase v2.0 features:
- ✨ Enhanced phase portraits with `auto_scale_r=True`
- ✨ Domain set operations (union, intersection, difference)
- ✨ New perceptual colormaps (OkLCh-based)
- ✨ Comprehensive accessibility testing
- ✨ 10+ modulus scaling modes
- ✨ STL bisection workflow

See [CHANGELOG.md](../CHANGELOG.md) for full release notes.

## 🤝 Contributing Examples

Have an interesting use case? We welcome contributions!

1. Follow the notebook structure in `notebooks/`
2. Include estimated time and prerequisites
3. Use the standard demo function where possible
4. Test accessibility with CVD simulation
5. Submit a pull request

## 📖 Citation

If you use complexplorer in research, please cite:
- Complexplorer library (see main README)
- Wegert, E. (2012). "Visual Complex Functions: An Introduction with Phase Portraits"

Happy exploring! 🌈📐
