# Tutorial Notebooks

This folder contains the main tutorial notebooks for Complexplorer v2.0, organized as a progressive learning path.

## Learning Path

### Core Tutorials (Complete in Order)

1. **01_getting_started.ipynb** (~10 min)
   - 2D visualization basics
   - Domains: Rectangle, Disk, Annulus
   - Enhanced phase portraits only
   - Your first complex function plots
   - **Prerequisites**: Python 3.11+, basic complex numbers

2. **02_domains_advanced.ipynb** (~15 min)
   - Domain set operations: union, intersection, difference
   - Complex domain composition
   - Viewing windows and masking
   - **Prerequisites**: Notebook #1

3. **03_pyvista_3d.ipynb** (~20 min)
   - High-performance 3D landscapes with PyVista
   - matplotlib vs PyVista comparison (15-30x faster)
   - Riemann sphere introduction
   - Inline vs interactive visualization modes
   - **Prerequisites**: Notebook #1, PyVista installed

4. **04_colormaps_comprehensive.ipynb** (~25 min)
   - All 13 colormap families showcased
   - Perceptually uniform colormaps (PerceptualPastel, Isoluminant, CubehelixPhase)
   - Artistic colormaps (AnalogousWedge, DivergingWarmCool, InkPaper, etc.)
   - Enhanced phase features: auto_scale_r, phase_sectors
   - Use case recommendations
   - **Prerequisites**: Notebook #1

5. **05_accessibility_cvd.ipynb** (~20 min)
   - **Color vision deficiency (CVD) simulation**
   - Protanopia, Deuteranopia, Tritanopia testing
   - Colormap accessibility rankings
   - Design guidelines for inclusive visualization
   - **Prerequisites**: Notebook #4, colorspacious library

6. **06_riemann_sphere.ipynb** (~30 min)
   - Stereographic projection mathematics
   - 2D hemisphere charts (matplotlib)
   - 3D sphere visualization (matplotlib & PyVista)
   - Functions with poles, zeros, essential singularities
   - **Prerequisites**: Notebooks #1, #3

7. **07_modulus_scaling.ipynb** (~25 min)
   - All 10+ modulus scaling modes explained
   - Parameter exploration (arctan, logarithmic, adaptive, etc.)
   - Custom scaling functions
   - Use cases for 3D landscapes and Riemann sphere
   - **Prerequisites**: Notebooks #3, #6

8. **08_stl_export.ipynb** (~30 min)
   - 3D printing workflow for mathematical ornaments
   - STL generation with OrnamentGenerator
   - **Slicer bisection workflow** (PrusaSlicer/Cura)
   - Domain restrictions for numerical stability
   - Gluing, painting, and finishing techniques
   - **Prerequisites**: Notebooks #6, #7

## Quick Access

- **New to Complexplorer?** Start with #1
- **Want 3D visualization?** Jump to #3 (after #1)
- **Choosing a colormap?** See #4
- **Creating accessible figures?** See #5
- **3D printing?** See #8

## Notes

- All notebooks use the new `phase_sectors` parameter (v2.0 breaking change)
- PyVista notebooks use `notebook=True` for inline display
  - Use `notebook=False` for interactive high-quality windows (blocks execution)
- Output cells are cleared by default (run to see results)
- Standard demo function: `f(z) = (z - 1) / (z**2 + z + 1)`

## See Also

- `../applications/` - Application examples (FFT, special functions, conformal maps, dynamics)
- `../` - Python scripts for showcases and interactive demos
- `../../docs/` - Full documentation and API reference
