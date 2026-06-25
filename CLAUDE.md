# CLAUDE.md - Complexplorer Project Guide

## Project Overview

Complexplorer is a Python library for visualization of complex functions, inspired by Elias Wegert's book "Visual Complex Functions - An Introduction with Phase Portraits". The library provides tools to create beautiful visualizations of complex-valued functions using various color mapping techniques and plot types, and can export 3D-printable mathematical ornaments (STL).

## Key Concepts

- **Complex Domains**: Rectangular, disk, and annular regions in the complex plane, plus composite domains built via set operations (union, intersection, difference)
- **Color Maps**: Schemes to visualize complex values — classic Phase portraits, grayscale Chessboard/PolarChessboard/LogRings patterns, and a family of perceptually-uniform colormaps (OKLAB/OkLCh, cubehelix, etc.)
- **Modulus Scaling**: ~10 transfer functions mapping `|f(z)|` to radius/height, used by 3D landscapes, Riemann relief, and STL export
- **Visualization Types**: 2D plots, 3D analytic landscapes, and Riemann sphere projections (matplotlib and PyVista backends)
- **STL Export**: Modulus-scaled Riemann sphere ornaments for 3D printing

## Project Structure

```
complexplorer/
├── complexplorer/              # Main library package
│   ├── __init__.py             # Public API surface (see __all__)
│   ├── _version.py
│   ├── api.py                  # High-level API: show(), plot(), *_preset()
│   ├── exceptions.py           # ComplexplorerError hierarchy
│   ├── core/
│   │   ├── domain.py           # Domain, Rectangle, Disk, Annulus, CompositeDomain
│   │   ├── colormap.py         # Colormap base + Phase, OklabPhase, Chessboard, … (14 colormaps)
│   │   ├── scaling.py          # ModulusScaling modes + get_scaling_preset()
│   │   ├── functions.py        # phase, sawtooth, stereographic_projection, …
│   │   ├── color_utils.py      # OkLCh/HSL/cubehelix color conversions
│   │   └── constants.py
│   ├── plotting/
│   │   ├── matplotlib/
│   │   │   └── plot_2d.py       # plot, pair_plot, riemann_chart, riemann_hemispheres (2D only)
│   │   ├── pyvista/
│   │   │   ├── plot_3d.py       # plot_landscape_pv, pair_plot_landscape_pv
│   │   │   └── riemann.py       # riemann_pv
│   │   └── validation.py        # Shared plot input validation
│   ├── export/
│   │   └── stl/                 # OrnamentGenerator, create_ornament, mesh repair
│   └── utils/
│       ├── backend.py          # Matplotlib backend detection/setup
│       ├── mesh.py             # Mesh generation (incl. Riemann sphere)
│       ├── logging.py
│       └── validation.py
├── openspec/                   # OpenSpec specs and change proposals (see below)
├── examples/                   # Example notebooks and output images
├── tests/unit/                 # Unit tests (pytest)
├── pyproject.toml
└── README.md
```

> Note: the public API is re-exported flat from the top-level package (e.g. `cp.Rectangle`, `cp.Phase`, `cp.plot`). Import from `complexplorer` directly; the submodule layout above is internal organization.

## Spec-Driven Development (OpenSpec)

This project uses **OpenSpec** for managing changes. Baseline behavioral-contract specs for the existing system live in `openspec/specs/`, organized into 10 capabilities: `domains`, `colormaps`, `modulus-scaling`, `core-functions`, `plotting-2d`, `plotting-3d-mpl`, `plotting-3d-pyvista`, `riemann-sphere`, `stl-export`, `high-level-api`.

For any non-trivial change, create an OpenSpec change proposal (`/opsx:propose` or `/opsx:explore`) rather than editing code directly, then implement against it. Project context for artifact generation lives in `openspec/config.yaml`. Validate specs with `openspec validate --specs`.

## Dependencies

- Python >= 3.11
- numpy >= 1.26.0
- matplotlib >= 3.8.0 (2D backend)
- scipy >= 1.11.0 (for mesh interpolation and signal processing)
- PyVista >= 0.47 (the 3D backend **and STL export** — a required dependency as of 3.0)

Optional dependencies:
- PyQt6 >= 6.5.0 (for interactive matplotlib plots in CLI scripts)

As of 3.0 PyVista is a required core dependency (the sole 3D backend), so the former
`HAS_PYVISTA` / `HAS_STL_EXPORT` capability flags have been removed — those features are
always available.

## Development Setup

The project uses `uv` for fast Python package management:

```bash
# Create and activate virtual environment (if not already done)
uv venv
source .venv/bin/activate

# Install the package in editable mode with all development dependencies
uv pip install -e ".[dev]"

# Or install with all optional dependencies (includes PyVista and PyQt6)
uv pip install -e ".[all]"

# Or install specific optional features
uv pip install -e ".[qt]"      # For interactive matplotlib plots
uv pip install -e ".[pyvista]" # For high-performance 3D + STL export
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=complexplorer --cov-report=html

# Run specific test file
pytest tests/unit/test_domain.py -v
```

## Development Guidelines

### Testing

The project has a comprehensive unit test suite in `tests/unit/`, run with pytest.

### Code Style

- Follow PEP 8 Python style guidelines
- Use descriptive variable names that reflect mathematical concepts
- Add comprehensive docstrings for all public functions and classes
- Include mathematical formulas in docstrings where applicable
- Use type hints
- Raise the library's domain-specific exceptions (subclasses of `ComplexplorerError` in `exceptions.py`) rather than bare `ValueError`/`RuntimeError`

### Common Tasks

1. **Adding a new color map**: Create a new class in `core/colormap.py` that inherits from `Colormap` (or `BasePhasePortrait` for enhanced phase portraits) and implements the HSV/RGB conversion contract
2. **Adding a new domain type**: Create a new class in `core/domain.py` that inherits from `Domain` and implements the `contains()` method
3. **Adding a new modulus scaling mode**: Add a static method to `ModulusScaling` in `core/scaling.py` (and a preset in `get_scaling_preset()` if warranted)
4. **Adding a new plot type**: Add functions to `plotting/matplotlib/` or `plotting/pyvista/` following existing patterns
5. **Remember**: capture the intended behavior in an OpenSpec change first (see above)

### Mathematical Background

The library deals with complex functions f: ℂ → ℂ. Key conventions:
- **Phase**: `arg(z)` normalized to `[0, 2π)`, mapped to hue
- **Modulus**: `|z|` used for brightness, contour patterns, or 3D height/radius
- **Stereographic projection**: maps the complex plane to the unit Riemann sphere; the origin and ∞ occupy opposite poles, the unit circle maps to the equator
- **Enhanced phase portraits**: show both phase and modulus information via sawtooth-modulated brightness
- **Domain membership** boundaries are inclusive; colormap RGB/HSV channels are in `[0, 1]`

### Performance Considerations

- All 3D rendering goes through PyVista (the matplotlib 3D backend was removed in 3.0),
  which is high-quality and fast
- The Riemann sphere mesh is rectangular, which is inefficient at poles
- Domain meshing is deferred until plot time for flexibility

### PyVista Integration

High-performance PyVista-based functions:
- `plot_landscape_pv()`: Fast 3D landscape visualization
- `pair_plot_landscape_pv()`: Side-by-side domain/codomain comparison
- `riemann_pv()`: Riemann sphere with multiple modulus scaling options

**Important**: For best quality, use PyVista functions via command-line scripts rather than Jupyter notebooks. The Jupyter trame backend has severe aliasing issues. See the `examples/` directory for interactive demos.

### Future Improvements

- Animation capabilities for parameter exploration
- Optimized viewing windows for domain intersections
- Documentation framework (Sphinx/MkDocs)
- Additional mesh generation options (e.g. geodesic spheres to avoid polar artifacts)

## Quick Reference

### Basic Usage Pattern

```python
import complexplorer as cp

# Define domain
domain = cp.Rectangle(re_length=4, im_length=4)

# Define function
def f(z):
    return (z - 1) / (z**2 + z + 1)

# Choose color map (auto-scaled enhanced phase)
cmap = cp.Phase(phase_sectors=6, auto_scale_r=True)

# Create visualization (note: cmap is a keyword argument)
cp.plot(domain, f, cmap=cmap)
```

### Quick Exploration (high-level API)

```python
import complexplorer as cp

# One-liner with sensible defaults
cp.show(lambda z: (z**2 - 1) / (z**2 + 1))

# Bundled presets
cp.plot(lambda z: 1/z, **cp.publication_preset())
```

### Common Color Maps

- `Phase()`: Basic or enhanced phase portraits
  - Use `auto_scale_r=True` for automatic square cell sizing
  - Set `phase_sectors` for the number of phase sectors
  - Adjust `scale_radius` to control cell size
- `Chessboard()`: Cartesian grid pattern
- `PolarChessboard()`: Polar grid pattern
- `LogRings()`: Logarithmic black/white rings
- Perceptual family: `OklabPhase()`, `PerceptualPastel()`, `AnalogousWedge()`, `DivergingWarmCool()`, `Isoluminant()`, `CubehelixPhase()`, `InkPaper()`, `EarthTopographic()`, `FourQuadrant()`

### Plot Types

#### Matplotlib-based (2D only):
- `plot()`: Basic 2D visualization
- `pair_plot()`: Side-by-side domain and codomain
- `riemann_chart()` / `riemann_hemispheres()`: Flat stereographic hemisphere charts

#### PyVista-based (all 3D; the matplotlib 3D functions were removed in 3.0):
- `plot_landscape_pv()`: 3D analytic landscape
- `pair_plot_landscape_pv()`: Side-by-side domain/codomain 3D
- `riemann_pv()`: Interactive Riemann **sphere** (a single-valued function on the compactified plane)
- `riemann_surface_pv()`: Riemann **surface** of a multivalued family — the multi-sheeted cover (`power` roots `z^(1/n)`, or `log`). Distinct from the sphere: this is the surface on which a multivalued function becomes single-valued.

### STL Export (3D printing)

```python
from complexplorer.export.stl import OrnamentGenerator

ornament = OrnamentGenerator(lambda z: z / (z**10 - 1), resolution=200)
ornament.generate_and_save("ornament.stl", size_mm=80)
```
