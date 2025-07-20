# CLAUDE.md - Complexplorer Project Guide

## Project Overview

Complexplorer is a Python library for visualization of complex functions, inspired by Elias Wegert's book "Visual Complex Functions - An Introduction with Phase Portraits". The library provides tools to create beautiful visualizations of complex-valued functions using various color mapping techniques and plot types.

## Key Concepts

- **Complex Domains**: Rectangular, disk, and annular regions in the complex plane
- **Color Maps**: Various schemes to visualize complex values (Phase portraits, Chessboard patterns, Logarithmic rings)
- **Visualization Types**: 2D plots, 3D analytic landscapes, and Riemann sphere projections

## Project Structure

```
complexplorer/
├── complexplorer/          # Main library package
│   ├── __init__.py        # Package initialization
│   ├── domain.py          # Domain classes (Rectangle, Disk, Annulus)
│   ├── cmap.py            # Color map classes (Phase, Chessboard, etc.)
│   ├── funcs.py           # Supporting functions (phase, sawtooth, stereographic)
│   ├── plots_2d.py        # 2D plotting functions
│   └── plots_3d.py        # 3D plotting functions
├── examples/              # Example notebooks and output images
├── tests/                 # Unit tests with comprehensive coverage
├── pyproject.toml         # Project configuration
└── README.md              # Project documentation
```

## Dependencies

- Python >= 3.11
- numpy >= 1.26.0
- matplotlib >= 3.8.0

## Development Setup

The project uses `uv` for fast Python package management. To set up the development environment:

```bash
# Create and activate virtual environment (if not already done)
uv venv
source .venv/bin/activate

# Install the package in editable mode with all development dependencies
uv pip install -e ".[dev]"

# Or install with all optional dependencies (includes PyVista)
uv pip install -e ".[all]"
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

The project has a comprehensive unit test suite with 116 tests covering all major functionality. Tests are located in the `tests/` directory and can be run using pytest.

### Code Style

- Follow PEP 8 Python style guidelines
- Use descriptive variable names that reflect mathematical concepts
- Add comprehensive docstrings for all public functions and classes
- Include mathematical formulas in docstrings where applicable
- Use type hints

### Common Tasks

1. **Adding a new color map**: Create a new class in `cmap.py` that inherits from `Cmap` and implements `hsv()` and `rgb()` methods
2. **Adding a new domain type**: Create a new class in `domain.py` that inherits from `Domain` and implements the `contains()` method
3. **Adding a new plot type**: Add functions to either `plots_2d.py` or `plots_3d.py` following existing patterns

### Mathematical Background

The library deals with complex functions f: ℂ → ℂ. Key mathematical concepts:
- Phase: arg(z) mapped to colors
- Modulus: |z| used for brightness or patterns
- Stereographic projection: Maps complex plane to Riemann sphere
- Enhanced phase portraits: Show both phase and modulus information

### Performance Considerations

- Matplotlib is not optimized for 3D rendering, so 3D plots can be slow
- PyVista provides 15-30x faster 3D rendering with better quality
- The Riemann sphere plot uses a rectangular mesh which is inefficient at poles
- Domain meshing is deferred until plot time for flexibility

### PyVista Integration

The project includes high-performance PyVista-based 3D plotting functions:
- `plot_landscape_pv()`: Fast 3D landscape visualization
- `pair_plot_landscape_pv()`: Side-by-side domain/codomain comparison
- `riemann_pv()`: Riemann sphere with multiple modulus scaling options

**Important**: For best quality, use PyVista functions via command-line scripts rather than Jupyter notebooks. The Jupyter trame backend has severe aliasing issues. See `examples/interactive_demo.py` for an optimal interactive experience.

### Future Improvements (from README)

- Auto-tuning of modulus discretization for square enhanced regions
- PyVista implementation for 3D plots
- Triangular meshing for Riemann sphere
- STL file export for 3D printing
- Optimized viewing windows for domain intersections

## Quick Reference

### Basic Usage Pattern

```python
import complexplorer as cp

# Define domain
domain = cp.Rectangle(re_length=4, im_length=4)

# Define function
def f(z):
    return (z - 1) / (z**2 + z + 1)

# Choose color map
cmap = cp.Phase(n_phase=6, enhance_phase=True)

# Create visualization
cp.plot(domain, f, cmap)
```

### Common Color Maps

- `Phase()`: Basic or enhanced phase portraits
- `Chessboard()`: Cartesian grid pattern
- `PolarChessboard()`: Polar grid pattern
- `LogRings()`: Logarithmic black/white rings

### Plot Types

- `plot()`: Basic 2D visualization
- `pair_plot()`: Side-by-side domain and codomain
- `plot_landscape()`: 3D surface plot
- `riemann()`: Riemann sphere visualization