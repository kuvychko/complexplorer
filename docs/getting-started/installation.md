# Installation

## Requirements

Complexplorer requires **Python 3.11 or higher**. You can check your Python version with:

```bash
python --version
```

## Basic Installation

Install Complexplorer from PyPI using pip:

```bash
pip install complexplorer
```

This installs the core library with its basic dependencies:

- `numpy` >= 1.26.0
- `matplotlib` >= 3.8.0
- `scipy` >= 1.11.0

## Optional Dependencies

Complexplorer offers several optional feature sets that you can install as needed:

### Interactive Matplotlib Plots (Recommended for CLI Scripts)

For interactive matplotlib plots in command-line scripts:

```bash
pip install "complexplorer[qt]"
```

This installs PyQt6 >= 6.5.0 for enhanced matplotlib interactivity.

### High-Performance 3D Visualizations (Highly Recommended)

For cinema-quality 3D visualizations with 15-30x better performance:

```bash
pip install "complexplorer[pyvista]"
```

This installs PyVista >= 0.45.0 for high-performance 3D rendering. **Strongly recommended** if you plan to:

- Create Riemann relief maps
- Work with high-resolution 3D landscapes
- Generate publication-quality 3D visualizations
- Export interactive 3D scenes

!!! tip "PyVista for Best Experience"
    PyVista provides dramatically better 3D rendering quality and performance compared to matplotlib. Use it via command-line scripts (not Jupyter notebooks) for optimal antialiasing and interactivity.

### All Optional Features

To install everything at once:

```bash
pip install "complexplorer[all]"
```

This installs all optional dependencies including PyQt6 and PyVista.

## Development Installation

If you want to contribute to Complexplorer or run tests, clone the repository and install in editable mode:

```bash
# Clone the repository
git clone https://github.com/kuvychko/complexplorer.git
cd complexplorer

# Create and activate virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode with development dependencies
pip install -e ".[dev]"
```

The `dev` extra includes:

- `pytest` for running tests
- `pytest-cov` for coverage reports
- All optional dependencies (PyQt6, PyVista)

### Using uv (Faster Alternative)

The project uses [`uv`](https://github.com/astral-sh/uv) for fast Python package management:

```bash
# Install uv if you don't have it
pip install uv

# Create virtual environment and install
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
```

## Verifying Installation

Test that Complexplorer is installed correctly:

```python
import complexplorer as cp
print(cp.__version__)

# Test basic functionality
domain = cp.Rectangle(2, 2)
f = lambda z: z**2
cp.plot(domain, f)
```

You should see version information and a plot window showing the phase portrait of z^2.

## Platform-Specific Notes

### Windows

- PyQt6 and PyVista install smoothly via pip
- For best PyVista performance, ensure you have updated graphics drivers

### macOS

- PyQt6 works out of the box
- PyVista may require XQuartz for off-screen rendering: `brew install --cask xquartz`

### Linux

- PyQt6 may require additional system packages:
  ```bash
  # Ubuntu/Debian
  sudo apt-get install python3-pyqt6

  # Fedora
  sudo dnf install python3-qt6
  ```
- PyVista should work without additional dependencies

## Upgrading from v1.x

If you're upgrading from Complexplorer v1.x to v2.0:

```bash
pip install --upgrade complexplorer
```

!!! warning "Breaking Changes in v2.0"
    Version 2.0 introduces breaking API changes. See the [Migration Guide](https://github.com/kuvychko/complexplorer/blob/main/MIGRATION_GUIDE_V2.md) for details on updating your code.

Key changes:

- `n_phi` parameter renamed to `phase_sectors`
- New colormap families (PerceptualPastel, Isoluminant, etc.)
- Enhanced phase portrait auto-scaling improvements
- Simplified API with `plot()` function

## Troubleshooting

### Import Errors

If you get import errors, ensure you're using Python 3.11+:

```bash
python --version
```

### PyVista Display Issues

If PyVista plots don't display:

1. **Jupyter notebooks**: Use `notebook=True` parameter or switch to command-line scripts
2. **Remote SSH**: Set `show=False` and use `filename='output.png'` to save plots
3. **Graphics drivers**: Update your graphics drivers for best performance

### Matplotlib Backend Issues

If matplotlib plots don't appear interactively:

```python
import matplotlib
matplotlib.use('QtAgg')  # or 'TkAgg', 'WXAgg'
```

## Next Steps

Now that you have Complexplorer installed, check out:

- [Quick Start Guide](quickstart.md) - Create your first visualization
- [User Guide](../user-guide/domains.md) - Learn about domains and colormaps
- [Examples Gallery](../examples/gallery.md) - Explore beautiful visualizations
