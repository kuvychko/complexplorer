# Complexplorer Documentation

Welcome to the Complexplorer documentation! This library provides elegant visualizations for complex-valued functions using advanced mathematical visualization techniques.

## 📚 Documentation Sections

### 🎨 [Gallery](gallery/README.md)
Explore beautiful visualizations with code examples. See phase portraits, analytic landscapes, and Riemann sphere projections.

### 📖 Tutorials
- [Getting Started](../examples/getting_started.ipynb) - Quick introduction to complexplorer
- [Advanced Features](../examples/advanced_features.ipynb) - Enhanced visualizations and PyVista
- [API Cookbook](../examples/api_cookbook.ipynb) - Code recipes and patterns

### 🔧 User Guides
- [PyVista Usage Guide](pyvista_usage_guide.md) - High-performance 3D visualization
- [STL Export Guide](stl_export_guide.md) - Creating 3D-printable ornaments
- [Migration Guide](../MIGRATION_GUIDE.md) - Upgrading from older versions

### 📐 Technical Reference
- [Technical Documentation](technical_reference.md) - In-depth technical details including:
  - Modulus scaling analysis
  - Icosphere mesh generation
  - PyVista implementation details
  - STL ornament generation

## 🚀 Quick Start

```python
import complexplorer as cp

# Create a domain
domain = cp.Rectangle(4, 4)

# Define a function
func = lambda z: (z - 1) / (z**2 + z + 1)

# Create enhanced phase portrait
cmap = cp.Phase(n_phi=6, auto_scale_r=True)

# Visualize!
cp.plot(domain, func, cmap=cmap)
```

## 🔍 Core Concepts

### Domains
- `Rectangle(re_length, im_length)` - Rectangular region
- `Disk(radius)` - Circular region
- `Annulus(inner_radius, outer_radius)` - Ring-shaped region

### Color Maps
- `Phase()` - Basic or enhanced phase portraits
- `Chessboard()` - Cartesian grid pattern
- `PolarChessboard()` - Polar grid pattern
- `LogRings()` - Logarithmic black/white rings

### Plotting Functions
**2D Visualization:**
- `plot()` - Basic domain coloring
- `pair_plot()` - Side-by-side domain and codomain
- `riemann_chart()` - Riemann hemisphere visualization

**3D Visualization (Matplotlib):**
- `plot_landscape()` - 3D surface plot
- `pair_plot_landscape()` - Side-by-side 3D plots
- `riemann()` - Riemann sphere visualization

**3D Visualization (PyVista) - 15-30x faster:**
- `plot_landscape_pv()` - High-performance 3D landscape
- `pair_plot_landscape_pv()` - Side-by-side with linked cameras
- `riemann_pv()` - Interactive Riemann sphere with modulus scaling

## 🎯 Interactive Demo

For the best experience with PyVista visualizations:

```bash
python examples/interactive_showcase.py
```

This provides a menu-driven interface with:
- 8 pre-defined complex functions
- 8 color schemes
- Multiple domain types
- Resolution and scaling controls

## 🤝 Getting Help

- Browse the [Gallery](gallery/README.md) for visual examples
- Check the [API Cookbook](../examples/api_cookbook.ipynb) for code patterns
- Read the [Technical Reference](technical_reference.md) for deep dives
- Report issues on [GitHub](https://github.com/user/complexplorer/issues)

## 📦 Installation

```bash
# Basic installation
pip install complexplorer

# With PyVista support
pip install "complexplorer[pyvista]"

# With all optional features
pip install "complexplorer[all]"
```

## 🔮 Recent Updates

- **Modular Architecture**: Clean separation of core, plotting, and export modules
- **PyVista Integration**: High-performance 3D visualization with GPU acceleration
- **STL Export**: Create 3D-printable ornaments from complex functions
- **Enhanced Phase Portraits**: Automatic modulus scaling for better visualization
- **Comprehensive Test Suite**: 330+ tests ensuring reliability

---

For more details, explore the documentation sections above or dive into the example notebooks!