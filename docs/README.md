# Complexplorer Documentation

Welcome to the Complexplorer documentation! This library provides elegant visualizations for complex-valued functions.

## 📚 Documentation Sections

### 🎨 [Gallery](gallery/README.md)
Explore beautiful visualizations with code examples. See phase portraits, analytic landscapes, and Riemann sphere projections.

### 📖 Tutorials
- [Basic Usage](../examples/plots_example.ipynb) - Getting started with complexplorer
- [Domains & Color Maps](../examples/domains_cmaps_example.ipynb) - Advanced visualization techniques

### 🔧 API Reference
- **Domains**: `Rectangle`, `Disk`, `Annulus` - Define regions in the complex plane
- **Color Maps**: `Phase`, `Chessboard`, `PolarChessboard`, `LogRings` - Map complex values to colors
- **2D Plots**: `plot`, `pair_plot`, `riemann_chart`, `riemann_hemispheres`
- **3D Plots**: `plot_landscape`, `pair_plot_landscape`, `riemann`

### 📐 Mathematical Background
Complex function visualization relies on several key concepts:
- **Phase portraits**: Color represents the argument (angle) of complex numbers
- **Enhanced portraits**: Additional structure shows modulus information
- **Riemann sphere**: Compactification of the complex plane including infinity

## 🚀 Quick Start

```python
import complexplorer as cp

# Create a domain
domain = cp.Rectangle(4, 4)

# Define a function
f = lambda z: (z - 1) / (z**2 + z + 1)

# Visualize!
cp.plot(domain, f)
```

## 🤝 Getting Help

- Check the [Gallery](gallery/README.md) for examples
- Use `help(function_name)` for detailed documentation
- Report issues on [GitHub](https://github.com/kuvychko/complexplorer/issues)

## 🔮 Future Plans

- Interactive visualizations with PyVista
- Export to STL for 3D printing
- More special functions in the gallery
- Performance optimizations for large-scale visualizations