# Complexplorer

[![PyPI version](https://badge.fury.io/py/complexplorer.svg)](https://badge.fury.io/py/complexplorer)
[![Python](https://img.shields.io/pypi/pyversions/complexplorer.svg)](https://pypi.org/project/complexplorer/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*We cannot directly see the minute details of a Dedekind cut, nor is it clear that arbitrarily great or
arbitrarily tiny times or lengths actually exist in nature. One could say that 
the so-called ‘real numbers’ are as much a product of mathematicians’ 
imaginations as are the complex numbers. Yet we shall find that complex 
numbers, as much as reals, and perhaps even more, find a unity with 
nature that is truly remarkable. It is as though Nature herself is as 
impressed by the scope and consistency of the complex-number system 
as we are ourselves, and has entrusted to these numbers the precise 
operations of her world at its minutest scales.* ...

*Moreover, to refer just to the scope and to the consistency of complex 
numbers does not do justice to this system. There is something more 
which, in my view, can only be referred to as ‘magic’.*

[Road to Reality](https://www.ams.org/notices/200606/rev-blank.pdf), Chapter 4 - Magical Complex Numbers, Sir Roger Penrose

**Complexplorer** is a Python library for elegant visualization of complex-valued functions. Create stunning phase portraits, analytic landscapes, and Riemann sphere projections with just a few lines of code.

<p align="center">
  <img src="examples/gallery/Enhanced_phase_portrait_phase_and_modulus_enhanced_2d.png" width="45%">
  <img src="examples/gallery/riemann_sphere_3d.png" width="45%">
</p>

## ✨ Features

- **🎨 Rich visualization options**: Phase portraits, enhanced phase portraits, chessboard patterns, and more
- **🗺️ Flexible domains**: Rectangles, disks, annuli, and custom domains via composition
- **📊 Multiple plot types**: 2D images, 3D analytic landscapes, Riemann sphere projections
- **🖨️ 3D Printing Support**: Export complex function visualizations as STL files for 3D printing
- **🧩 Composable design**: Mix any domain, color map, and plot type
- **🚀 Lightweight**: Requires only NumPy and Matplotlib

## 📦 Installation

**Requirements**: Python 3.11 or higher

```bash
pip install complexplorer

# Optional: For interactive matplotlib plots in CLI scripts
pip install "complexplorer[qt]"

# Optional: For high-performance 3D visualizations
pip install "complexplorer[pyvista]"

# Optional: Install everything
pip install "complexplorer[all]"
```

## 🚀 Quick Start

```python
import complexplorer as cp
import numpy as np

# Define a complex function
def f(z):
    return (z - 1) / (z**2 + z + 1)

# Create a domain
domain = cp.Rectangle(3, 3)

# Visualize!
cp.plot(domain, f)
```

## 🎨 Gallery

Explore the full range of visualizations in our [**Gallery**](docs/gallery/README.md), featuring:
- Phase portraits with various enhancements
- Chessboard and polar patterns  
- 3D analytic landscapes
- Riemann sphere projections

<p align="center">
  <a href="docs/gallery/README.md">
    <img src="examples/gallery/Polar_chessboard_log_modulus_spacing_2d.png" width="30%">
    <img src="examples/gallery/Phase_portrait_phase_enhanced_3d.png" width="30%">
    <img src="examples/gallery/riemann_chart_2d.png" width="30%">
  </a>
</p>

## 📚 Documentation

- **[Gallery](docs/gallery/README.md)** - Visual showcase with code examples
- **[Getting Started](examples/getting_started.ipynb)** - Beginner-friendly introduction
- **[Advanced Features](examples/advanced_features.ipynb)** - 3D visualization, STL export, and more
- **[API Cookbook](examples/api_cookbook.ipynb)** - Ready-to-use code recipes
- **[Interactive Demo](examples/interactive_showcase.py)** - Run `python examples/interactive_showcase.py`
- **API Reference** - Use `help()` on any function or class

## 🛠️ Advanced Example

```python
# Create an enhanced phase portrait with auto-scaling for square cells
domain = cp.Annulus(0.5, 2, center=1j)  # Annular domain
cmap = cp.Phase(n_phi=6, auto_scale_r=True, v_base=0.4)  # Auto-scaled enhanced phase

# 2D visualization with domain and codomain side-by-side
cp.pair_plot(domain, f, cmap=cmap, figsize=(10, 5))

# 3D analytic landscape
cp.plot_landscape(domain, f, cmap=cmap, z_scale=0.3)

# 3D landscape with modulus scaling for better visualization
cp.plot_landscape(domain, f, cmap=cmap, modulus_mode='arctan')

# Riemann sphere projection
cp.riemann(f, resolution=800, cmap=cmap)
```

### 🚀 High-Performance 3D Visualizations with PyVista

For interactive, high-quality 3D visualizations, Complexplorer includes PyVista-based plotting functions:

```python
# High-performance 3D landscape
cp.plot_landscape_pv(domain, f, cmap=cmap, notebook=False, show=True)

# Interactive Riemann sphere with modulus scaling
cp.riemann_pv(f, modulus_scaling='arctan', notebook=False, show=True)
```

**⚠️ Important Note:** For best quality, we strongly recommend using PyVista visualizations via command-line scripts rather than Jupyter notebooks. The Jupyter backend (trame) has significant aliasing issues that cannot be compensated with higher resolution. See `examples/interactive_demo.py` for an excellent CLI-based interactive experience.

### 📊 Modulus Scaling for 3D Landscapes

Control how the magnitude (modulus) of complex values is mapped to height in 3D visualizations:

```python
# Different scaling modes for various visualization needs
cp.plot_landscape(domain, f, modulus_mode='constant')     # Phase only (flat)
cp.plot_landscape(domain, f, modulus_mode='arctan')       # Smooth bounded scaling
cp.plot_landscape(domain, f, modulus_mode='logarithmic')  # Emphasize poles/zeros
cp.plot_landscape(domain, f, modulus_mode='adaptive')     # Auto-adjust to data

# Custom scaling function for specific needs
def custom_scale(moduli):
    return np.tanh(moduli / 2)  # Custom transformation

cp.plot_landscape(domain, f, modulus_mode='custom', 
                 modulus_params={'scaling_func': custom_scale})
```

Available modes: `none`, `constant`, `linear`, `arctan`, `logarithmic`, `linear_clamp`, `power`, `sigmoid`, `adaptive`, `hybrid`, `custom`. See `examples/modulus_scaling_showcase.py` for comprehensive examples.

### 🎯 Domain Restrictions

Control numerical stability and focus visualizations on regions of interest by restricting to specific domains:

```python
# Avoid infinity at large distances
domain = cp.Disk(radius=5, center=0)
cp.riemann_pv(f, domain=domain, scaling='arctan')

# Exclude origin for functions with poles
domain = cp.Annulus(inner_radius=0.1, outer_radius=10, center=0)
ornament = cp.OrnamentGenerator(func=lambda z: 1/z, domain=domain)
```

Domain restrictions work with all visualization functions and are especially useful for:
- Functions with essential singularities
- Focusing on specific regions of the complex plane
- Improving numerical stability in STL generation
- Creating cleaner 3D prints by excluding problematic areas

### 🖨️ 3D Printing Support

Transform your complex function visualizations into physical objects! Complexplorer can export Riemann sphere visualizations as STL files suitable for 3D printing:

```python
# STL export is available with PyVista installed
from complexplorer.export.stl import OrnamentGenerator

# Create STL files from your function
ornament = OrnamentGenerator(
    func=lambda z: (z - 1) / (z**2 + z + 1),
    resolution=150,
    scaling='arctan',
    cmap=cp.Phase(n_phi=12, auto_scale_r=True)
)

# Generate print-ready STL file
stl_file = ornament.generate_ornament(
    output_file='complex_ornament.stl',
    size_mm=80,
    smooth=True
)
```

Features:
- Automatic mesh healing for watertight models
- Flat bisection planes for easy printing without supports
- Multiple modulus scaling methods
- Domain restrictions to avoid numerical instabilities
- Intelligent handling of singularities through neighbor interpolation
- Compatible with all complexplorer colormaps

See `examples/advanced_features.ipynb` for interactive STL export examples.

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs or suggest features via [Issues](https://github.com/kuvychko/complexplorer/issues)
- Submit pull requests with improvements
- Share your visualizations and examples
- Improve documentation

## 📖 Citation

If you use Complexplorer in your research, please cite:

```bibtex
@software{complexplorer,
  author = {Igor Kuvychko},
  title = {Complexplorer: A Python library for visualization of complex functions},
  url = {https://github.com/kuvychko/complexplorer},
  year = {2024}
}
```

## 🙏 Acknowledgments

This library was inspired by Elias Wegert's beautiful book ["Visual Complex Functions"](https://link.springer.com/book/10.1007/978-3-0348-0180-5) and benefited greatly from his feedback and suggestions.

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.