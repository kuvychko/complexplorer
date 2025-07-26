# Complexplorer Examples

This directory contains tutorials, examples, and interactive demos for the complexplorer library.

## 📚 Jupyter Notebooks

### 1. Getting Started (`getting_started.ipynb`)
- Installation and basic setup
- Your first complex function visualization
- Introduction to domains and colormaps
- Basic 2D and 3D plotting
- **Start here if you're new to complexplorer!**

### 2. Advanced Features (`advanced_features.ipynb`)
- Enhanced phase portraits with auto-scaling
- All colormap types in detail
- High-performance 3D visualization with PyVista
- Riemann sphere projections
- Performance optimization tips

### 3. STL Export Demo (`stl_export_demo.ipynb`)
- Step-by-step guide to 3D printing complex functions
- Visualize functions on the Riemann sphere
- Export as STL files with various scaling options
- Tips for successful 3D printing
- Custom scaling functions

### 4. API Cookbook (`api_cookbook.ipynb`)
- Gallery of common complex functions
- Domain manipulation patterns
- Colormap selection guide
- PyVista quality optimization
- Ready-to-use code recipes

## 🖥️ Interactive Scripts

### 1. Interactive Showcase (`interactive_showcase.py`)
A comprehensive menu-driven demo that includes:
- 2D phase portraits
- 3D landscapes (PyVista)
- Riemann sphere visualizations
- STL export for 3D printing
- Batch processing capabilities

Run with:
```bash
python interactive_showcase.py
```

### 2. Gallery Generator (`generate_gallery.py`)
Generates a complete gallery of visualizations with code snippets:
- Creates high-quality images
- Includes source code for each example
- Generates HTML index
- Perfect for documentation

Run with:
```bash
python generate_gallery.py [output_directory]
```

## 🎯 Quick Start Guide

1. **New to complexplorer?** Start with `getting_started.ipynb`
2. **Want to explore interactively?** Run `python interactive_showcase.py`
3. **Looking for specific examples?** Check `api_cookbook.ipynb`
4. **Need high-quality 3D plots?** See PyVista examples in `advanced_features.ipynb`

## 💡 Important Tips

### PyVista in Jupyter Notebooks
For high-quality 3D visualizations in Jupyter, always use `notebook=False`:

```python
plotter = cp.plot_landscape_pv(domain, func, notebook=False, show=True)
```

This opens an external window with full interactivity and anti-aliasing.

### Performance
- **2D plots**: Use matplotlib (built-in)
- **3D plots**: Always use PyVista functions (`*_pv`)
- PyVista is 15-30x faster than matplotlib for 3D

### STL Export
The library can export 3D-printable STL files. See `stl_export_demo.ipynb` for a complete guide:
```python
from complexplorer.export.stl import OrnamentGenerator
generator = OrnamentGenerator(func, resolution=150)
generator.generate_ornament('output.stl', size_mm=80)
```

## 📁 Directory Structure

```
examples/
├── README.md              # This file
├── getting_started.ipynb  # Beginner tutorial
├── advanced_features.ipynb # Advanced topics
├── stl_export_demo.ipynb  # 3D printing guide
├── api_cookbook.ipynb     # Code recipes
├── interactive_showcase.py # Interactive demo
├── generate_gallery.py    # Gallery generator
├── gallery/              # Gallery images and old generator
└── archive/              # Old examples (for reference)
```

## 🔗 Additional Resources

- [Complexplorer Documentation](https://github.com/YOUR_USERNAME/complexplorer)
- [Visual Complex Functions](http://www.visual.wegert.com/) - Book by Elias Wegert
- [PyVista Documentation](https://docs.pyvista.org/) - For 3D visualization

## 🐛 Troubleshooting

### "No module named complexplorer"
Make sure complexplorer is installed:
```bash
pip install -e ..  # From the examples directory
```

### PyVista window doesn't appear
- Ensure you're using `notebook=False` in Jupyter
- Check that you have a display available (X11, Wayland, etc.)
- Try updating PyVista: `pip install -U pyvista`

### Low quality 3D plots in Jupyter
- Always use `notebook=False` for external high-quality windows
- The inline notebook backend has severe aliasing issues