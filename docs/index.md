# Complexplorer

**Transform complex mathematics into tangible art.** Complexplorer brings complex function visualization into the physical world through stunning Riemann relief maps and 3D-printable mathematical ornaments.

!!! info "Version 2.0 Released!"
    Major improvements include 8 new perceptually-optimized colormaps, cleaner API, and enhanced performance.
    See the [Migration Guide](https://github.com/kuvychko/complexplorer/blob/main/MIGRATION_GUIDE_V2.md) if upgrading from v1.x.

## What Makes Complexplorer Unique

Unlike other domain coloring libraries, Complexplorer offers:

- **🎨 Riemann Relief Maps**: First library to offer modulus-scaled Riemann sphere visualizations of complex functions (Riemann relief maps)
- **🖨️ Direct STL Export**: Transform any complex function into a 3D-printable mathematical ornament
- **🚀 PyVista Integration**: 15-30x faster 3D rendering with cinema-quality output
- **🔧 Advanced Domain Composition**: Create complex domains through set operations (union, intersection, difference)
- **📊 Flexible Modulus Mapping**: 10+ scaling modes to highlight different function features

## Quick Start

```python
import complexplorer as cp

# Define your complex function
f = lambda z: (z**2 - 1) / (z**2 + 1)

# Visualize as an interactive Riemann relief map
cp.riemann_pv(f, modulus_mode='arctan', resolution=800)

# Export as a 3D-printable mathematical ornament
from complexplorer.export.stl import OrnamentGenerator

ornament = OrnamentGenerator(f, resolution=200)
ornament.generate_and_save('my_mathematical_ornament.stl', size_mm=80)
```

## Installation

**Requirements**: Python 3.11 or higher

```bash
pip install complexplorer

# Optional: For interactive matplotlib plots
pip install "complexplorer[qt]"

# Optional: For high-performance 3D visualizations
pip install "complexplorer[pyvista]"

# Optional: Install everything
pip install "complexplorer[all]"
```

For detailed installation instructions, see the [Installation Guide](getting-started/installation.md).

## The Magic of Complex Numbers

> *We cannot directly see the minute details of a Dedekind cut, nor is it clear that arbitrarily great or
> arbitrarily tiny times or lengths actually exist in nature. One could say that
> the so-called 'real numbers' are as much a product of mathematicians'
> imaginations as are the complex numbers. Yet we shall find that complex
> numbers, as much as reals, and perhaps even more, find a unity with
> nature that is truly remarkable.*
>
> — Sir Roger Penrose, [Road to Reality](https://www.ams.org/notices/200606/rev-blank.pdf), Chapter 4

## Next Steps

- **[Quick Start Guide](getting-started/quickstart.md)** - Create your first visualization in 5 minutes
- **[User Guide](user-guide/domains.md)** - Learn about domains, colormaps, and plotting
- **[Examples Gallery](examples/gallery.md)** - Explore beautiful visualizations
- **[API Reference](api/core.md)** - Detailed documentation of all functions

## Contributing

Contributions are welcome! Please visit our [GitHub repository](https://github.com/kuvychko/complexplorer) to:

- Report bugs or suggest features
- Submit pull requests
- Share your visualizations
- Improve documentation

## License

- **Code**: [MIT License](https://github.com/kuvychko/complexplorer/blob/main/LICENSE) — free for personal, academic, or commercial use
- **Renders & STL outputs**: [CC BY-NC 4.0](https://github.com/kuvychko/complexplorer/blob/main/LICENSE.art) — free for non-commercial use

## Acknowledgments

This library was inspired by Elias Wegert's beautiful book ["Visual Complex Functions"](https://link.springer.com/book/10.1007/978-3-0348-0180-5) and benefited greatly from his feedback and suggestions.
