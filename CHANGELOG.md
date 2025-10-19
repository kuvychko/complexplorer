# Changelog

All notable changes to complexplorer will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025-01-XX

### Breaking Changes

#### Parameter Renaming
- **BREAKING**: Renamed `n_phi` → `phase_sectors` across all colormaps and functions
  - Affects: `Phase`, `OklabPhase`, `PerceptualPastel`, `PolarChessboard`, and all related functions
  - Migration: Replace `n_phi=6` with `phase_sectors=6`
  - Reason: Clearer parameter naming consistent with mathematical terminology

- **DEPRECATED**: STL export parameter names unified with plotting functions
  - `OrnamentGenerator` and `create_ornament()`: `scaling` → `modulus_mode`, `scaling_params` → `modulus_params`
  - Old names still work with deprecation warnings
  - Migration: Replace `scaling='arctan'` with `modulus_mode='arctan'`, `scaling_params={}` with `modulus_params={}`
  - Reason: API consistency across all 3D visualization and export functions

### Added

#### New Colormaps (13 Total)
- **PerceptualPastel**: OkLCh-based pastels with uniform brightness, ideal for print
- **Isoluminant**: Constant lightness with phase encoded purely in hue
- **CubehelixPhase**: Grayscale-safe colormap for scientific publishing
- **AnalogousWedge**: Harmonious colors from compressed hue range (ocean/sunset themes)
- **DivergingWarmCool**: Warm/cool divergence for cartographic style
- **InkPaper**: Minimalist near-monochrome with subtle phase tints
- **EarthTopographic**: Terrain-inspired with earth tones and hillshade
- **FourQuadrant**: Bauhaus-inspired with 4 color anchors
- **OklabPhase**: Perceptually uniform OKLAB color space phase portraits

#### Enhanced Phase Portrait Features
- **Auto-scaling**: `auto_scale_r=True` automatically calculates spacing for square cells
- **Scale radius control**: `scale_radius` parameter to adjust cell size reference
- **Combined enhancements**: Use phase sectors + modulus contours simultaneously
- **Brightness control**: `v_base` parameter for contrast adjustment

#### Comprehensive Documentation (5,571 lines)
- **Getting Started Guide**: Installation and quickstart tutorials
- **User Guide**: Complete documentation for domains, colormaps, 2D/3D plotting, Riemann sphere
- **API Reference**: Full API documentation with mkdocstrings
- **Gallery**: Visual showcase with 50+ code examples
- **Development Guide**: Contributing guidelines and architecture documentation
- **MkDocs Material**: Professional documentation site with search and navigation

#### Modulus Scaling System
- 10+ modulus scaling modes for 3D/Riemann sphere: `arctan`, `logarithmic`, `adaptive`, `sigmoid`, `power`, `linear_clamp`, `hybrid`
- STL-specific parameter defaults for 3D printability
- Custom parameter support for all scaling modes
- **NEW**: Added modulus scaling to matplotlib's `riemann()` function for feature parity with PyVista backend
  - All scaling modes now available in both matplotlib and PyVista
  - Default remains `'constant'` (unit sphere) for backwards compatibility

#### Logging Framework
- Comprehensive logging with proper log levels (DEBUG, INFO, WARNING, ERROR)
- Module-specific loggers for debugging
- Performance timing for slow operations
- Configurable log output

#### Validation System
- Centralized input validation with clear error messages
- Custom exception hierarchy: `ValidationError`, `DomainError`, `ColormapError`, `PlottingError`
- Immediate validation (fail-fast approach)

#### Type Hints
- Complete type hints for all public APIs
- Return type annotations for utility functions
- NumPy array type hints

### Improved

#### Colormap System Refactoring
- **Eliminated 800+ lines of duplication** by consolidating shared enhancement logic
- Unified handling of phase sectors, modulus contours, and auto-scaling
- Consistent parameter validation across all colormaps
- Better code maintainability and extensibility

#### OkLCh Color Space
- Proper gamut clipping for out-of-gamut colors
- Improved color accuracy in perceptually uniform colormaps
- Better handling of extreme chroma values

#### Plot Validation
- Consolidated validation into shared `plotting.validation` module
- Consistent validation across matplotlib and PyVista backends
- Better error messages with actionable suggestions

#### Performance
- Vectorized color space conversions
- Optimized mesh generation for Riemann sphere
- Reduced memory footprint in 3D plotting

#### Documentation
- Added mathematical formulas in docstrings
- Comprehensive examples for all major features
- Cross-referencing between related functions
- Troubleshooting sections

### Fixed
- Riemann sphere mesh generation at poles (avoid singularities)
- Color space conversion edge cases
- PyVista notebook backend aliasing issues (documented workarounds)
- UTF-8 encoding in documentation
- Parameter validation consistency

### Changed
- Minimum Python version: 3.11+ (unchanged)
- Dependencies: NumPy 1.26.0+, Matplotlib 3.8.0+, SciPy 1.11.0+
- Optional dependencies: PyVista 0.45.0+, PyQt6 6.5.0+

### Deprecated
- Legacy parameter names (will be removed in v3.0)

### Removed
- Old colormap duplication code (consolidated)

### Internal Refactoring
- Modularized colormap enhancement logic
- Separated color space utilities into `core.color_utils`
- Created `core.scaling` module for modulus scaling
- Better separation of concerns across modules

### Migration Guide

#### Updating Parameter Names
```python
# Before (v1.x)
cmap = cp.Phase(n_phi=6, auto_scale_r=True)

# After (v2.0)
cmap = cp.Phase(phase_sectors=6, auto_scale_r=True)
```

#### Using New Colormaps
```python
# Elegant pastels for print
cmap = cp.PerceptualPastel(phase_sectors=6, auto_scale_r=True)

# Minimalist near-monochrome
cmap = cp.InkPaper()

# Terrain-inspired
cmap = cp.EarthTopographic()
```

#### Auto-Scaling Feature
```python
# Automatically calculate spacing for square cells
cmap = cp.Phase(phase_sectors=6, auto_scale_r=True, scale_radius=0.8)
```

### Contributors
- Core development and refactoring
- Documentation and examples
- Testing and validation

---

## [1.0.0] - 2025-07-27

### Major Refactoring
- Complete restructuring of the codebase with modular architecture
- Migrated from flat structure to organized submodules (`core`, `plotting`, `export`)
- Introduced abstract base classes for extensibility
- Standardized API with consistent parameter naming (`n` → `resolution`)

### New Features
- **Riemann Relief Maps**:
  - Riemann Relief Maps (or Mathematical Ornaments) with many different types of modulus scaling functions
  
- **PyVista Integration**: High-performance 3D visualizations with 15-30x speed improvement
  - `plot_landscape_pv()`, `pair_plot_landscape_pv()`, `riemann_pv()` functions
  - Interactive rotation, zooming, and navigation
  - HTML export capability for sharing visualizations
  
- **STL Export**: Generate 3D-printable mathematical ornaments
  - Export Riemann sphere and analytic landscape visualizations
  - Create mathematical art pieces and educational models
  
- **Enhanced Phase Portraits**: Auto-scaling for optimal visualization
  - Automatic square cell sizing with `auto_scale_r=True`
  - Improved modulus scaling options for all 3D plots
  
- **Domain Operations**: Set operations for complex domains
  - Union, intersection, and difference operations
  - Composite domains with automatic viewing window calculation

### Improvements
- Added PyQt6 backend support for interactive matplotlib plots
- Fixed Riemann sphere grid visualization and phase coloring symmetry
- Improved singularity handling in all plot types
- Added comprehensive unit test suite (341 tests with full coverage)
- Enhanced numerical stability and warning management

### API Changes
- Minimum Python version raised to 3.11
- Added optional dependencies: `[pyvista]`, `[qt]`, `[all]`
- Deprecated `sawtooth_legacy` function (removed)
- Consistent parameter naming across all functions

### Documentation
- New tutorial notebooks with clear examples
- Interactive CLI demo for optimal PyVista experience
- CLAUDE.md project guide for AI-assisted development

### Breaking Changes
- Removed backward compatibility with pre-0.1.2 versions
- Changed module structure (imports may need updating)
- Some function signatures updated for consistency

## [0.1.2] - Previous Release
- Initial public release with basic functionality