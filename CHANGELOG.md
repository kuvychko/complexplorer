# Changelog

All notable changes to complexplorer will be documented in this file.

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