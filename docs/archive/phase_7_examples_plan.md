# Phase 7: Examples and Tutorials Consolidation Plan

## Current State Analysis

### Existing Examples Overview

**Notebooks (7 total):**
1. `domains_and_colormaps_tutorial.ipynb` - Comprehensive domain/colormap guide
2. `plotting_tutorial.ipynb` - Complete plotting capabilities guide
3. `pyvista_basics_tutorial.ipynb` - PyVista fundamentals (not complexplorer-specific)
4. `pyvista_complex_demo.ipynb` - PyVista integration showcase
5. `pyvista_jupyter_popup_fixed.ipynb` - Jupyter quality workaround
6. `riemann_sphere_demo.ipynb` - Riemann sphere with PyVista
7. `stl_ornament_demo.ipynb` - 3D printing workflow

**Scripts (5+ total):**
1. `interactive_demo.py` - Menu-driven PyVista showcase
2. `autoscale_demo.py` - Auto-scaled enhanced phase portraits
3. `stl_demo.py` - STL generation examples
4. `inspect_stl.py` - STL file inspection utility
5. `generate_gallery_images.py` - Gallery image generation
6. `generate_additional_examples.py` - Additional examples

### Key Issues Identified

1. **API Inconsistency**: Most use current API but some imports are outdated
2. **Too Many Files**: 7 notebooks + 5 scripts is overwhelming for users
3. **Redundancy**: PyVista quality warnings repeated across multiple notebooks
4. **Quality Issues**: Jupyter+PyVista has severe aliasing (mentioned 4+ times)
5. **No Clear Path**: Users don't know where to start or what order to follow

## Proposed Consolidated Structure

### Target: 3 Notebooks + 2 Scripts

#### Notebooks (with PyVista high-quality pop-ups):

1. **`getting_started.ipynb`** - First steps with complexplorer
   - Installation and setup
   - Basic 2D visualizations
   - Domains and colormaps introduction
   - Simple 3D visualization with PyVista (using `notebook=False`)
   - Clear instructions on quality differences

2. **`advanced_features.ipynb`** - Deep dive into capabilities
   - Enhanced phase portraits and auto-scaling
   - High-quality 3D with PyVista (`notebook=False` for pop-ups)
   - Riemann sphere projections
   - Custom domains and compositions
   - STL export workflow
   - Performance comparisons (matplotlib vs PyVista)

3. **`api_cookbook.ipynb`** - Practical recipes and patterns
   - Common function visualizations
   - Domain restriction techniques
   - Colormap selection guide
   - PyVista parameter tuning
   - Quality vs performance trade-offs

#### Scripts (CLI for batch processing):

1. **`interactive_showcase.py`** - High-quality interactive demo
   - Consolidate current `interactive_demo.py`
   - Add STL export options
   - Include all visualization types
   - Menu-driven exploration
   - Batch processing capabilities

2. **`generate_gallery.py`** - Gallery image generation
   - Update to use new API consistently
   - Generate all gallery images
   - Include code snippets in output
   - High-resolution outputs

#### Archive/Remove:
- `pyvista_basics_tutorial.ipynb` (not complexplorer-specific)
- `pyvista_jupyter_popup_fixed.ipynb` (document workaround elsewhere)
- `pyvista_complex_demo.ipynb` (merge into interactive_showcase.py)
- `riemann_sphere_demo.ipynb` (merge into interactive_showcase.py)
- `autoscale_demo.py` (merge into notebooks)
- `inspect_stl.py` (utility, move to tools/ if needed)

## Execution Checklist

### Phase 7.1: Prepare New Structure
- [ ] Create `examples/archive/` directory for old examples
- [ ] Create new notebook templates with consistent structure
- [ ] Set up proper imports for new API

### Phase 7.2: Create `getting_started.ipynb`
- [ ] Write introduction section with installation
- [ ] Add "Hello World" example: simple polynomial
- [ ] Introduce domains: Rectangle, Disk, Annulus
- [ ] Basic colormaps: Phase(), Phase(n_phi=6)
- [ ] First 2D visualizations with plot()
- [ ] Side-by-side with pair_plot()
- [ ] Introduction to 3D with PyVista:
  - [ ] Explain notebook=False for quality
  - [ ] Simple plot_landscape_pv() example
  - [ ] Compare inline vs pop-up quality
- [ ] Add exercises and experimentation prompts

### Phase 7.3: Create `advanced_features.ipynb`
- [ ] Enhanced phase portraits with auto_scale_r
- [ ] All colormap types with examples
- [ ] High-performance 3D with PyVista:
  - [ ] plot_landscape_pv() with parameters
  - [ ] pair_plot_landscape_pv()
  - [ ] riemann_pv() with modulus scaling
  - [ ] Performance comparison: matplotlib vs PyVista
- [ ] Domain composition (union, intersection)
- [ ] Custom domain creation
- [ ] STL export workflow:
  - [ ] OrnamentGenerator usage
  - [ ] Mesh validation
  - [ ] 3D printing tips
- [ ] Publication figure examples

### Phase 7.4: Create `api_cookbook.ipynb`
- [ ] Polynomial gallery (quadratic, cubic, quartic)
- [ ] Rational functions with poles
- [ ] Transcendental functions (exp, sin, log)
- [ ] Essential singularities
- [ ] Domain restriction patterns
- [ ] Colormap selection guide
- [ ] PyVista quality tips:
  - [ ] Always use notebook=False in Jupyter
  - [ ] Resolution guidelines
  - [ ] Camera positioning
- [ ] Performance optimization:
  - [ ] When to use matplotlib vs PyVista
  - [ ] Resolution vs quality trade-offs

### Phase 7.5: Update `interactive_showcase.py`
- [ ] Consolidate menu system
- [ ] Add all function presets
- [ ] Include all visualization types:
  - [ ] 2D phase portraits
  - [ ] 3D landscapes (PyVista)
  - [ ] Riemann sphere (PyVista)
  - [ ] STL export option
- [ ] Add colormap selection
- [ ] Include modulus scaling options
- [ ] Add save/export capabilities

### Phase 7.6: Update `generate_gallery.py`
- [ ] Update all imports to new API
- [ ] Fix domain constructor calls
- [ ] Ensure consistent style
- [ ] Generate updated code snippets
- [ ] Create gallery index file

### Phase 7.7: Migration and Cleanup
- [ ] Move old notebooks to archive/
- [ ] Update references in documentation
- [ ] Test all new examples
- [ ] Verify gallery images generate correctly
- [ ] Update README.md links

### Phase 7.8: Quality Assurance
- [ ] Run all notebooks top-to-bottom
- [ ] Test interactive_showcase.py on multiple platforms
- [ ] Verify STL exports are valid
- [ ] Check all code examples use new API
- [ ] Ensure no legacy imports remain

## Success Criteria

1. **Clarity**: New users can start with `getting_started.ipynb` and progress naturally
2. **Completeness**: All major features are demonstrated
3. **Quality**: PyVista examples use `notebook=False` for high-quality pop-ups
4. **Consistency**: All examples use new API imports and syntax
5. **Practicality**: Cookbook provides ready-to-use patterns

## Notes

- Prioritize clarity over comprehensiveness
- Each notebook should be self-contained
- Always demonstrate PyVista with `notebook=False` for quality
- Include clear explanations about quality differences (inline vs pop-up)
- Show performance benefits: PyVista is 15-30x faster than matplotlib
- Keep total execution time reasonable (< 5 min per notebook)

## Key Quality Pattern for Notebooks

```python
# For high-quality 3D visualization in Jupyter
plotter = cp.plot_landscape_pv(domain, func, notebook=False, show=True)

# Explain to users:
# - notebook=False creates external window with full quality
# - Inline display (notebook=True) has severe aliasing
# - External windows are interactive and high-resolution
```