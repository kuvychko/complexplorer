# PyVista Implementation Plan for Complexplorer

## Overview

This plan outlines the implementation of PyVista-based 3D visualizations for complexplorer, prioritizing interactive visualizations while maintaining support for static figure generation and export.

## Core Principles

1. **Interactive First**: All visualizations default to interactive mode using the trame backend
2. **Static Support**: Every function supports static figure generation via `interactive=False` parameter
3. **Export Flexibility**: Support for multiple formats (PNG, PDF, STL, VTK, etc.)
4. **Performance**: Leverage PyVista's GPU acceleration for 15-30x speedup over matplotlib
5. **Accurate Coloring**: Per-vertex coloring eliminates interpolation artifacts

## Phase 1: Core 3D Plot Functions

### 1.1 Create `plots_3d_pyvista.py`

Implement PyVista analogs for all functions in `plots_3d.py`:

```python
# Core functions to implement:
- plot_landscape_pv(domain, func, cmap, ...)
- pair_plot_landscape_pv(domain, func, cmap, ...)
- landscape visualization with accurate domain coloring
```

### 1.2 Key Features for Each Function

- **Interactive Mode** (default):
  - Full 3D navigation with mouse
  - Keyboard shortcuts (e.g., 'e' for edges, 'w' for wireframe)
  - Real-time parameter adjustment widgets
  
- **Static Mode**:
  - High-quality rendering with customizable camera positions
  - Support for publication-quality figures
  - Batch processing capability

### 1.3 Common Parameters

```python
def plot_landscape_pv(
    domain, 
    func,
    cmap=None,
    n=100,
    interactive=True,        # Toggle interactive/static
    camera_position='iso',   # Predefined or custom views
    show_edges=False,       # Edge display control
    edge_color='gray',
    z_scale=1.0,           # Height scaling
    log_z=False,           # Logarithmic height option
    filename=None,         # Save to file
    window_size=(800, 600),
    return_plotter=False,  # For advanced customization
    **kwargs
):
```

## Phase 2: Icosahedral Meshing

### 2.1 Implement Icosahedral Mesh Generation

Create `mesh_utils.py` with icosahedral sphere generation:

```python
class IcosphereGenerator:
    def __init__(self, radius=1.0, subdivisions=4):
        """
        Generate icosahedral sphere mesh.
        
        Parameters:
        - radius: Sphere radius
        - subdivisions: Refinement level (0-6 recommended)
        """
    
    def generate(self):
        """Returns PyVista PolyData mesh"""
        
    def get_vertices(self):
        """Returns vertex coordinates"""
        
    def get_faces(self):
        """Returns triangular faces"""
```

### 2.2 Key Advantages

- **Uniform point distribution**: Unlike rectangular grids, no pole singularities
- **Efficient**: Triangular faces ideal for GPU rendering
- **Scalable**: Subdivision levels control resolution
- **STL-ready**: Direct export for 3D printing

### 2.3 Implementation Details

1. Start with 12-vertex icosahedron
2. Recursive subdivision of triangular faces
3. Project vertices to unit sphere
4. Maintain vertex indexing for efficient updates

## Phase 3: Riemann Sphere Visualization

### 3.1 Core Implementation

```python
def riemann_pv(
    func,
    n_subdivisions=4,      # Icosphere refinement
    scaling='arctan',      # Modulus scaling method
    scaling_params=None,   # Custom scaling parameters
    project_from_north=True,
    interactive=True,
    show_grid=True,        # Latitude/longitude lines
    show_axes=True,
    colorbar=True,
    **kwargs
):
```

### 3.2 Scaling Functions

Implement flexible modulus-to-radius mapping:

```python
class ModulusScaling:
    @staticmethod
    def arctan(modulus, r_min=0.2, r_max=1.0):
        """Default: maps [0, ∞) to [r_min, r_max]"""
        return r_min + (r_max - r_min) * (2/π * arctan(modulus))
    
    @staticmethod
    def logarithmic(modulus, base=np.e, r_min=0.2, r_max=1.0):
        """Logarithmic scaling for large dynamic range"""
        
    @staticmethod
    def linear_clamp(modulus, m_max=10, r_min=0.2, r_max=1.0):
        """Linear with saturation"""
        
    @staticmethod
    def custom(modulus, scaling_func):
        """User-defined scaling function"""
```

### 3.3 Enhanced Features

1. **Infinity Handling**: Special treatment for poles and essential singularities
2. **Grid Overlays**: Optional latitude/longitude lines
3. **Cross-sections**: Interactive slicing planes
4. **Animation**: Parameter variation over time

## Phase 4: Additional Enhancements

### 4.1 Advanced Visualization Features

1. **Contour Integration**:
   - Phase contours (constant argument)
   - Modulus contours (constant magnitude)
   - Critical point highlighting

2. **Multi-Function Comparison**:
   ```python
   def compare_functions_pv(funcs, titles, layout=(2, 2), **kwargs):
       """Side-by-side comparison with linked cameras"""
   ```

3. **Domain Coloring Options**:
   - Enhanced phase portraits with magnitude encoding
   - Custom color maps for special functions
   - Perceptually uniform color spaces

### 4.2 Performance Optimizations

1. **Mesh Caching**: Reuse domain meshes for multiple functions
2. **GPU Arrays**: Direct GPU memory manipulation for animations
3. **Level-of-Detail**: Automatic resolution adjustment based on zoom
4. **Parallel Evaluation**: Multi-threaded function evaluation

### 4.3 Export Enhancements

1. **3D Printing Support**:
   - STL export with proper scaling
   - Thickness addition for printability
   - Support material indicators

2. **Publication Features**:
   - Vector graphics (PDF, SVG)
   - High-DPI rendering
   - LaTeX-compatible labels

3. **Web Integration**:
   - glTF export for web viewers
   - Jupyter widget embedding
   - Static HTML reports

## Phase 5: Integration and Documentation

### 5.1 API Design

Maintain consistency with existing complexplorer API while adding PyVista-specific features:

```python
# Automatic backend selection
complexplorer.use_backend('pyvista')  # Set globally
plot_landscape(..., backend='pyvista')  # Per-function

# Backward compatibility
plot_landscape(..., use_pyvista=True)  # Convenience flag
```

### 5.2 Documentation

1. **Tutorials**:
   - "Getting Started with PyVista in Complexplorer"
   - "Interactive 3D Exploration of Complex Functions"
   - "Creating Publication-Quality 3D Figures"

2. **Examples Gallery**:
   - Essential singularities
   - Branch cuts visualization
   - Möbius transformations
   - Conformal mappings

3. **Performance Guide**:
   - Optimal mesh resolutions
   - GPU utilization tips
   - Memory management

## Implementation Timeline

### Week 1-2: Core Infrastructure
- Set up `plots_3d_pyvista.py`
- Implement basic `plot_landscape_pv`
- Create test suite

### Week 3-4: Icosahedral Meshing
- Implement `IcosphereGenerator`
- Optimize subdivision algorithm
- Validate uniform distribution

### Week 5-6: Riemann Sphere
- Implement `riemann_pv` with icosahedral mesh
- Add scaling functions
- Test with pathological cases

### Week 7-8: Enhancements
- Add advanced features
- Optimize performance
- Create comprehensive examples

### Week 9-10: Integration
- API unification
- Documentation
- User testing and feedback

## Success Metrics

1. **Performance**: 15-30x speedup over matplotlib
2. **Quality**: No color interpolation artifacts
3. **Usability**: Intuitive interactive controls
4. **Compatibility**: Seamless integration with existing code
5. **Export**: Support for all major 3D and image formats

## Future Possibilities

1. **VR/AR Support**: Immersive complex function exploration
2. **Cloud Rendering**: Server-side computation for complex animations
3. **Machine Learning**: GPU-accelerated optimization on complex surfaces
4. **Educational Tools**: Interactive textbook integration
5. **Research Applications**: Real-time parameter exploration for mathematical research

## Notes

- Prioritize user experience in interactive mode
- Ensure all functions have comprehensive docstrings
- Include performance benchmarks in tests
- Consider accessibility (colorblind-friendly options)
- Plan for future WebGPU support