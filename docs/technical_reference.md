# Modulus Scaling Analysis for Riemann Sphere Visualization

## Current Scaling Methods

The complexplorer library currently provides 5 modulus scaling methods for mapping complex function moduli `|f(z)|` from `[0, ∞)` to a finite radius range `[r_min, r_max]`:

### 1. **Constant** ⭐⭐⭐
```python
r = radius  # Fixed radius, no modulus information
```
- **Use case**: Traditional Riemann sphere showing only phase
- **Visual quality**: Clean but loses magnitude information

### 2. **Linear** ⭐
```python
r = 1 + scale * |f(z)|
```
- **Problem**: Unbounded growth, poor for functions with poles
- **Visual quality**: Often produces extreme distortions

### 3. **Arctan** ⭐⭐⭐⭐
```python
normalized = (2/π) * arctan(|f(z)|)
r = r_min + (r_max - r_min) * normalized
```
- **Use case**: General purpose, smooth compression
- **Visual quality**: Good balance, but can be too uniform

### 4. **Logarithmic** ⭐⭐
```python
log_mod = log(|f(z)|) / log(base)
normalized = 1 / (1 + exp(-log_mod))
r = r_min + (r_max - r_min) * normalized
```
- **Use case**: Functions with exponential growth
- **Visual quality**: Can overemphasize small values

### 5. **Linear Clamp** ⭐⭐⭐
```python
clamped = min(|f(z)|, m_max)
normalized = clamped / m_max
r = r_min + (r_max - r_min) * normalized
```
- **Use case**: Focus on specific modulus range
- **Visual quality**: Good for bounded regions, harsh cutoff

### 6. **Power** (Exists but not exposed) ⭐⭐⭐
```python
normalized = (|f(z)| / max(|f|))^exponent
r = r_min + (r_max - r_min) * normalized
```
- **Use case**: Fine-tuning compression/expansion
- **Visual quality**: Flexible but requires max calculation

### 7. **Custom** (Referenced but not implemented) ⭐⭐⭐⭐⭐
- Allows user-defined scaling functions

## Problems with Current Methods

1. **Lack of artistic control**: Most methods are purely mathematical
2. **Poor handling of zeros/poles**: Extreme values dominate visualization
3. **No perceptual uniformity**: Linear changes in modulus don't map to perceptually uniform changes
4. **Limited dynamic range control**: Hard to visualize functions with both small and large values

## Proposed New Scaling Methods

### High Usefulness (⭐⭐⭐⭐⭐)

#### 1. **Sigmoid Family**
```python
def sigmoid_scaling(moduli, steepness=1.0, center=1.0, r_min=0.2, r_max=1.0):
    """Smooth S-curve scaling with adjustable transition."""
    normalized = 1 / (1 + np.exp(-steepness * (moduli - center)))
    return r_min + (r_max - r_min) * normalized
```
- **Benefits**: Smooth transition, adjustable center and steepness
- **Use case**: Functions with clear "interesting" modulus range

#### 2. **Adaptive Percentile**
```python
def adaptive_percentile_scaling(moduli, low_percentile=10, high_percentile=90, r_min=0.2, r_max=1.0):
    """Scale based on data percentiles, ignoring extreme outliers."""
    p_low = np.percentile(moduli[np.isfinite(moduli)], low_percentile)
    p_high = np.percentile(moduli[np.isfinite(moduli)], high_percentile)
    
    normalized = np.clip((moduli - p_low) / (p_high - p_low), 0, 1)
    return r_min + (r_max - r_min) * normalized
```
- **Benefits**: Automatically adapts to data range, ignores outliers
- **Use case**: General purpose, especially for unknown functions

#### 3. **Hybrid Linear-Logarithmic**
```python
def hybrid_scaling(moduli, transition=1.0, r_min=0.2, r_max=1.0):
    """Linear for small values, logarithmic for large values."""
    small_mask = moduli <= transition
    large_mask = ~small_mask
    
    normalized = np.zeros_like(moduli)
    # Linear part: [0, transition] -> [0, 0.5]
    normalized[small_mask] = 0.5 * moduli[small_mask] / transition
    # Log part: (transition, ∞) -> (0.5, 1]
    normalized[large_mask] = 0.5 + 0.5 * np.tanh(np.log(moduli[large_mask] / transition))
    
    return r_min + (r_max - r_min) * normalized
```
- **Benefits**: Preserves detail at both small and large scales
- **Use case**: Functions with interesting behavior near zeros and poles

### Medium-High Usefulness (⭐⭐⭐⭐)

#### 4. **Smooth Step**
```python
def smooth_step_scaling(moduli, steps=[0.1, 1, 10], r_values=[0.2, 0.5, 0.8, 1.0]):
    """Piecewise smooth scaling with multiple plateaus."""
    # Implementation using cubic Hermite interpolation between steps
    # Creates visually distinct "levels" while maintaining smoothness
```
- **Benefits**: Creates clear visual layers for different magnitude ranges
- **Use case**: Highlighting specific modulus values (e.g., |f|=1)

#### 5. **Gaussian Bump**
```python
def gaussian_bump_scaling(moduli, center=1.0, width=0.5, bump_height=0.3, r_min=0.2, r_max=1.0):
    """Emphasize values near a specific modulus."""
    base = r_min + (r_max - r_min) * np.tanh(0.5 * moduli)
    bump = bump_height * np.exp(-((moduli - center) / width) ** 2)
    return np.clip(base + bump, r_min, r_max)
```
- **Benefits**: Highlights interesting modulus values
- **Use case**: Emphasizing unit circle or other special values

#### 6. **Perceptual Uniform**
```python
def perceptual_uniform_scaling(moduli, r_min=0.2, r_max=1.0):
    """Based on Stevens' power law for magnitude perception."""
    # Psychophysical scaling: perceived = actual^0.3 for brightness
    normalized = np.power(moduli / (1 + moduli), 0.3)
    return r_min + (r_max - r_min) * normalized
```
- **Benefits**: Perceptually linear changes
- **Use case**: Scientific visualization where perception matters

### Medium Usefulness (⭐⭐⭐)

#### 7. **Reciprocal**
```python
def reciprocal_scaling(moduli, offset=1.0, r_min=0.2, r_max=1.0):
    """Inverse scaling: emphasizes small values."""
    normalized = 1 - 1 / (1 + moduli / offset)
    return r_min + (r_max - r_min) * normalized
```
- **Benefits**: Shows detail near zeros
- **Use case**: Functions with interesting behavior near zeros

#### 8. **Sawtooth Modulus**
```python
def sawtooth_scaling(moduli, period=1.0, r_min=0.2, r_max=1.0):
    """Periodic scaling creating rings."""
    normalized = (moduli % period) / period
    return r_min + (r_max - r_min) * normalized
```
- **Benefits**: Shows modulus contours clearly
- **Use case**: Educational visualization of modulus levels

#### 9. **Artistic Wave**
```python
def wave_scaling(moduli, frequency=2.0, amplitude=0.1, r_min=0.2, r_max=1.0):
    """Base scaling with sinusoidal perturbation."""
    base = np.tanh(0.5 * moduli)
    wave = amplitude * np.sin(frequency * np.pi * moduli)
    normalized = np.clip(base + wave, 0, 1)
    return r_min + (r_max - r_min) * normalized
```
- **Benefits**: Creates visually interesting patterns
- **Use case**: Artistic visualizations

### Lower Usefulness (⭐⭐)

#### 10. **Threshold Bands**
```python
def threshold_bands_scaling(moduli, thresholds=[0.1, 1, 10], r_min=0.2, r_max=1.0):
    """Discrete bands based on thresholds."""
    # Creates distinct rings but lacks smoothness
```

#### 11. **Exponential Decay**
```python
def exp_decay_scaling(moduli, decay_rate=1.0, r_min=0.2, r_max=1.0):
    """Exponential decay from maximum radius."""
    normalized = np.exp(-decay_rate * moduli)
    return r_min + (r_max - r_min) * normalized
```

## Implementation Recommendations

### 1. **Fix Missing Custom Implementation**
```python
@staticmethod
def custom(moduli: np.ndarray, scaling_func: Callable[[np.ndarray], np.ndarray],
          r_min: float = 0.2, r_max: float = 1.0) -> np.ndarray:
    """Apply user-defined scaling function."""
    normalized = scaling_func(moduli)
    # Ensure output is in [0, 1]
    normalized = np.clip(normalized, 0, 1)
    return r_min + (r_max - r_min) * normalized
```

### 2. **Add Most Useful Methods**
Priority order for implementation:
1. Custom (fix missing implementation)
2. Sigmoid family
3. Adaptive percentile
4. Hybrid linear-logarithmic
5. Smooth step
6. Perceptual uniform

### 3. **Improve Existing Methods**
- Add `power` method to the exposed interface
- Make `linear` method bounded by default
- Add better parameter defaults based on common use cases

### 4. **Create Scaling Presets**
```python
SCALING_PRESETS = {
    'balanced': {'method': 'sigmoid', 'steepness': 2.0, 'center': 1.0},
    'detail_near_zero': {'method': 'hybrid', 'transition': 0.5},
    'highlight_unit': {'method': 'gaussian_bump', 'center': 1.0, 'width': 0.3},
    'artistic': {'method': 'wave', 'frequency': 3.0, 'amplitude': 0.1},
    'scientific': {'method': 'perceptual_uniform'},
}
```

## Visual Quality Comparison

### Best for General Use:
1. **Adaptive Percentile**: Automatically handles any function well
2. **Sigmoid**: Smooth and adjustable for known ranges
3. **Hybrid**: Good balance between small and large value detail

### Best for Specific Cases:
- **Zeros**: Reciprocal, Hybrid (linear part)
- **Poles**: Arctan, Sigmoid with high center
- **Unit Circle**: Gaussian bump centered at 1
- **Educational**: Sawtooth, Smooth step
- **Artistic**: Wave, Custom with creative functions

## Conclusion

The current scaling methods are functional but limited in their ability to create visually pleasing and informative visualizations. The proposed extensions focus on:

1. **Adaptability**: Methods that automatically adjust to data
2. **Perceptual quality**: Scaling that matches human perception
3. **Artistic control**: Options for creative visualization
4. **Special features**: Highlighting specific mathematical properties

Implementing the top 5-6 proposed methods would significantly enhance the visual quality and usefulness of Riemann sphere visualizations in complexplorer.

---

# Icosphere Technical Specification


## Overview

This document provides technical details for implementing icosahedral sphere meshing for Riemann sphere visualization in complexplorer.

## Mathematical Foundation

### Base Icosahedron

The icosahedron has 12 vertices, 20 faces, and 30 edges. The vertices in a unit icosahedron centered at origin:

```python
# Golden ratio
φ = (1 + √5) / 2

# Normalized vertices
vertices = [
    (±1, ±φ, 0),
    (0, ±1, ±φ),  
    (±φ, 0, ±1)
]
# All 12 combinations, then normalized to unit sphere
```

### Subdivision Algorithm

1. **Edge Midpoint Subdivision**:
   - For each triangular face (v1, v2, v3)
   - Create midpoints: m12, m23, m31
   - Form 4 new triangles: (v1,m12,m31), (v2,m23,m12), (v3,m31,m23), (m12,m23,m31)

2. **Sphere Projection**:
   - After subdivision, project all vertices to unit sphere
   - v_normalized = v / ||v||

3. **Recursive Refinement**:
   - Level 0: 12 vertices, 20 faces
   - Level 1: 42 vertices, 80 faces
   - Level n: V = 10×4^n + 2, F = 20×4^n

## Implementation Details

### Data Structure

```python
class IcosphereData:
    vertices: np.ndarray  # (N, 3) coordinates
    faces: np.ndarray     # (M, 3) vertex indices
    edges: set           # Edge cache for subdivision
    vertex_map: dict     # Maps edge midpoints to vertex indices
    
    # Additional attributes for Riemann sphere
    complex_coords: np.ndarray  # Stereographic projection
    north_pole_index: int       # Special handling for infinity
    south_pole_index: int       # Origin mapping
```

### Stereographic Projection

For Riemann sphere visualization:

```python
def stereographic_projection(x, y, z, from_north=True):
    """Project from sphere to complex plane."""
    if from_north:
        # North pole (0,0,1) to infinity
        w = (x + 1j*y) / (1 - z)  # z < 1
    else:
        # South pole (0,0,-1) to infinity  
        w = (x + 1j*y) / (1 + z)  # z > -1
    return w

def inverse_stereographic(w, to_north=True):
    """Project from complex plane to sphere."""
    u, v = w.real, w.imag
    u2v2 = u**2 + v**2
    
    if to_north:
        x = 2*u / (1 + u2v2)
        y = 2*v / (1 + u2v2)
        z = (u2v2 - 1) / (1 + u2v2)
    else:
        x = 2*u / (1 + u2v2)
        y = 2*v / (1 + u2v2)
        z = (1 - u2v2) / (1 + u2v2)
    
    return x, y, z
```

### Modulus Scaling Integration

```python
def apply_scaling(vertices, complex_values, scaling_func):
    """
    Scale sphere vertices based on function modulus.
    
    Parameters:
    - vertices: (N, 3) sphere coordinates
    - complex_values: (N,) complex function values
    - scaling_func: Callable mapping modulus to radius
    
    Returns:
    - scaled_vertices: (N, 3) scaled coordinates
    """
    moduli = np.abs(complex_values)
    radii = scaling_func(moduli)
    
    # Handle infinity specially
    finite_mask = np.isfinite(complex_values)
    radii[~finite_mask] = scaling_func(np.inf)
    
    # Scale vertices
    scaled = vertices * radii[:, np.newaxis]
    return scaled
```

## Optimizations

### 1. Vertex Deduplication

Use spatial hashing to avoid duplicate vertices:

```python
def get_or_create_vertex(v, tolerance=1e-10):
    key = tuple(np.round(v / tolerance) * tolerance)
    if key not in vertex_cache:
        vertex_cache[key] = len(vertices)
        vertices.append(v)
    return vertex_cache[key]
```

### 2. Parallel Evaluation

Utilize NumPy vectorization and multiprocessing:

```python
def evaluate_function_on_sphere(func, vertices, n_workers=None):
    # Convert to complex coordinates
    complex_coords = stereographic_projection(vertices)
    
    # Parallel evaluation
    if n_workers:
        chunks = np.array_split(complex_coords, n_workers)
        with Pool(n_workers) as pool:
            results = pool.map(func, chunks)
        return np.concatenate(results)
    else:
        return func(complex_coords)
```

### 3. Adaptive Refinement

Refine mesh based on function behavior:

```python
def adaptive_subdivide(mesh, func_values, threshold=0.1):
    """
    Subdivide faces where function changes rapidly.
    
    High gradient areas get more refinement.
    """
    for face in mesh.faces:
        v1, v2, v3 = face
        f1, f2, f3 = func_values[v1], func_values[v2], func_values[v3]
        
        # Check phase and magnitude variation
        phase_var = np.std(np.angle([f1, f2, f3]))
        mag_var = np.std(np.log1p(np.abs([f1, f2, f3])))
        
        if phase_var > threshold or mag_var > threshold:
            subdivide_face(face)
```

## Special Considerations

### 1. Pole Treatment

- North pole represents complex infinity
- Requires special handling for functions with poles
- Consider small excluded neighborhood around poles

### 2. Branch Cuts

- Detect and visualize branch cuts
- Add extra vertices along cuts for proper coloring
- Optional cut highlighting

### 3. Performance Targets

- 10k vertices: Interactive (60 FPS)
- 100k vertices: Smooth (30 FPS)
- 1M vertices: High quality (10 FPS)

## Testing Strategy

1. **Geometric Tests**:
   - Verify uniform distribution (Voronoi cell areas)
   - Check face orientation consistency
   - Validate stereographic projection

2. **Function Tests**:
   - Polynomials: z^n
   - Rational: (z-a)/(z-b)
   - Essential singularities: e^(1/z)
   - Branch points: sqrt(z), log(z)

3. **Performance Benchmarks**:
   - Mesh generation time vs subdivision level
   - Function evaluation overhead
   - Rendering frame rates

## References

1. [Spherical Subdivision Methods](https://www.cs.cmu.edu/~kmcrane/Projects/Other/SphericalSubdivision.pdf)
2. [Efficient Icosphere Generation](https://schneide.blog/2016/07/15/generating-an-icosphere-in-c/)
3. [Stereographic Projection in Complex Analysis](https://en.wikipedia.org/wiki/Riemann_sphere)

---

# PyVista Implementation Plan


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

---

# STL Ornament Generation Plan


## Overview

Create a module for generating 3D-printable Christmas ornaments from complex function visualizations on the Riemann sphere. The module will leverage existing PyVista capabilities while addressing the unique challenges of creating watertight, printable meshes.

## Architecture

### Module Structure
```
complexplorer/
├── stl_export/
│   ├── __init__.py
│   ├── ornament_generator.py    # Main ornament generation class
│   ├── mesh_healing.py          # Mesh repair and healing algorithms
│   ├── mesh_cutting.py          # Sphere bisection and capping
│   ├── decorations.py           # Hanging hooks and embellishments
│   └── presets.py               # Common ornament configurations
```

## Workflow

### 1. Interactive Visualization and Selection
```python
# User workflow example
import complexplorer as cp

# Step 1: Visualize function interactively
func = lambda z: (z**2 - 1) / (z**2 + 1)
cp.riemann_pv(func, scaling='arctan', cmap=cp.Phase(12))

# Step 2: Once satisfied, create ornament generator
ornament = cp.OrnamentGenerator(
    func=func,
    scaling='arctan',
    cmap=cp.Phase(12),
    size_mm=80  # Target diameter in millimeters
)
```

### 2. Mesh Generation and Healing

#### Key Challenges and Solutions:

**Branch Cut Tears**
- Complex functions often have branch cuts that create discontinuities
- Solution: Detect and bridge discontinuities before cutting
- Use vertex proximity analysis to identify tears
- Create bridge faces to seal gaps

**Modulus Scaling Artifacts**
- Different scaling methods (arctan, log) can create extreme deformations
- Solution: Implement radius limits and smoothing
- Add optional Laplacian smoothing post-processing

**Mesh Healing Strategy**
```python
class MeshHealer:
    def heal_mesh(self, mesh: pv.PolyData) -> pv.PolyData:
        """Comprehensive mesh healing pipeline."""
        # 1. Remove duplicate vertices
        mesh = mesh.clean(tolerance=1e-6)
        
        # 2. Detect and fix non-manifold edges
        mesh = self._fix_non_manifold_edges(mesh)
        
        # 3. Fill small holes (from numerical issues)
        mesh = mesh.fill_holes(hole_size=10)
        
        # 4. Detect and bridge branch cut tears
        mesh = self._bridge_branch_cuts(mesh)
        
        # 5. Ensure consistent normals
        mesh = mesh.compute_normals(consistent_normals=True)
        
        # 6. Optional smoothing
        if self.smooth:
            mesh = mesh.smooth(n_iter=50, relaxation_factor=0.1)
        
        return mesh
```

### 3. Cutting Plane Options

**Primary Cutting Modes:**

1. **Real Axis Cut (X-Z plane)**
   - Natural for functions with real symmetry
   - Cut plane: y = 0
   - Preserves Re(z) axis features

2. **Imaginary Axis Cut (Y-Z plane)**  
   - Good for functions with imaginary symmetry
   - Cut plane: x = 0
   - Preserves Im(z) axis features

3. **Custom Angle Cut**
   - Rotate cutting plane by specified angle
   - Useful for highlighting specific features

4. **Optimal Cut Detection**
   - Analyze function symmetries
   - Choose cut that minimizes visible seams

**Cutting Implementation:**
```python
class SphereCutter:
    def cut_sphere(self, mesh: pv.PolyData, cut_mode: str = 'real') -> Tuple[pv.PolyData, pv.PolyData]:
        """Cut sphere into two halves."""
        if cut_mode == 'real':
            normal = [0, 1, 0]  # Cut along y=0
        elif cut_mode == 'imaginary':
            normal = [1, 0, 0]  # Cut along x=0
        elif cut_mode.startswith('angle:'):
            angle = float(cut_mode.split(':')[1])
            normal = [np.cos(angle), np.sin(angle), 0]
        
        # Create cutting plane through origin
        plane = pv.Plane(center=[0, 0, 0], direction=normal, i_size=3, j_size=3)
        
        # Cut mesh
        clipped_top = mesh.clip(normal=normal)
        clipped_bottom = mesh.clip(normal=[-n for n in normal])
        
        # Cap the open edges
        top_capped = self._cap_mesh(clipped_top, normal)
        bottom_capped = self._cap_mesh(clipped_bottom, normal)
        
        return top_capped, bottom_capped
```

### 4. Edge Capping for Watertight Meshes

**Capping Strategy:**
1. Extract boundary edges from cut
2. Create planar triangulation of boundary
3. Option for decorative rim/lip for strength
4. Ensure proper normal orientation

```python
def _cap_mesh(self, mesh: pv.PolyData, cut_normal: List[float]) -> pv.PolyData:
    """Cap open edges to create watertight mesh."""
    # Extract boundary edges
    edges = mesh.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False
    )
    
    # Create cap using Delaunay triangulation
    if edges.n_points > 0:
        # Project boundary to cutting plane
        projected = self._project_to_plane(edges.points, cut_normal)
        
        # Triangulate
        cap = pv.PolyData(projected).delaunay_2d()
        
        # Merge with original mesh
        mesh = mesh + cap
        mesh = mesh.clean()
    
    return mesh
```

### 5. Hanging Hook Design

**Hook Options:**

1. **Simple Loop**
   ```python
   def add_simple_loop(mesh, position='top', size_mm=5):
       """Add a printable loop at the pole."""
       # Create torus
       loop = pv.Torus(
           r=size_mm/2,  # Major radius
           c=size_mm/8   # Minor radius (wire thickness)
       )
       # Position at north pole
       # Merge with ornament
   ```

2. **Integrated Tab**
   - Flat tab with hole
   - Better for resin printing
   - Stronger attachment

3. **Decorative Cap**
   - Ornamental top piece
   - Hides pole singularities
   - Multiple style options

### 6. Dimensional Scaling and Export

**Size Considerations:**
```python
class OrnamentScaler:
    def __init__(self, target_diameter_mm: float = 80):
        self.target_diameter = target_diameter_mm
        self.min_wall_thickness = 2.0  # mm
        self.min_feature_size = 0.5   # mm
        
    def scale_for_printing(self, mesh: pv.PolyData) -> pv.PolyData:
        """Scale mesh to target physical dimensions."""
        # Get current bounds
        bounds = mesh.bounds
        current_diameter = max(
            bounds[1] - bounds[0],  # X extent
            bounds[3] - bounds[2],  # Y extent
            bounds[5] - bounds[4]   # Z extent
        )
        
        # Calculate scale factor
        scale = self.target_diameter / current_diameter
        
        # Apply scaling
        mesh.points *= scale
        
        # Verify minimum feature sizes
        self._check_printability(mesh)
        
        return mesh
```

**Export Options:**
1. **Solid ornament** (Phase 1) - Simple, robust, heavier
2. **Hollow ornament** (Phase 3) - Advanced feature requiring:
   - Inset surface generation with specified wall thickness
   - Proper handling of complex topology
   - Drainage holes for resin printing
   - Verification of minimum wall thickness throughout

## Advanced Features

### 1. Function Embossing
- Add function formula as raised/recessed text
- Include function name or mathematical expression
- Position on flat cap area or integrate into rim
- Adjustable text depth and font size
- Example: "f(z) = (z²-1)/(z²+1)" or custom name like "Möbius 2024"

```python
def add_text_embossing(mesh, text, position='cap', depth_mm=1.0, font_size_mm=5.0):
    """Add embossed text to ornament."""
    # Create text as 3D mesh
    text_mesh = create_3d_text(text, font_size_mm, depth_mm)
    
    # Position on cap or rim
    if position == 'cap':
        # Center on the flat cap area
        text_mesh.translate(cap_center)
    elif position == 'rim':
        # Curve along the rim edge
        text_mesh = curve_text_on_cylinder(text_mesh, rim_radius)
    
    # Boolean union or difference for raised/recessed
    return mesh.boolean_union(text_mesh)
```

### 2. Preset Gallery
```python
# Common ornament presets with suggested names for embossing
ORNAMENT_PRESETS = {
    'classic_mobius': {
        'func': lambda z: (z - 1) / (z + 1),
        'name': 'Möbius Transform',
        'formula': 'f(z) = (z-1)/(z+1)',
        'scaling': 'constant',
        'cmap': Phase(12),
        'size': 75
    },
    'festive_polynomial': {
        'func': lambda z: z**4 - 1,
        'name': 'Quartic Star',
        'formula': 'f(z) = z⁴ - 1',
        'scaling': 'arctan',
        'cmap': Phase(n_phi=8, auto_scale_r=True),
        'size': 80
    },
    'snowflake': {
        'func': lambda z: (z**6 - 1) / (z**6 + 1),
        'name': 'Hexagonal Snowflake',
        'formula': 'f(z) = (z⁶-1)/(z⁶+1)',
        'scaling': 'arctan',
        'scaling_params': {'r_min': 0.3, 'r_max': 1.0},
        'cmap': Phase(6),
        'size': 90
    },
    'julia_festive': {
        'func': lambda z: z**2 + 0.3 + 0.5j,
        'name': 'Julia Ornament',
        'formula': 'f(z) = z² + (0.3+0.5i)',
        'scaling': 'logarithmic',
        'cmap': Phase(n_phi=12, auto_scale_r=True),
        'size': 85
    }
}
```

### 3. Quality Validation
```python
class PrintabilityChecker:
    def validate(self, mesh: pv.PolyData) -> Dict[str, bool]:
        """Check mesh for 3D printing issues."""
        return {
            'is_watertight': mesh.is_manifold,
            'has_consistent_normals': self._check_normals(mesh),
            'min_thickness_ok': self._check_thickness(mesh),
            'no_self_intersections': not mesh.collision(mesh),
            'printable_overhangs': self._check_overhangs(mesh)
        }
```

## Implementation Priority

### Phase 1: Core Functionality (Solid Ornaments)
1. Basic mesh healing for rectangular meshes
2. Simple cutting along real/imaginary axes  
3. Basic edge capping (flat caps)
4. STL export with proper scaling
5. Basic watertight validation

### Phase 2: Robustness & Features
1. Advanced healing for branch cuts
2. Custom cutting angles
3. Improved capping with decorative rim options
4. Simple loop hanging hook
5. Function text embossing
6. Comprehensive printability validation

### Phase 3: Advanced Features
1. Hollow ornament generation with wall thickness
2. Drainage holes for hollow ornaments
3. Preset gallery with popular functions
4. Advanced decorative cap options
5. Automatic optimal cut plane detection

## Example Usage

```python
import complexplorer as cp

# Interactive design phase
func = lambda z: (z**3 - 1) / (z**2 + 1)
cp.riemann_pv(func, scaling='arctan', cmap=cp.Phase(12, auto_scale_r=True))

# Production phase
ornament = cp.stl.OrnamentGenerator(
    func=func,
    resolution=150,
    scaling='arctan',
    scaling_params={'r_min': 0.2, 'r_max': 1.0},
    cmap=cp.Phase(12, auto_scale_r=True)
)

# Generate solid ornament (Phase 1)
ornament.heal_mesh()
top, bottom = ornament.cut(mode='real', add_rim=True)
top = ornament.add_hook(top, style='loop')

# Add optional embossing (Phase 2)
top = ornament.add_text(top, "f(z) = (z³-1)/(z²+1)", position='cap')

# Validate
if ornament.validate_printability(top):
    ornament.export('christmas_function_top.stl', top, size_mm=80)
    ornament.export('christmas_function_bottom.stl', bottom, size_mm=80)

# Future: Hollow generation (Phase 3)
# hollow_config = {'wall_thickness_mm': 2.5, 'add_drainage': True}
# top_hollow = ornament.make_hollow(top, **hollow_config)
```

## Technical Considerations

### Mesh Healing Order
The optimal order for mesh operations:
1. **Generate** base Riemann sphere mesh
2. **Heal** the complete sphere (fix tears, holes)
3. **Smooth** if needed (optional Laplacian smoothing)
4. **Cut** into halves
5. **Cap** the cut edges
6. **Add** decorations (hooks, text)
7. **Validate** final mesh
8. **Scale** to target dimensions
9. **Export** to STL

### Key Insight: Heal Before Cutting
Healing the complete sphere before cutting is crucial because:
- Branch cuts are easier to detect on the full sphere
- Symmetry helps identify matching tear edges
- Complete topology provides better context for repairs
- Capping is simpler on healed meshes

## Testing Strategy

1. **Test Functions**
   - Simple: `z`, `z^2`, `1/z`
   - Branch cuts: `sqrt(z)`, `log(z)`
   - Essential singularities: `exp(1/z)`, `sin(1/z)`
   - Meromorphic: Various rational functions

2. **Validation Suite**
   - Watertight check (Phase 1)
   - Self-intersection detection (Phase 1)
   - Overhang detection (Phase 2)
   - Support requirement analysis (Phase 2)
   - Wall thickness analysis (Phase 3 - for hollow ornaments)

3. **Physical Testing**
   - Print solid test ornaments in PLA/PETG (Phase 1)
   - Test hang strength and hook durability (Phase 2)
   - Verify text embossing legibility (Phase 2)
   - Test hollow ornaments with various wall thicknesses (Phase 3)