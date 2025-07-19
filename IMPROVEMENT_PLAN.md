# Complexplorer Library Improvement Plan

## Executive Summary

This document outlines a comprehensive improvement plan for the complexplorer library, focusing on three main priorities:
1. Establishing a robust unit testing framework
2. Implementing high-performance PyVista-based 3D visualizations with icosahedral meshing
3. Creating a Christmas ornament generator for 3D printing complex functions

The plan also includes additional enhancements to improve usability, performance, and mathematical capabilities.

## Phase 1: Unit Testing Infrastructure

### 1.1 Testing Framework Setup

**Objective**: Create comprehensive test coverage for all existing functionality to ensure mathematical correctness and enable safe refactoring.

**Structure**:
```
tests/
├── unit/
│   ├── test_domain.py          # Test Domain, Rectangle, Disk, Annulus
│   ├── test_cmap.py            # Test Phase, Chessboard, PolarChessboard, LogRings
│   ├── test_funcs.py           # Test phase(), sawtooth(), stereographic()
│   ├── test_plots_2d.py        # Test 2D plotting functions
│   └── test_plots_3d.py        # Test 3D plotting functions
├── fixtures/
│   ├── test_functions.py       # Common complex functions for testing
│   └── reference_images/       # Baseline images for visual regression tests
└── conftest.py                 # pytest configuration
```

### 1.2 Domain Testing (`test_domain.py`)

```python
class TestDomain:
    def test_base_domain_creation(self):
        """Test Domain base class initialization and attributes."""
    
    def test_domain_mesh_generation(self):
        """Test mesh generation with various n values."""
    
    def test_domain_masking(self):
        """Test inmask/outmask generation."""
    
    def test_domain_operations(self):
        """Test union and intersection operations."""

class TestRectangle:
    def test_rectangle_creation(self):
        """Test Rectangle domain with various parameters."""
    
    def test_rectangle_contains(self):
        """Test point containment for rectangular domains."""
    
    def test_rectangle_window(self):
        """Test viewing window calculation."""

class TestDisk:
    def test_disk_creation(self):
        """Test Disk domain creation."""
    
    def test_disk_boundary(self):
        """Test points on and near disk boundary."""

class TestAnnulus:
    def test_annulus_creation(self):
        """Test Annulus with various inner/outer radii."""
    
    def test_annulus_degenerate_cases(self):
        """Test edge cases like inner_radius = 0 or = outer_radius."""
```

### 1.3 Color Map Testing (`test_cmap.py`)

```python
class TestPhase:
    def test_phase_basic(self):
        """Test basic phase coloring without enhancement."""
    
    def test_phase_enhanced(self):
        """Test enhanced phase portraits."""
    
    def test_phase_at_special_points(self):
        """Test behavior at z=0, z=∞, branch points."""
    
    def test_hsv_rgb_conversion(self):
        """Verify HSV to RGB conversion accuracy."""

class TestChessboard:
    def test_chessboard_pattern(self):
        """Test chessboard grid generation."""
    
    def test_chessboard_period(self):
        """Test different period values."""

class TestPolarChessboard:
    def test_polar_pattern(self):
        """Test polar coordinate grid."""
    
    def test_logarithmic_spacing(self):
        """Test log-spaced radial lines."""

class TestLogRings:
    def test_ring_generation(self):
        """Test logarithmic ring patterns."""
```

### 1.4 Function Testing (`test_funcs.py`)

```python
class TestPhaseFunction:
    def test_phase_range(self):
        """Ensure phase ∈ [0, 2π)."""
    
    def test_phase_quadrants(self):
        """Test phase in all four quadrants."""
    
    def test_phase_branch_cut(self):
        """Test behavior along negative real axis."""

class TestStereographic:
    def test_stereographic_projection(self):
        """Test forward projection z → (x,y,z)."""
    
    def test_stereographic_inverse(self):
        """Test (x,y,z) → z inverse."""
    
    def test_north_vs_south_pole(self):
        """Test projection from different poles."""
    
    def test_infinity_handling(self):
        """Test behavior as |z| → ∞."""
```

### 1.5 Visual Regression Testing

```python
class TestVisualRegression:
    @pytest.mark.parametrize("plot_func,params", [
        (plot, {"domain": Rectangle(4, 4), "func": lambda z: z**2}),
        (riemann, {"func": lambda z: (z-1)/(z**2+z+1)}),
        # ... more test cases
    ])
    def test_plot_regression(self, plot_func, params, image_regression):
        """Compare generated plots against reference images."""
        fig = plot_func(**params)
        image_regression.check(fig)
```

## Phase 2: PyVista Implementation

### 2.1 Architecture Design

**Goal**: Create high-performance 3D visualizations while maintaining backward compatibility.

**Module Structure**:
```python
# complexplorer/pyvista_backend.py
try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    
class PyVistaBackend:
    """Base class for PyVista-based plotting."""
    
    @staticmethod
    def check_availability():
        if not PYVISTA_AVAILABLE:
            raise ImportError(
                "PyVista is required for this feature.\n"
                "Install with: pip install complexplorer[pyvista]"
            )
```

### 2.2 Analytic Landscape Implementation

```python
# complexplorer/plots_3d_pyvista.py

def plot_landscape_pv(domain, func, cmap=Phase(6), n=200,
                     height_scale=1.0, 
                     camera_position='iso',
                     lighting='three_point',
                     show_edges=False,
                     interactive=False,
                     backend='static',
                     window_size=(800, 600),
                     **kwargs):
    """
    High-performance 3D landscape plot using PyVista.
    
    Parameters
    ----------
    domain : Domain
        Complex domain to visualize
    func : callable
        Complex function f: C → C
    cmap : Cmap
        Color map for phase/modulus coloring
    n : int
        Mesh resolution (points along longest axis)
    height_scale : float
        Scale factor for z-axis (modulus)
    camera_position : str or tuple
        'iso', 'top', 'front' or (x,y,z) tuple
    lighting : str
        'three_point', 'none', or custom
    show_edges : bool
        Show mesh edges
    interactive : bool
        Enable interactive mode
    backend : str
        'static' for images, 'trame' for web-based interaction
    
    Returns
    -------
    plotter : pyvista.Plotter or image array
    """
    
    # Generate mesh
    z = domain.mesh(n)
    mask = domain.outmask(n)
    f = func(z)
    if mask is not None:
        f[mask] = np.nan
    
    # Create structured grid
    x = np.real(z)
    y = np.imag(z)
    z_height = np.abs(f) * height_scale
    
    grid = pv.StructuredGrid(x, y, z_height)
    
    # Apply colors
    rgb = cmap.rgb(f)
    colors = (rgb.reshape(-1, 3) * 255).astype(np.uint8)
    grid.point_data['colors'] = colors
    
    # Create plotter
    if interactive:
        plotter = pv.Plotter()
        plotter.add_mesh(grid, 
                        scalars='colors',
                        rgb=True,
                        show_edges=show_edges)
        
        # Set camera and lighting
        _set_camera_position(plotter, camera_position)
        _configure_lighting(plotter, lighting)
        
        if backend == 'trame':
            plotter.show(jupyter_backend='trame')
        else:
            plotter.show()
            
        return plotter
    else:
        # Static rendering
        plotter = pv.Plotter(off_screen=True, window_size=window_size)
        plotter.add_mesh(grid, 
                        scalars='colors',
                        rgb=True,
                        show_edges=show_edges)
        
        _set_camera_position(plotter, camera_position)
        _configure_lighting(plotter, lighting)
        
        plotter.show()
        return plotter.screenshot()
```

### 2.3 Icosahedral Mesh Implementation

```python
# complexplorer/icosahedral_mesh.py

class IcosahedralSphere:
    """
    Generate icosahedral mesh for efficient sphere tessellation.
    
    This avoids the pole clustering problem of rectangular meshes.
    """
    
    def __init__(self, subdivisions=4):
        self.subdivisions = subdivisions
        self.vertices = None
        self.faces = None
        self._generate_base_icosahedron()
        self._subdivide()
    
    def _generate_base_icosahedron(self):
        """Create the 12 vertices of a regular icosahedron."""
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        
        # Vertices in (±1, ±φ, 0) permutations
        verts = [
            [-1,  phi,  0], [ 1,  phi,  0], [-1, -phi,  0], [ 1, -phi,  0],
            [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
            [ phi,  0, -1], [ phi,  0,  1], [-phi,  0, -1], [-phi,  0,  1]
        ]
        
        # Normalize to unit sphere
        self.vertices = np.array(verts) / np.sqrt(1 + phi**2)
        
        # Define 20 triangular faces
        self.faces = np.array([
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ])
    
    def _subdivide(self):
        """Recursively subdivide triangular faces."""
        for _ in range(self.subdivisions):
            new_faces = []
            edge_cache = {}
            
            for face in self.faces:
                # Get vertices of triangle
                v0, v1, v2 = face
                
                # Get or create midpoint vertices
                mid01 = self._get_midpoint(v0, v1, edge_cache)
                mid12 = self._get_midpoint(v1, v2, edge_cache)
                mid20 = self._get_midpoint(v2, v0, edge_cache)
                
                # Create 4 new triangles
                new_faces.extend([
                    [v0, mid01, mid20],
                    [v1, mid12, mid01],
                    [v2, mid20, mid12],
                    [mid01, mid12, mid20]
                ])
            
            self.faces = np.array(new_faces)
    
    def _get_midpoint(self, v0, v1, cache):
        """Get or create normalized midpoint between two vertices."""
        key = tuple(sorted([v0, v1]))
        
        if key in cache:
            return cache[key]
        
        # Create new vertex at midpoint
        mid = (self.vertices[v0] + self.vertices[v1]) / 2
        mid = mid / np.linalg.norm(mid)  # Normalize to sphere
        
        idx = len(self.vertices)
        self.vertices = np.vstack([self.vertices, mid])
        cache[key] = idx
        
        return idx
    
    def to_complex_plane(self):
        """Map sphere points to complex plane via stereographic projection."""
        # Avoid division by zero at north pole
        z = self.vertices[:, 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            w = (self.vertices[:, 0] + 1j * self.vertices[:, 1]) / (1 - z)
            # Handle north pole
            w[z == 1] = np.inf
        return w
    
    def to_pyvista_mesh(self):
        """Convert to PyVista PolyData mesh."""
        # Each face is a triangle, so prepend 3 to each face
        faces_pv = np.hstack([
            np.full((len(self.faces), 1), 3),
            self.faces
        ]).flatten()
        
        return pv.PolyData(self.vertices, faces_pv)
```

### 2.4 Riemann Sphere with Icosahedral Mesh

```python
def riemann_pv(func, subdivisions=5, cmap=Phase(6, 0.6),
              camera_position='default',
              show_grid=False,
              interactive=False,
              **kwargs):
    """
    Plot complex function on Riemann sphere using efficient icosahedral mesh.
    
    Parameters
    ----------
    func : callable
        Complex function to visualize
    subdivisions : int
        Icosahedral subdivision level (higher = finer mesh)
    cmap : Cmap
        Color map for the visualization
    camera_position : str or tuple
        Camera viewing angle
    show_grid : bool
        Show mesh edges
    interactive : bool
        Enable interactive mode
    
    Returns
    -------
    plotter or image array
    """
    # Generate icosahedral mesh
    sphere = IcosahedralSphere(subdivisions)
    mesh = sphere.to_pyvista_mesh()
    
    # Evaluate function at mesh points
    z = sphere.to_complex_plane()
    f_vals = func(z)
    
    # Apply color map
    rgb = cmap.rgb(f_vals)
    colors = (rgb * 255).astype(np.uint8)
    mesh.point_data['colors'] = colors
    
    # Create plotter
    plotter = pv.Plotter(off_screen=not interactive)
    plotter.add_mesh(mesh, 
                    scalars='colors',
                    rgb=True,
                    show_edges=show_grid,
                    smooth_shading=True)
    
    # Configure camera
    if camera_position == 'default':
        plotter.camera_position = [(2.5, 2.5, 2.5), (0, 0, 0), (0, 0, 1)]
    else:
        plotter.camera_position = camera_position
    
    plotter.set_background('white')
    
    if interactive:
        plotter.show()
    else:
        plotter.show()
        return plotter.screenshot()
```

### 2.5 Migration Warnings

Update existing matplotlib functions with performance warnings:

```python
# In plots_3d.py
def plot_landscape(...):
    """Original docstring..."""
    
    import warnings
    warnings.warn(
        "This matplotlib-based 3D function can be slow for detailed plots.\n"
        "For better performance and interactivity, use plot_landscape_pv().\n"
        "Install PyVista: pip install complexplorer[pyvista]",
        PerformanceWarning,
        stacklevel=2
    )
    
    # Original implementation...
```

## Phase 3: Christmas Ornament Generator

### 3.1 Core Implementation

```python
# complexplorer/ornaments.py

class ComplexOrnament:
    """
    Generate 3D-printable ornaments from complex functions.
    
    Uses a modified stereographic projection where the radius
    is scaled by |f(z)| to create interesting 3D shapes.
    """
    
    def __init__(self, func, base_shape='sphere', projection='modified_stereo'):
        self.func = func
        self.base_shape = base_shape
        self.projection = projection
        self.mesh = None
        
    def generate(self, 
                radius=40,           # mm
                thickness=2.0,       # mm wall thickness
                max_radius_scale=1.5,  # Maximum radius scaling
                subdivisions=5,      # Mesh detail
                cmap=Phase(6)):      # Color map for preview
        """
        Generate the 3D mesh for the ornament.
        
        Parameters
        ----------
        radius : float
            Base radius in mm
        thickness : float
            Wall thickness for hollow ornament
        max_radius_scale : float
            Maximum scaling factor for radius modulation
        subdivisions : int
            Icosahedral subdivision level
        cmap : Cmap
            Color map for preview visualization
        """
        # Generate base icosahedral mesh
        sphere = IcosahedralSphere(subdivisions)
        vertices = sphere.vertices.copy()
        
        # Evaluate function
        z = sphere.to_complex_plane()
        f_vals = self.func(z)
        
        # Calculate radius scaling
        modulus = np.abs(f_vals)
        # Normalize and clip
        modulus = np.nan_to_num(modulus, nan=1.0, posinf=max_radius_scale)
        modulus = np.clip(modulus / np.median(modulus), 
                         1/max_radius_scale, max_radius_scale)
        
        # Scale vertices
        outer_verts = vertices * (radius * modulus[:, np.newaxis])
        
        # Create inner surface for hollow ornament
        inner_verts = vertices * ((radius - thickness) * modulus[:, np.newaxis])
        
        # Combine meshes
        self._create_watertight_mesh(outer_verts, inner_verts, sphere.faces)
        
        # Store color information for preview
        self.colors = cmap.rgb(f_vals)
        
        return self
    
    def _create_watertight_mesh(self, outer_verts, inner_verts, faces):
        """Create a watertight mesh with inner and outer surfaces."""
        n_verts = len(outer_verts)
        
        # Combine vertices
        all_verts = np.vstack([outer_verts, inner_verts])
        
        # Outer faces (unchanged)
        outer_faces = faces
        
        # Inner faces (reversed winding)
        inner_faces = faces[:, ::-1] + n_verts
        
        # Connect edges (simplified - assumes sphere topology)
        # This would need more work for complex shapes
        
        # Create PyVista mesh
        faces_pv = []
        for face in outer_faces:
            faces_pv.extend([3] + list(face))
        for face in inner_faces:
            faces_pv.extend([3] + list(face))
            
        self.mesh = pv.PolyData(all_verts, np.array(faces_pv))
    
    def add_hanger(self, style='loop', size=5.0, position='top'):
        """
        Add hanging attachment to the ornament.
        
        Parameters
        ----------
        style : str
            'loop', 'hook', or 'cap'
        size : float
            Size of the hanger in mm
        position : str or tuple
            'top', 'north_pole', or (x,y,z) coordinates
        """
        if style == 'loop':
            # Create a torus for the loop
            loop = pv.Torus(r=size/2, c=size/6)
            
            # Position at top
            if position == 'top':
                # Find highest point
                top_idx = np.argmax(self.mesh.points[:, 2])
                top_point = self.mesh.points[top_idx]
                loop.translate(top_point + [0, 0, size/2])
            
            # Merge with main mesh
            self.mesh = self.mesh.merge(loop)
    
    def export_stl(self, filename, binary=True):
        """
        Export ornament as STL file for 3D printing.
        
        Parameters
        ----------
        filename : str
            Output filename (should end with .stl)
        binary : bool
            Use binary STL format (smaller files)
        """
        if self.mesh is None:
            raise ValueError("Generate mesh first using generate()")
        
        # Ensure mesh is watertight
        if not self.mesh.is_all_triangles():
            self.mesh.triangulate()
        
        # Check and fix normals
        self.mesh.compute_normals(inplace=True)
        
        # Save STL
        self.mesh.save(filename, binary=binary)
        
        # Print statistics
        print(f"Exported {filename}:")
        print(f"  Vertices: {self.mesh.n_points}")
        print(f"  Triangles: {self.mesh.n_cells}")
        print(f"  Bounding box: {self.mesh.bounds}")
        print(f"  Recommended print size: {np.ptp(self.mesh.bounds[::2]):.1f} x "
              f"{np.ptp(self.mesh.bounds[1::2]):.1f} x "
              f"{np.ptp(self.mesh.bounds[2::2]):.1f} mm")
    
    def preview(self, show_colors=True, show_wireframe=False):
        """Interactive 3D preview of the ornament."""
        plotter = pv.Plotter()
        
        if show_colors and hasattr(self, 'colors'):
            # Show with function colors
            colors_uint8 = (self.colors * 255).astype(np.uint8)
            # Only color the outer surface
            n_outer = len(self.colors)
            all_colors = np.vstack([colors_uint8, 
                                   np.full((n_outer, 3), 200, dtype=np.uint8)])
            self.mesh.point_data['colors'] = all_colors
            plotter.add_mesh(self.mesh, scalars='colors', rgb=True)
        else:
            # Show as solid color
            plotter.add_mesh(self.mesh, color='gold', 
                           show_edges=show_wireframe)
        
        plotter.show_grid()
        plotter.show_axes()
        plotter.add_text("3D Ornament Preview", font_size=12)
        plotter.show()
```

### 3.2 Preset Beautiful Functions

```python
# complexplorer/ornament_gallery.py

ORNAMENT_PRESETS = {
    'snowflake': {
        'func': lambda z: z**6 - 1,
        'params': {'max_radius_scale': 1.3, 'subdivisions': 5}
    },
    'flower': {
        'func': lambda z: (z**5 - 1) / (z - 1),
        'params': {'max_radius_scale': 1.5, 'subdivisions': 6}
    },
    'star': {
        'func': lambda z: z**5 + 1/z**5,
        'params': {'max_radius_scale': 1.4, 'subdivisions': 5}
    },
    'spiral': {
        'func': lambda z: np.exp(z/5),
        'params': {'max_radius_scale': 2.0, 'subdivisions': 6}
    },
    'wave': {
        'func': lambda z: np.sin(z) + np.cos(z),
        'params': {'max_radius_scale': 1.3, 'subdivisions': 5}
    },
    'julia': {
        'func': lambda z: z**2 + complex(-0.7, 0.27015),
        'params': {'max_radius_scale': 1.6, 'subdivisions': 6}
    }
}

def create_ornament_from_preset(name, **kwargs):
    """Create ornament from preset design."""
    if name not in ORNAMENT_PRESETS:
        raise ValueError(f"Unknown preset: {name}")
    
    preset = ORNAMENT_PRESETS[name]
    params = preset['params'].copy()
    params.update(kwargs)
    
    ornament = ComplexOrnament(preset['func'])
    ornament.generate(**params)
    return ornament
```

### 3.3 Batch Generation Script

```python
# examples/generate_ornaments.py

def generate_ornament_collection():
    """Generate a collection of ornaments for 3D printing."""
    
    for name, preset in ORNAMENT_PRESETS.items():
        print(f"Generating {name} ornament...")
        
        ornament = create_ornament_from_preset(name)
        ornament.add_hanger(style='loop')
        
        # Export STL
        ornament.export_stl(f"ornaments/{name}_ornament.stl")
        
        # Save preview image
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(ornament.mesh, color='gold')
        plotter.camera_position = 'iso'
        plotter.show()
        plotter.screenshot(f"ornaments/{name}_preview.png")
```

## Phase 4: Additional Enhancements

### 4.1 Performance Optimizations

```python
# complexplorer/cache.py
from functools import lru_cache

class MeshCache:
    """Cache expensive mesh computations."""
    
    @lru_cache(maxsize=32)
    def get_icosahedral_mesh(self, subdivisions):
        return IcosahedralSphere(subdivisions)
    
    @lru_cache(maxsize=128)
    def get_domain_mesh(self, domain_hash, n):
        # Cache domain meshes
        pass
```

### 4.2 New Domain Types

```python
# complexplorer/domain.py additions

class Polygon(Domain):
    """Polygonal domain defined by vertices."""
    
    def __init__(self, vertices):
        self.vertices = np.array(vertices)
        # Calculate bounding box for window
        super().__init__(...)
    
    def contains(self, z):
        """Use winding number algorithm."""
        pass

class HalfPlane(Domain):
    """Half-plane domain."""
    
    def __init__(self, angle=0, include_boundary=True):
        self.angle = angle
        self.include_boundary = include_boundary
        super().__init__(...)
```

### 4.3 Export Utilities

```python
# complexplorer/export.py

def export_to_web(plot_data, filename='plot.html'):
    """Export visualization to interactive HTML using Plotly."""
    pass

def export_animation(func_sequence, filename='animation.mp4'):
    """Create animation from sequence of functions."""
    pass
```

## Implementation Timeline

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| **Phase 1: Unit Tests** | 2 weeks | - Complete test suite with >90% coverage<br>- Visual regression tests<br>- Test documentation |
| **Phase 2: PyVista Core** | 3 weeks | - PyVista backend architecture<br>- plot_landscape_pv implementation<br>- Basic icosahedral mesh |
| **Phase 2.5: Icosahedral Riemann** | 1 week | - Complete icosahedral sphere class<br>- riemann_pv implementation<br>- Performance benchmarks |
| **Phase 3: Ornaments** | 2 weeks | - ComplexOrnament class<br>- STL export functionality<br>- Preset gallery<br>- 3D printing guide |
| **Phase 4: Polish** | 1 week | - Documentation updates<br>- Example notebooks<br>- Performance warnings<br>- Bug fixes |

## Success Metrics

1. **Test Coverage**: >90% code coverage with comprehensive edge case testing
2. **Performance**: 
   - 10-50x speedup for 3D plots with >10k points
   - Interactive rotation at 60 FPS for typical plots
3. **Mesh Efficiency**: 60% fewer triangles for equivalent visual quality on Riemann sphere
4. **3D Printing**: Successfully printed ornaments with common FDM printers
5. **User Experience**: Clear migration path with helpful warnings and documentation

## Development Best Practices

1. **Incremental Development**: Each phase builds on previous work
2. **Backward Compatibility**: All existing code continues to work
3. **Optional Dependencies**: PyVista remains optional via extras_require
4. **Documentation First**: Update docs alongside code changes
5. **Performance Testing**: Benchmark before/after each optimization

## Next Steps

1. Set up pytest and create initial test structure
2. Implement core domain and color map tests
3. Create PyVista demo notebook to validate approach
4. Begin icosahedral mesh implementation

This plan provides a clear roadmap for transforming complexplorer into a best-in-class complex function visualization library while maintaining its elegant design and ease of use.