# STL Christmas Ornament Generation Plan

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