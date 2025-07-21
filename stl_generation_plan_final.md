# STL Generation Reimplementation Plan

## 1. Code Analysis and Consolidation

### 1.1 Analyze Current Implementation
- [ ] Read and document all functions in `complexplorer/mesh_utils.py`
- [ ] Read and document all functions in `complexplorer/stl_export/mesh_utils.py`
- [ ] Identify duplicated functionality
- [ ] Identify STL-specific vs general mesh utilities

### 1.2 File Reorganization Strategy
**Decision: Option B - Clear separation of concerns**
- Keep general mesh utilities in `complexplorer/mesh_utils.py`
- Rename `complexplorer/stl_export/mesh_utils.py` to `complexplorer/stl_export/stl_utils.py`
- Move STL-specific functions (cutting, capping, healing) to the renamed file
- Update all imports accordingly

### 1.3 Remove Icosahedral Grid Support
- [ ] Remove all icosahedral mesh generation code
- [ ] Remove `IcosahedralSphereGenerator` class
- [ ] Keep only `RectangularSphereGenerator` (lat-lon grid)
- [ ] Simplify mesh generation interface

## 2. Research Existing STL Libraries

### 2.1 Investigate Python STL Libraries
- [ ] `numpy-stl` - Check if it has mesh cutting capabilities
- [ ] `trimesh` - More advanced mesh operations
- [ ] `pymesh` - Professional mesh processing
- [ ] `meshlab` Python bindings - If available

### 2.2 Evaluate Capabilities
- [ ] Can they cut meshes along arbitrary planes?
- [ ] Can they ensure watertight meshes?
- [ ] Can they cap open boundaries?
- [ ] Performance comparison with current implementation

## 3. Basic Sphere Implementation

### 3.1 Generate Perfect Unit Sphere
```python
def generate_unit_sphere(resolution: int = 100) -> pv.PolyData:
    """Generate a perfect unit sphere with rectangular grid."""
    # Use only rectangular (lat-lon) grid
    # Avoid exact poles (use small offset like 0.01)
    # Return clean PyVista mesh
```

### 3.2 Validate Sphere Quality
- [ ] Check that it's manifold
- [ ] Check that it's watertight
- [ ] Visualize and verify with user
- [ ] Test at different resolutions (50, 100, 150)

## 4. Implement General Bisecting Planes

### 4.1 Plane Definition with Collinearity Check
```python
def define_bisecting_plane(z1: complex, z2: complex) -> dict:
    """
    Define a bisecting plane passing through origin and two complex points.
    
    Parameters
    ----------
    z1, z2 : complex
        Two points in the complex plane that define the bisecting plane.
        The plane passes through origin, z1, and z2 (when projected to sphere).
    
    Returns
    -------
    dict with:
        - origin: [0, 0, 0]
        - normal: normalized normal vector to the plane
        
    Raises
    ------
    ValueError
        If the three points (origin, z1, z2) are collinear.
    """
    # Project z1, z2 to sphere via inverse stereographic projection
    p1 = inverse_stereographic(z1)
    p2 = inverse_stereographic(z2)
    
    # Check collinearity
    # Points are collinear if cross product is near zero
    cross = np.cross(p1, p2)
    if np.linalg.norm(cross) < 1e-6:
        raise ValueError(f"Points are collinear: origin, {z1}, {z2}")
    
    # Normal is the cross product
    normal = cross / np.linalg.norm(cross)
    
    return {
        'origin': np.array([0, 0, 0]),
        'normal': normal
    }
```

### 4.2 Special Cases for Common Cuts
```python
def get_standard_cut(mode: str) -> dict:
    """Get standard cutting planes."""
    if mode == 'real':
        # Cut along real axis (y=0 plane)
        return define_bisecting_plane(1+0j, -1+0j)
    elif mode == 'imaginary':
        # Cut along imaginary axis (x=0 plane)
        return define_bisecting_plane(0+1j, 0-1j)
    elif mode.startswith('angle:'):
        # Cut at specific angle
        angle = float(mode.split(':')[1]) * np.pi / 180
        z1 = np.exp(1j * angle)
        z2 = np.exp(1j * (angle + np.pi))
        return define_bisecting_plane(z1, z2)
    elif mode == 'c-plane':
        # C-plane cut: origin + 1 + j (45° angle in complex plane)
        # This creates a tilted plane through origin, 1, and j
        return define_bisecting_plane(1+0j, 0+1j)
```

### 4.3 General Cutting Implementation
```python
def cut_sphere_with_plane(sphere: pv.PolyData,
                         plane_origin: np.ndarray,
                         plane_normal: np.ndarray) -> tuple[pv.PolyData, pv.PolyData]:
    """
    Cut sphere with arbitrary plane through origin.
    
    Returns two halves, each properly capped.
    """
    # Use PyVista clip with exact plane
    # Ensure boundaries are perfectly planar
    # Cap with robust triangulation
    # Handle all orientations correctly
```

### 4.4 Validation Steps
- [ ] Test horizontal cut (real axis)
- [ ] Test vertical cut (imaginary axis)
- [ ] Test diagonal cuts at 45°, 30°, 60°
- [ ] Test C-plane cut (plane through origin, 1, and j)
- [ ] Test collinearity detection (should raise error for z1=1, z2=2)
- [ ] Test arbitrary cuts defined by two complex points
- [ ] Verify watertightness for all cuts

## 5. Add Function-Based Radius Scaling

### 5.1 Implement Scaling Methods
```python
class RadiusScaling:
    @staticmethod
    def constant(moduli: np.ndarray, radius: float = 1.0) -> np.ndarray:
        """No scaling - constant radius."""
        
    @staticmethod
    def arctan(moduli: np.ndarray, r_min: float = 0.2, r_max: float = 1.0) -> np.ndarray:
        """Smooth arctangent scaling: maps [0,∞) to [r_min, r_max]."""
        
    @staticmethod
    def linear_clamp(moduli: np.ndarray, m_max: float = 10, 
                     r_min: float = 0.2, r_max: float = 1.0) -> np.ndarray:
        """Linear scaling up to m_max, then clamped."""
        
    @staticmethod
    def logarithmic(moduli: np.ndarray, base: float = np.e,
                    r_min: float = 0.2, r_max: float = 1.0) -> np.ndarray:
        """Logarithmic scaling for large dynamic range."""
```

### 5.2 Apply Scaling to Sphere
```python
def apply_radius_scaling(sphere: pv.PolyData, 
                        func: Callable,
                        scaling_method: str = 'arctan',
                        scaling_params: dict = None) -> pv.PolyData:
    """Apply radius scaling based on complex function."""
    # Evaluate function at sphere points (via stereographic projection)
    # Handle infinities and singularities
    # Calculate moduli
    # Apply selected scaling method
    # Scale vertex positions radially
    # Preserve mesh topology
```

## 6. STL Topology Validation

### 6.1 Implement Validation Suite
```python
def validate_stl_topology(mesh: pv.PolyData) -> dict:
    """Comprehensive topology validation."""
    return {
        'is_manifold': check_manifold(mesh),
        'is_watertight': check_watertight(mesh),
        'has_degenerate_faces': check_degenerate(mesh),
        'boundary_edges': count_boundary_edges(mesh),
        'non_manifold_edges': count_non_manifold_edges(mesh),
        'connected_components': count_components(mesh),
        'face_normals_consistent': check_normals(mesh),
        'self_intersections': check_self_intersections(mesh)
    }
```

### 6.2 Minimal Repair Functions
```python
def repair_mesh_minimal(mesh: pv.PolyData) -> pv.PolyData:
    """Minimal mesh repair - only fix critical issues."""
    # Remove degenerate faces (area < epsilon)
    # Fix normal orientations
    # Remove duplicate vertices
    # Do NOT aggressively smooth or modify geometry
```

## 7. Testing Strategy

### 7.1 Test Cases
1. **Identity function** `f(z) = z`
   - With constant scaling → perfect sphere
   - With arctan scaling → smooth deformation
   - Various cutting planes including C-plane cut

2. **Simple pole** `f(z) = 1/z`
   - Handle singularity at origin
   - Test domain restriction

3. **Rational function** `f(z) = (z-1)/(z²+z+1)`
   - Multiple singularities
   - Complex geometry

4. **Collinearity tests**
   - Should fail: define_bisecting_plane(1+0j, 2+0j)
   - Should fail: define_bisecting_plane(1+0j, -1+0j) with origin
   - Should pass: define_bisecting_plane(1+0j, 0+1j)

5. **Custom cuts**
   - C-plane cut (through origin, 1, and j)
   - Define plane with z1 = 1+i, z2 = -1+i
   - Define plane with z1 = 2, z2 = 2i

### 7.2 Validation Criteria
- [ ] No "peg" artifacts at poles
- [ ] Watertight meshes for all cuts
- [ ] Proper scaling applied
- [ ] Clean, planar boundaries at cut plane
- [ ] Correct orientation of cut halves
- [ ] File sizes under 10MB for reasonable resolutions

## 8. Implementation Timeline

1. **Phase 1**: File reorganization (30 min)
   - Rename files, update imports
   - Remove icosahedral code

2. **Phase 2**: Research external libraries (30 min)
   - Quick evaluation of trimesh capabilities

3. **Phase 3**: Basic sphere generation (30 min)
   - Clean rectangular grid implementation

4. **Phase 4**: General cutting implementation (2 hours)
   - Arbitrary plane cutting with collinearity check
   - C-plane cut support (origin + 1 + j)
   - Robust capping algorithm

5. **Phase 5**: Scaling implementation (1 hour)
   - All scaling methods
   - Proper singularity handling

6. **Phase 6**: Validation and testing (1 hour)
   - Comprehensive test suite
   - Visual verification

## 9. Key Design Decisions

1. **File organization**: ✓ Clear separation (Option B)
2. **Mesh type**: ✓ Rectangular grid only
3. **Cutting planes**: ✓ General bisecting planes via two complex points
4. **Special cuts**: ✓ Real, imaginary, angle, and C-plane (1+j)
5. **Collinearity check**: ✓ Implemented in define_bisecting_plane
6. **External library**: TBD after research
7. **Mesh healing**: Minimal intervention only
8. **Performance**: Prioritize correctness over speed

## 10. Success Criteria

- [ ] Clean codebase with no duplicate functions
- [ ] No visual artifacts (especially pole "pegs")
- [ ] Watertight STL files for all test cases
- [ ] Support for arbitrary bisecting planes with validation
- [ ] Support for C-plane cut (origin + 1 + j)
- [ ] Robust collinearity detection
- [ ] Works reliably for various complex functions
- [ ] Clear API and documentation
- [ ] Reasonable performance (< 5s for 100x100 mesh)