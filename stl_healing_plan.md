# STL Mesh Healing Plan

## Core Insight
Instead of implementing cutting for STL export (users can cut in their slicer), focus on creating properly healed, watertight meshes that handle singularities correctly.

## Key Example: Identity Function
The identity function f(z) = z with arctan scaling demonstrates the core challenge:
- At z = ∞ (north pole), we have a singularity
- Arctan scaling creates a finite radius, but leaves a tear/hole
- This tear should be healed with a planar cap

## Mesh Healing Strategy

### 1. Singularity Detection
```python
def detect_mesh_holes(mesh: pv.PolyData) -> List[BoundaryLoop]:
    """
    Detect holes in the mesh manifold.
    
    Returns list of boundary loops that need healing.
    """
    # Extract boundary edges
    # Group into connected loops
    # Return each loop's vertices
```

### 2. Hole Classification
```python
def classify_hole(loop: BoundaryLoop, mesh: pv.PolyData) -> HoleType:
    """
    Classify the type of hole for appropriate healing.
    
    Types:
    - POLE_SINGULARITY: At north/south pole (z ≈ ±1)
    - PLANAR_TEAR: Approximately planar boundary
    - COMPLEX_HOLE: Irregular boundary
    """
```

### 3. Healing Methods

#### 3.1 Planar Cap (for pole singularities)
```python
def heal_with_planar_cap(mesh: pv.PolyData, loop: BoundaryLoop) -> pv.PolyData:
    """
    Heal hole with a planar triangulated cap.
    
    Best for:
    - Singularities at infinity (north pole)
    - Tears that are approximately planar
    """
    # Find best-fit plane for boundary loop
    # Triangulate the planar region
    # Add to mesh
```

#### 3.2 Smooth Interpolation
```python
def heal_with_interpolation(mesh: pv.PolyData, loop: BoundaryLoop) -> pv.PolyData:
    """
    Heal hole by interpolating from surrounding geometry.
    
    Best for:
    - Small holes in smooth regions
    - Non-singular tears
    """
```

### 4. Mesh Validation
```python
def validate_healed_mesh(mesh: pv.PolyData) -> ValidationReport:
    """
    Ensure mesh is suitable for 3D printing.
    
    Checks:
    - Is manifold (no non-manifold edges)
    - Is watertight (no boundary edges)
    - Has consistent normals
    - No self-intersections
    - No degenerate faces
    """
```

## Implementation Steps

### Phase 1: Robust Hole Detection
- [ ] Implement boundary edge extraction
- [ ] Group edges into connected loops
- [ ] Handle multiple disconnected holes

### Phase 2: Smart Healing
- [ ] Implement planar cap healing for pole singularities
- [ ] Add best-fit plane calculation
- [ ] Ensure proper normal orientation

### Phase 3: Validation Suite
- [ ] Manifold checking
- [ ] Watertightness verification
- [ ] Normal consistency
- [ ] Face quality metrics

## Test Cases

### 1. Identity Function
```python
# f(z) = z with arctan scaling
# Expected: Hole at north pole, healed with planar cap
```

### 2. Simple Pole
```python
# f(z) = 1/z
# Expected: Holes at both poles, both healed appropriately
```

### 3. Multiple Singularities
```python
# f(z) = (z-1)/(z²+z+1)
# Expected: Multiple holes, each healed based on type
```

## Benefits of This Approach

1. **User Flexibility**: Users can cut/orient however they want in slicer
2. **Mathematical Correctness**: Proper handling of singularities
3. **Robustness**: Works for any complex function
4. **Simplicity**: No need for complex cutting algorithms

## Success Criteria

- [ ] All test functions produce watertight meshes
- [ ] No "peg" artifacts at poles
- [ ] Smooth, natural healing at singularities
- [ ] Fast performance (< 1s for 100x100 mesh)
- [ ] Clear API for customization