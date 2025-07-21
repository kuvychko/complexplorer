# Unified STL Mesh Healing Plan - No Cutting Required!

## Core Philosophy
**Users can cut/orient meshes however they want in their slicer software.** Our job is to provide a properly healed, watertight Riemann sphere mesh that handles singularities gracefully.

## The Problem
Complex functions create singularities that manifest as holes/tears in the Riemann sphere mesh when using modulus scaling (arctan, logarithmic, etc.). These holes must be healed to create a watertight mesh suitable for 3D printing.

## Key Insight: Spherical Shell Clipping
Instead of trying to heal jagged, complex tears from singularities, we:
1. Clip the mesh to a clean spherical shell (r_min to r_max)
2. This creates clean, circular boundaries at the clipping radii
3. Cap these circular boundaries with simple radial triangulation
4. Result: Watertight mesh ready for 3D printing

## Implementation Strategy

### 1. Spherical Shell Clipping
```python
def spherical_shell_clip(mesh: pv.PolyData, r_min: float, r_max: float) -> pv.PolyData:
    """
    Clip mesh to spherical shell using PyVista's clip_scalar.
    Creates clean circular boundaries instead of jagged tears.
    """
    # Add radius as scalar field
    mesh["radius"] = np.linalg.norm(mesh.points, axis=1)
    
    # Clip at outer boundary
    clipped = mesh.clip_scalar("radius", value=r_max)
    
    # Clip at inner boundary (if r_min > 0)
    if r_min > 0:
        clipped = clipped.clip_scalar("radius", value=r_min, invert=True)
    
    return clipped
```

### 2. Cap Circular Boundaries
```python
def cap_spherical_boundaries(mesh: pv.PolyData) -> pv.PolyData:
    """
    Cap the clean circular boundaries created by spherical clipping.
    Much simpler than trying to heal arbitrary jagged tears!
    """
    # Extract boundary loops (will be circles at r_min and r_max)
    loops = extract_boundary_loops(mesh)
    
    # Cap each loop with radial triangulation
    for loop in loops:
        cap = create_radial_cap(loop)
        mesh = mesh + cap
    
    return mesh
```

### 3. Complete Pipeline
```python
def generate_healed_ornament(func: Callable, **params) -> pv.PolyData:
    """
    Generate a complete, watertight Riemann sphere mesh.
    NO CUTTING - users do that in their slicer!
    """
    # 1. Generate Riemann sphere with modulus scaling
    sphere = generate_riemann_sphere(func, **params)
    
    # 2. Determine clipping bounds from scaling parameters
    r_min = params.get('r_min', 0.2) * 0.95  # Small margin
    r_max = params.get('r_max', 1.0) * 1.05  # Small margin
    
    # 3. Clip to spherical shell
    clipped = spherical_shell_clip(sphere, r_min, r_max)
    
    # 4. Cap boundaries
    healed = cap_spherical_boundaries(clipped)
    
    # 5. Optional smoothing
    if params.get('smooth', True):
        healed = healed.smooth_taubin(n_iter=20)
    
    # 6. Validate
    validate_mesh_for_printing(healed)
    
    return healed
```

## Advantages Over Complex Hole Detection

1. **Simplicity**: No need to classify hole types or handle complex geometries
2. **Robustness**: Works for ANY singularity configuration
3. **Clean Results**: Circular boundaries are easy to cap cleanly
4. **Performance**: Fast clipping operations instead of complex geometric analysis
5. **Predictable**: User knows exactly where the mesh will be clipped

## API Design

```python
class OrnamentGenerator:
    def generate_ornament(self, 
                         output_file: str,
                         size_mm: float = 80,
                         use_spherical_healing: bool = True):
        """
        Generate a complete watertight ornament.
        
        NO cut_mode parameter - users cut in their slicer!
        """
        # Generate healed sphere
        sphere = self.generate_sphere()
        healed = self.heal_mesh(use_spherical_healing=True)
        
        # Export single complete mesh
        self.export(output_file, healed, size_mm)
```

## Test Functions

1. **Identity**: f(z) = z (singularity at ∞)
2. **Simple Pole**: f(z) = 1/z (singularity at 0)
3. **Multiple Singularities**: f(z) = (z-1)/(z²+z+1)
4. **Ring of Singularities**: f(z) = 1/(z⁴-1)

## File Cleanup

Delete these files as they're no longer needed:
- `stl_healing_plan.md` (superseded)
- `stl_healing_plan_v2.md` (superseded)
- `mesh_healing_advanced.py` (overly complex)
- Any cutting/bisection related code

## Success Metrics

- ✓ Single watertight mesh output (no halves!)
- ✓ No "peg" artifacts
- ✓ Clean handling of all singularities
- ✓ Fast generation (< 2s for 150x150 mesh)
- ✓ User flexibility in slicing orientation

## Next Steps

1. Update `OrnamentGenerator` to remove all cutting logic
2. Simplify API to just output single complete meshes
3. Update examples and documentation
4. Delete old healing plans and complex hole detection code