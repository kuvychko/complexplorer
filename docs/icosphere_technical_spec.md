# Icosahedral Sphere Meshing Technical Specification

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