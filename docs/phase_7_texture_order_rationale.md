# Order of Operations: Why Texture Must Apply After Modulus Scaling

## The Problem

If texture displacement is applied before modulus scaling, the scaling transformation will distort the texture features:

1. **Non-uniform distortion**: Ridges near poles get compressed while those near zeros get stretched
2. **Lost tactile clarity**: What should be uniform ridges become variable height bumps
3. **Misaligned features**: Texture no longer corresponds to visual colormap boundaries

## Visual Example

Consider a function with a pole at origin and the chessboard colormap:

### Incorrect Order (Texture → Scaling):
```
Original sphere → Add texture ridges → Apply log scaling
                  (uniform height)      (ridges get stretched/compressed)
```
Result: Ridges near the pole become huge mountains, ridges far away become tiny bumps

### Correct Order (Scaling → Texture):
```
Original sphere → Apply log scaling → Add texture ridges
                  (deform sphere)     (uniform height on deformed surface)
```
Result: All ridges have consistent height, providing clear tactile feedback

## Mathematical Justification

Let:
- `S₀`: Original unit sphere points
- `f`: Complex function values at sphere points
- `T`: Texture displacement function based on colormap
- `M`: Modulus scaling transformation (arctan, log, etc.)
- `n`: Surface normals

### Incorrect: (S₀ + T(f)·n₀) → M
The modulus scaling M acts on already displaced points, distorting the texture non-uniformly.

### Correct: M(S₀) + T(f)·n₁
where n₁ are normals of the scaled surface M(S₀).

## Implementation Implications

1. **Normal Recomputation**: After modulus scaling, we must recompute normals before applying texture
2. **Texture Independence**: Texture displacement is computed from original f values, not scaled positions
3. **Consistent Height**: Ridge height in mm remains uniform across the entire ornament

## Benefits

1. **Tactile Clarity**: Fingers feel consistent ridges/grooves
2. **Visual-Tactile Correspondence**: Texture matches colormap boundaries exactly
3. **Predictable Results**: Users know that 0.8mm ridges means 0.8mm everywhere
4. **Print Quality**: Uniform features are easier to print reliably

## Code Structure

```python
# In riemann_pv and OrnamentGenerator:

# 1. Compute f on original sphere
f_values = func(complex_points)

# 2. Apply modulus scaling to deform sphere
if scaling != 'constant':
    scaled_mesh = apply_modulus_scaling(mesh, f_values, scaling)
    scaled_mesh.compute_normals()  # Critical step!
else:
    scaled_mesh = mesh

# 3. Apply texture using colormap boundaries
if texture_height > 0:
    displacement = compute_texture_from_colormap(f_values, cmap)
    scaled_mesh.points += scaled_mesh.point_normals * displacement

# 4. Apply colors (always based on original f_values)
colors = cmap.rgb(f_values)
```

This ensures the "relief map" metaphor is accurate - we're adding texture relief to an already-shaped landscape.