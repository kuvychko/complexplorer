# Texture Preview Integration with PyVista Riemann Sphere

## Design Philosophy

Physical texture preview is integrated directly into the existing `riemann_pv` function, allowing users to visualize both color and texture displacement in the same interactive 3D viewer. This maintains the established workflow: visualize → verify → export.

## Updated riemann_pv API

```python
def riemann_pv(
    func: Callable[[Union[complex, np.ndarray]], Union[complex, np.ndarray]],
    n: int = 150,
    cmap: Optional[Colormap] = None,
    domain: Optional[Domain] = None,
    scaling: str = 'default',
    scaling_params: Optional[Dict[str, float]] = None,
    plotter: Optional['pv.Plotter'] = None,
    show: bool = True,
    show_grid: bool = False,
    show_edges: bool = False,
    # Texture preview parameters
    texture_height: float = 0.0,  # mm, 0 = no texture
    texture_mode: str = 'ridges',  # 'ridges', 'grooves', 'binary'
    texture_sharpness: float = 1.0,  # 0-1, edge detection sensitivity
    texture_preview_scale: float = 5.0,  # Exaggeration factor for preview
    **kwargs
) -> 'pv.Plotter':
    """
    Display Riemann sphere visualization with optional texture preview.
    
    Parameters
    ----------
    texture_height : float
        Physical texture height in mm. 0 = no texture (smooth surface).
        
    texture_mode : str
        How to create texture from colormap boundaries:
        - 'ridges': Raised lines at color boundaries
        - 'grooves': Indented lines at color boundaries  
        - 'binary': Direct height mapping (for binary colormaps)
        
    texture_sharpness : float
        Edge detection sensitivity (0-1). Higher = sharper edge detection.
        
    texture_preview_scale : float
        Scale factor for texture preview. Actual texture is multiplied
        by this factor for better visibility. Default 5.0 means 5x 
        exaggeration in the preview.
    """
```

## Implementation Details

### 1. Texture Calculation with Proper Order of Operations

```python
def _generate_textured_sphere_mesh(
    func: Callable,
    cmap: Colormap,
    n: int,
    domain: Optional[Domain],
    scaling: str,
    scaling_params: Dict[str, float],
    texture_height: float,
    texture_mode: str,
    texture_sharpness: float,
    preview_scale: float
) -> 'pv.PolyData':
    """Generate sphere mesh with modulus scaling THEN texture displacement."""
    
    # Generate base sphere mesh
    mesh = _generate_sphere_mesh(n)
    
    # Compute function values
    f_values = _compute_f_on_mesh(mesh, func, domain)
    
    # STEP 1: Apply modulus scaling to sphere positions
    if scaling != 'default':
        # Scale the sphere based on modulus
        mesh = _apply_modulus_scaling(mesh, f_values, scaling, scaling_params)
        # Recompute normals after scaling
        mesh.compute_normals(point_normals=True, inplace=True)
    
    # STEP 2: Apply texture displacement AFTER scaling
    if texture_height > 0:
        # Compute texture displacement from colormap
        # This uses the ORIGINAL f_values, not the scaled positions
        displacement = _compute_texture_from_colormap(
            f_values, cmap, texture_mode, texture_sharpness
        )
        
        # Scale for preview (will be documented in UI)
        displacement *= texture_height * preview_scale
        
        # Apply displacement along the UPDATED normals (post-scaling)
        normals = mesh.point_normals
        mesh.points += normals * displacement[:, np.newaxis]
        
        # Store actual (non-scaled) displacement for STL export
        mesh['texture_displacement'] = displacement / preview_scale
    
    # Add color data (based on original f_values)
    mesh['rgb'] = cmap.rgb(f_values)
    
    return mesh
```

### 2. Visual Feedback in PyVista

```python
# In riemann_pv function:
if texture_height > 0:
    # Add text annotation about texture preview
    plotter.add_text(
        f"Texture Preview: {texture_preview_scale}× scale\n"
        f"Actual height: ±{texture_height}mm",
        position='upper_right',
        font_size=10,
        color='white'
    )
    
    # Optionally add wireframe to show mesh deformation
    if show_edges:
        plotter.add_mesh(
            mesh.copy(),
            style='wireframe',
            color='gray',
            opacity=0.3,
            line_width=0.5
        )
```

### 3. Colormap-Specific Texture Generation

```python
def _compute_texture_from_colormap(
    f_values: np.ndarray,
    cmap: Colormap,
    mode: str,
    sharpness: float
) -> np.ndarray:
    """Convert colormap to texture displacement."""
    
    # Get HSV values from colormap
    hsv = cmap.hsv(f_values.ravel())
    
    if isinstance(cmap, (Chessboard, PolarChessboard, LogRings)):
        # Binary colormaps: use value channel directly
        values = hsv[:, 2]  # V channel
        
        if mode == 'binary':
            # Direct mapping: white=up, black=down
            return 2 * values - 1  # Maps [0,1] to [-1,1]
        else:
            # Edge detection for ridges/grooves
            gradient = _compute_sphere_gradient(values, mesh_shape)
            edges = gradient > 0.5
            
            if mode == 'ridges':
                return edges.astype(float)
            else:  # grooves
                return -edges.astype(float)
    
    else:
        # Continuous colormaps: detect gradients
        # ... (as in previous design)
```

## Workflow Integration

### 1. Interactive Preview Workflow

```python
# User workflow example:
import complexplorer as cp

# Define function and colormap
func = lambda z: (z**2 - 1) / (z**2 + 1)
cmap = cp.Phase(n_phi=12, r_linear_step=1.0)

# Preview with texture
cp.riemann_pv(
    func, 
    cmap=cmap,
    texture_height=0.8,  # 0.8mm ridges
    texture_mode='ridges',
    texture_preview_scale=10.0  # 10x for clear visibility
)

# If satisfied, export with actual scale
gen = cp.OrnamentGenerator(
    func=func,
    cmap=cmap,
    texture_height=0.8,  # Same as preview
    texture_mode='ridges'
)
gen.generate_stl('ornament.stl')
```

### 2. A/B Comparison

```python
# Compare smooth vs textured in same viewer
plotter = pv.Plotter(shape=(1, 2))

# Left: Smooth (colors only)
plotter.subplot(0, 0)
cp.riemann_pv(func, cmap=cmap, plotter=plotter, show=False)
plotter.add_text("Smooth Surface", position='upper_edge')

# Right: With texture
plotter.subplot(0, 1)
cp.riemann_pv(
    func, cmap=cmap, 
    texture_height=0.8,
    texture_mode='ridges',
    plotter=plotter, 
    show=False
)
plotter.add_text("With Texture", position='upper_edge')

plotter.link_views()
plotter.show()
```

## Advantages

1. **Unified Workflow**: Visualize → Verify → Export, all with same function
2. **Visual Verification**: See both color and texture together
3. **Interactive**: Rotate, zoom to inspect texture details
4. **Consistent**: Same mesh generation for preview and export
5. **Quality Check**: Spot artifacts or issues before printing

## Implementation Notes

### Critical Order of Operations
1. **Generate base sphere mesh** (unit sphere)
2. **Compute function values** on original sphere points
3. **Apply modulus scaling** (arctan, log, etc.) to deform sphere
4. **Recompute normals** after deformation
5. **Apply texture displacement** along new normals

This order ensures:
- Texture ridges maintain consistent height
- Colormap boundaries are accurately represented
- No distortion of tactile features by scaling

### Mesh Considerations
- Use same mesh resolution for preview and export
- Store unscaled displacement in mesh for STL export
- Apply displacement along vertex normals (AFTER scaling)
- Recompute normals after modulus scaling, before texture

### Performance
- Texture calculation adds minimal overhead
- Gradient computation is fast for typical resolutions
- Preview scale doesn't affect computation time
- Normal recomputation is necessary but fast

### User Experience
- Clear indication of preview scale in UI
- Document that preview is exaggerated for visibility
- Same parameters work for both preview and export
- Texture appears uniform regardless of modulus scaling

## Future Extensions

1. **Texture Intensity Slider**: Interactive adjustment in viewer
2. **Multiple Texture Layers**: Combine phase and modulus textures
3. **Adaptive Scaling**: Auto-adjust preview scale based on ornament size
4. **Export Preview**: Save screenshot with texture for documentation