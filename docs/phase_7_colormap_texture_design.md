# Colormap-Based Physical Texture Design

## Core Concept

Physical textures for 3D printing are derived directly from colormaps, creating tactile features at visual boundaries. This ensures intuitive correspondence between what users see in visualizations and what they feel on printed objects.

## Implementation Strategy

### 1. Base Texture Generator

```python
class OrnamentGenerator:
    def __init__(self,
                 func: Callable,
                 resolution: int = 150,
                 scaling: str = 'arctan',
                 scaling_params: Optional[Dict[str, Any]] = None,
                 cmap: Optional[Colormap] = None,
                 domain: Optional[Domain] = None,
                 # Texture parameters
                 texture_height: float = 0.0,  # mm, 0 = no texture
                 texture_mode: str = 'ridges',  # 'ridges', 'grooves', 'binary'
                 texture_sharpness: float = 1.0):  # 0-1, affects edge detection
```

### 2. Colormap-to-Texture Algorithm

```python
def _compute_texture_displacement(self, f_values: np.ndarray) -> np.ndarray:
    """Convert colormap boundaries to physical texture."""
    if self.texture_height == 0:
        return np.zeros_like(np.abs(f_values))
    
    # Get colormap HSV values
    hsv = self.cmap.hsv(f_values.ravel()).reshape((*f_values.shape, 3))
    
    # Detect boundaries based on colormap type
    if isinstance(self.cmap, (Chessboard, PolarChessboard, LogRings)):
        # Binary colormaps: use value channel directly
        displacement = self._binary_texture(hsv[:, :, 2])
    else:
        # Continuous colormaps: detect edges via gradients
        displacement = self._gradient_texture(hsv)
    
    return displacement * self.texture_height
```

### 3. Binary Colormap Textures

For discrete colormaps (Chessboard, PolarChessboard, LogRings):

```python
def _binary_texture(self, values: np.ndarray) -> np.ndarray:
    """Create texture from binary (black/white) patterns."""
    if self.texture_mode == 'binary':
        # Direct height mapping: white=raised, black=lowered
        return 2 * values - 1  # Maps [0,1] to [-1,1]
    else:
        # Edge detection for ridges/grooves
        gradient = self._compute_mesh_gradient(values)
        edges = gradient > 0.5  # Binary change detection
        
        if self.texture_mode == 'ridges':
            return edges.astype(float)
        else:  # grooves
            return -edges.astype(float)
```

### 4. Continuous Colormap Textures

For continuous colormaps (Phase with enhancements):

```python
def _gradient_texture(self, hsv: np.ndarray) -> np.ndarray:
    """Create texture from color gradients."""
    # Compute perceptually weighted gradient
    # Hue changes are most noticeable, then value, then saturation
    grad_h = self._compute_mesh_gradient(hsv[:, :, 0])
    grad_s = self._compute_mesh_gradient(hsv[:, :, 1])
    grad_v = self._compute_mesh_gradient(hsv[:, :, 2])
    
    # Handle hue wraparound (0-1 boundary)
    grad_h = np.minimum(grad_h, 1 - grad_h)
    
    # Weighted combination
    gradient_magnitude = np.sqrt(
        grad_h**2 +           # Hue changes (most important)
        0.3 * grad_s**2 +     # Saturation changes
        0.5 * grad_v**2       # Value changes
    )
    
    # Threshold based on sharpness parameter
    threshold = 0.1 + 0.4 * (1 - self.texture_sharpness)
    edges = gradient_magnitude > threshold
    
    if self.texture_mode == 'ridges':
        # Sharp ridges at boundaries
        return edges.astype(float)
    elif self.texture_mode == 'grooves':
        # Grooves at boundaries
        return -edges.astype(float)
    else:  # 'binary'
        # Quantize regions between edges
        return self._quantize_regions(hsv, edges)
```

### 5. Mesh-Aware Gradient Computation

```python
def _compute_mesh_gradient(self, values: np.ndarray) -> np.ndarray:
    """Compute gradient on spherical mesh."""
    # This is a simplified version - actual implementation
    # needs to account for mesh topology and metric
    
    if values.ndim == 1:  # Already flattened
        values_2d = values.reshape(self._mesh_shape)
    else:
        values_2d = values
    
    # Compute gradients with proper boundary handling
    grad_theta = np.gradient(values_2d, axis=0)
    grad_phi = np.gradient(values_2d, axis=1)
    
    # Account for spherical metric (simplified)
    # Near poles, phi gradients need scaling
    theta_weights = np.sin(self._theta_coords)
    grad_phi = grad_phi / (theta_weights + 1e-6)
    
    return np.sqrt(grad_theta**2 + grad_phi**2).ravel()
```

### 6. Texture Preview

```python
def preview_texture(self, ax=None, show_colormap=True):
    """Visualize texture as height map overlay."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Generate mesh and compute texture
    mesh = self._generate_sphere_mesh()
    f_values = self._compute_f_on_mesh(mesh)
    displacement = self._compute_texture_displacement(f_values)
    
    if show_colormap:
        # Show colormap with displacement as contours
        rgb = self.cmap.rgb(f_values)
        # ... plot colored sphere with contour lines at ridges
    else:
        # Show displacement as grayscale height map
        ax.imshow(displacement.reshape(self._mesh_shape), 
                  cmap='RdBu', vmin=-1, vmax=1)
```

## Usage Examples

### Example 1: Phase Portrait with Contour Lines
```python
# Enhanced phase portrait with automatic ridge placement
cmap = Phase(n_phi=12, r_linear_step=1.0, v_base=0.4)
gen = OrnamentGenerator(
    func=lambda z: (z**2 - 1) / (z**2 + 1),
    cmap=cmap,
    texture_height=0.8,  # 0.8mm ridges
    texture_mode='ridges',
    texture_sharpness=0.9  # Sharp edges
)
```

### Example 2: Chessboard with Binary Height
```python
# Chessboard pattern becomes raised/lowered squares
cmap = Chessboard(spacing=0.5)
gen = OrnamentGenerator(
    func=lambda z: z,
    cmap=cmap,
    texture_height=0.5,  # ±0.5mm height difference
    texture_mode='binary'  # Direct height mapping
)
```

### Example 3: Polar Pattern with Grooves
```python
# Polar chessboard with grooves at boundaries
cmap = PolarChessboard(n_phi=8, n_r=6)
gen = OrnamentGenerator(
    func=lambda z: z**3 - 1,
    cmap=cmap,
    texture_height=0.6,  # 0.6mm deep grooves
    texture_mode='grooves'
)
```

## Advantages of This Approach

1. **Intuitive**: Textures correspond directly to visual features
2. **Flexible**: Any colormap can become a texture
3. **Consistent**: Same boundaries in visual and tactile representation
4. **Efficient**: Reuses existing colormap calculations
5. **Predictable**: Users can preview exactly where textures will appear

## Special Considerations

### Auto-Scaling with Phase Colormap
When using `Phase(auto_scale_r=True)`, the ridges automatically appear at "square" boundaries in the enhanced phase portrait, creating a natural grid aligned with the phase sectors.

### Logarithmic Rings
`LogRings` colormap creates concentric circular ridges at exponentially spaced intervals - perfect for emphasizing pole/zero behavior.

### Custom Colormaps
Users can create custom colormaps that produce specific texture patterns, giving full control over the tactile design.

## Implementation Notes

1. **Edge Detection**: Use discrete gradients appropriate for mesh topology
2. **Displacement Limits**: Cap displacement at 30% of local mesh spacing
3. **Smoothing**: Optional local smoothing to prevent artifacts
4. **Preview**: Always provide visual preview before expensive STL generation