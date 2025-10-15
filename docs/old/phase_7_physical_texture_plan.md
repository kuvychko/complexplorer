# Phase 7: Physical Texture Implementation Plan

## Overview

Add physical texture to STL exports that creates sharp, tactile contours on the ornament surface. The texture will encode phase and/or modulus information using abrupt displacements similar to how the chessboard colormap creates high-contrast boundaries.

## Key Insights from Colormap Analysis

1. **Chessboard Pattern**: Uses `floor()` to quantize continuous values into discrete regions, then creates boolean (0 or 1) patterns
2. **Sawtooth Function**: Creates repeating ramps with sharp drops at boundaries
3. **High Contrast**: Both mechanisms avoid smooth transitions in favor of abrupt changes

## Implementation Strategy

### 1. Contour-Based Displacement

Instead of smooth sinusoidal displacement, use step functions and sharp ridges:

```python
def _compute_contour_displacement(self, f_values: np.ndarray, mode: str) -> np.ndarray:
    """Compute displacement creating sharp contour lines.
    
    Returns displacement values that create ridges (positive) or grooves (negative).
    """
    params = self.texture_params or {}
    
    if mode == 'phase_contours':
        # Create ridges at fixed phase intervals
        phase = np.angle(f_values)
        n_contours = params.get('n_phase_contours', 12)
        
        # Quantize phase into discrete levels
        phase_normalized = (phase + np.pi) / (2 * np.pi)  # [0, 1]
        phase_level = np.floor(phase_normalized * n_contours)
        
        # Create ridges at level boundaries
        # Use gradient to find where levels change
        return self._create_ridge_at_boundaries(phase_level)
        
    elif mode == 'modulus_contours':
        # Create rings at logarithmic modulus intervals
        modulus = np.abs(f_values)
        n_rings = params.get('n_modulus_rings', 10)
        base = params.get('log_base', np.e)
        
        # Logarithmic quantization
        with np.errstate(divide='ignore', invalid='ignore'):
            log_mod = np.log(modulus + 1) / np.log(base)
        mod_level = np.floor(log_mod * n_rings)
        
        return self._create_ridge_at_boundaries(mod_level)
        
    elif mode == 'chessboard':
        # Binary displacement matching chessboard pattern
        return self._create_chessboard_texture(f_values)
```

### 2. Ridge Creation at Boundaries

```python
def _create_ridge_at_boundaries(self, level_array: np.ndarray) -> np.ndarray:
    """Create ridges where quantized levels change.
    
    This creates a thin raised ridge at contour boundaries,
    similar to topographic maps.
    """
    params = self.texture_params or {}
    ridge_height = params.get('ridge_height', 0.8)  # mm
    ridge_width = params.get('ridge_width', 0.3)   # relative to mesh resolution
    ridge_profile = params.get('ridge_profile', 'sharp')  # 'sharp', 'rounded'
    
    # Find boundaries using discrete gradient
    # Note: This is simplified - actual implementation needs proper
    # gradient computation on the sphere mesh
    gradient_magnitude = self._compute_discrete_gradient(level_array)
    
    # Create ridge where gradient is high (level changes)
    is_boundary = gradient_magnitude > 0.5
    
    if ridge_profile == 'sharp':
        # Binary displacement
        displacement = ridge_height * is_boundary.astype(float)
    else:
        # Slightly rounded ridge
        displacement = ridge_height * np.tanh(5 * gradient_magnitude)
    
    return displacement
```

### 3. Chessboard-Style Binary Texture

```python
def _create_chessboard_texture(self, f_values: np.ndarray) -> np.ndarray:
    """Create raised/lowered regions in a chessboard pattern.
    
    This creates a tactile pattern where alternating regions
    are raised or lowered, making the pattern easily felt.
    """
    params = self.texture_params or {}
    height_difference = params.get('height_diff', 0.5)  # mm
    pattern_type = params.get('pattern', 'cartesian')  # or 'polar'
    
    if pattern_type == 'cartesian':
        # Cartesian chessboard
        spacing = params.get('spacing', 1.0)
        z_shifted = f_values / spacing
        
        real_idx = np.floor(np.real(z_shifted)).astype(int)
        imag_idx = np.floor(np.imag(z_shifted)).astype(int)
        
        # Binary pattern
        is_raised = ((real_idx + imag_idx) % 2 == 0)
        
    else:  # polar
        # Polar chessboard
        n_phi = params.get('n_phi', 12)
        n_r = params.get('n_r', 8)
        
        phase = np.angle(f_values)
        phase_idx = np.floor((phase + np.pi) * n_phi / (2 * np.pi)).astype(int)
        
        r = np.abs(f_values)
        r_idx = np.floor(np.log(r + 1) * n_r).astype(int)
        
        is_raised = ((phase_idx + r_idx) % 2 == 0)
    
    # Convert to displacement: raised = positive, lowered = negative
    displacement = height_difference * (2 * is_raised.astype(float) - 1) / 2
    
    return displacement
```

### 4. Groove/Valley Texture Option

```python
def _create_groove_texture(self, f_values: np.ndarray) -> np.ndarray:
    """Create grooves (negative displacement) along contours.
    
    Instead of ridges, this creates valleys that can guide
    touch or hold decorative materials (paint, foil, etc).
    """
    params = self.texture_params or {}
    groove_depth = params.get('groove_depth', 0.5)  # mm
    groove_width = params.get('groove_width', 0.4)
    
    # Similar to ridge creation but with negative displacement
    phase = np.angle(f_values)
    n_grooves = params.get('n_phase_grooves', 8)
    
    # Create grooves at specific phase values
    groove_phases = np.linspace(-np.pi, np.pi, n_grooves, endpoint=False)
    
    # Find points near groove phases
    min_distance = np.inf * np.ones_like(phase)
    for groove_phase in groove_phases:
        distance = np.abs(phase - groove_phase)
        # Handle phase wrap-around
        distance = np.minimum(distance, 2*np.pi - distance)
        min_distance = np.minimum(min_distance, distance)
    
    # Create groove profile
    in_groove = min_distance < (groove_width * np.pi / n_grooves)
    displacement = -groove_depth * in_groove.astype(float)
    
    return displacement
```

### 5. Updated API Design

```python
class OrnamentGenerator:
    def __init__(self,
                 func: Callable,
                 resolution: int = 150,
                 scaling: str = 'arctan',
                 scaling_params: Optional[Dict[str, Any]] = None,
                 cmap: Optional[Colormap] = None,
                 domain: Optional[Domain] = None,
                 # New texture parameters
                 texture_mode: Optional[str] = None,
                 texture_params: Optional[Dict[str, Any]] = None):
        """
        texture_mode : str, optional
            Physical texture mode:
            - None: No texture (smooth surface)
            - 'phase_contours': Sharp ridges at phase contour lines
            - 'modulus_contours': Rings at modulus levels
            - 'chessboard': Binary raised/lowered pattern
            - 'phase_grooves': Grooves along phase lines
            - 'combined': Multiple texture features
            
        texture_params : dict, optional
            Mode-specific parameters:
            
            For 'phase_contours':
            - n_phase_contours: Number of contour lines (default: 12)
            - ridge_height: Height in mm (default: 0.8)
            - ridge_profile: 'sharp' or 'rounded' (default: 'sharp')
            
            For 'modulus_contours':
            - n_modulus_rings: Number of rings (default: 10)
            - log_base: Logarithm base for spacing (default: e)
            - ridge_height: Height in mm (default: 0.8)
            
            For 'chessboard':
            - pattern: 'cartesian' or 'polar' (default: 'cartesian')
            - height_diff: Height difference in mm (default: 0.5)
            - spacing: Square size (cartesian) (default: 1.0)
            - n_phi, n_r: Divisions (polar) (default: 12, 8)
            
            For 'phase_grooves':
            - n_phase_grooves: Number of grooves (default: 8)
            - groove_depth: Depth in mm (default: 0.5)
            - groove_width: Relative width (default: 0.4)
        """
```

### 6. Mesh Processing Considerations

```python
def _apply_texture_with_mesh_quality(self, mesh: 'pv.PolyData', 
                                   f_values: np.ndarray) -> 'pv.PolyData':
    """Apply texture while maintaining mesh quality.
    
    Sharp displacements can create self-intersections or
    thin features. This method includes safeguards.
    """
    # Compute displacement
    displacement = self._compute_contour_displacement(f_values, self.texture_mode)
    
    # Limit displacement based on local mesh density
    edge_lengths = self._compute_average_edge_length(mesh)
    max_safe_displacement = 0.3 * edge_lengths  # 30% of local edge length
    
    # Clip displacement to safe range
    displacement = np.clip(displacement, 
                          -max_safe_displacement, 
                          max_safe_displacement)
    
    # Apply along normals
    normals = mesh.point_normals
    mesh.points += normals * displacement[:, np.newaxis]
    
    # Optional: Local smoothing only at problem areas
    if self.texture_params.get('fix_artifacts', True):
        mesh = self._fix_texture_artifacts(mesh)
    
    return mesh
```

### 7. Example Usage

```python
# Example 1: Phase contour lines (like topographic map)
gen = OrnamentGenerator(
    func=lambda z: (z**2 - 1) / (z**2 + 1),
    resolution=200,
    texture_mode='phase_contours',
    texture_params={
        'n_phase_contours': 16,
        'ridge_height': 1.0,  # 1mm ridges
        'ridge_profile': 'sharp'
    }
)

# Example 2: Polar chessboard texture
gen = OrnamentGenerator(
    func=lambda z: z**3 - 1,
    resolution=200,
    texture_mode='chessboard',
    texture_params={
        'pattern': 'polar',
        'n_phi': 12,
        'n_r': 6,
        'height_diff': 0.6  # 0.6mm difference
    }
)

# Example 3: Grooves for decorative inlay
gen = OrnamentGenerator(
    func=lambda z: np.sin(z),
    domain=Disk(3),
    texture_mode='phase_grooves',
    texture_params={
        'n_phase_grooves': 8,
        'groove_depth': 0.8,
        'groove_width': 0.3
    }
)

# Example 4: Combined texture
gen = OrnamentGenerator(
    func=lambda z: (z - 1) / (z + 1),
    texture_mode='combined',
    texture_params={
        'features': ['phase_contours', 'modulus_contours'],
        'phase_weight': 0.7,
        'modulus_weight': 0.3,
        'n_phase_contours': 12,
        'n_modulus_rings': 6
    }
)
```

## Implementation Priority

1. **Phase 1**: Basic contour ridges
   - Implement `phase_contours` mode with sharp ridges
   - Add mesh quality safeguards
   - Test with simple rational functions

2. **Phase 2**: Additional patterns
   - Add `chessboard` mode (both cartesian and polar)
   - Implement `modulus_contours`
   - Add groove option

3. **Phase 3**: Advanced features
   - Combined textures
   - Adaptive ridge width based on local curvature
   - Preview functionality with texture visualization

## Testing Strategy

1. **Visual Testing**:
   - Generate low-res previews with exaggerated displacement
   - Verify contours align with colormap patterns
   - Check for mesh artifacts

2. **Printability Testing**:
   - Ensure minimum feature size > 0.2mm (FDM limit)
   - Check for self-intersections
   - Validate watertight property is maintained

3. **Tactile Testing**:
   - Print samples with different parameters
   - Verify ridges/grooves are easily felt
   - Test durability of thin features

## Notes

- Sharp textures work better than smooth on curved surfaces
- Binary (on/off) patterns are most reliable for 3D printing
- Ridge height should be proportional to ornament size
- Consider printer resolution when setting feature sizes