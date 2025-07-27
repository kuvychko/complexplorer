# Physical Texture Implementation Checklist

## Phase 1: Core Infrastructure

### 1.1 Update riemann_pv Function Signature
- [ ] Add texture parameters to `riemann_pv` in `complexplorer/plotting/pyvista/riemann.py`
  - [ ] `texture_height: float = 0.0` (mm)
  - [ ] `texture_mode: str = 'ridges'` (ridges/grooves/binary)
  - [ ] `texture_sharpness: float = 1.0` (0-1)
  - [ ] `texture_preview_scale: float = 5.0`
- [ ] Update docstring with texture parameter descriptions
- [ ] Add texture parameters to kwargs filtering

### 1.2 Create Texture Computation Module
- [ ] Create `complexplorer/utils/texture.py`
- [ ] Implement `compute_texture_from_colormap()` function
  - [ ] Handle binary colormaps (Chessboard, PolarChessboard, LogRings)
  - [ ] Handle continuous colormaps (Phase)
  - [ ] Return displacement array matching mesh points

### 1.3 Implement Gradient Computation
- [ ] Add `compute_sphere_gradient()` in `texture.py`
  - [ ] Handle spherical mesh topology
  - [ ] Account for pole singularities
  - [ ] Handle wraparound at seams
- [ ] Add unit tests for gradient computation

## Phase 2: Mesh Generation Updates

### 2.1 Modify Sphere Mesh Generation
- [ ] Update `_generate_sphere_mesh()` to track mesh structure
  - [ ] Store theta/phi grid dimensions
  - [ ] Add mesh metadata for gradient computation
- [ ] Ensure consistent point ordering for gradient calculation

### 2.2 Implement Proper Order of Operations
- [ ] In riemann_pv mesh generation:
  1. [ ] Generate base sphere mesh
  2. [ ] Compute f_values on original sphere
  3. [ ] Apply modulus scaling (if not 'constant')
  4. [ ] Recompute normals after scaling
  5. [ ] Apply texture displacement
  6. [ ] Store texture metadata in mesh

### 2.3 Normal Computation
- [ ] Ensure `mesh.compute_normals()` is called after modulus scaling
- [ ] Verify normals point outward consistently
- [ ] Handle normal computation at poles correctly

## Phase 3: Texture Algorithms

### 3.1 Binary Colormap Textures
- [ ] Implement for Chessboard:
  - [ ] Binary mode: direct value mapping
  - [ ] Ridge mode: edge detection
  - [ ] Groove mode: inverted edge detection
- [ ] Implement for PolarChessboard (same modes)
- [ ] Implement for LogRings (same modes)
- [ ] Add tests with known patterns

### 3.2 Continuous Colormap Textures
- [ ] Implement for Phase colormap:
  - [ ] HSV gradient computation
  - [ ] Perceptual weighting (H > V > S)
  - [ ] Handle hue wraparound
  - [ ] Threshold based on sharpness parameter
- [ ] Test with enhanced phase portraits
- [ ] Verify ridges appear at sector boundaries

### 3.3 Edge Detection Refinements
- [ ] Implement adaptive thresholding
- [ ] Add edge thinning for cleaner ridges
- [ ] Handle gradient magnitude normalization
- [ ] Test at various mesh resolutions

## Phase 4: Preview Integration

### 4.1 Update PyVista Display
- [ ] Add texture info text overlay:
  - [ ] Preview scale indicator
  - [ ] Actual texture height
  - [ ] Mode (ridges/grooves/binary)
- [ ] Optional wireframe overlay to show deformation
- [ ] Ensure colors still display correctly

### 4.2 Mesh Metadata
- [ ] Store unscaled displacement in mesh['texture_displacement']
- [ ] Store texture parameters in mesh attributes
- [ ] Ensure metadata survives mesh operations

## Phase 5: STL Export Integration

### 5.1 Update OrnamentGenerator
- [ ] Add texture parameters matching riemann_pv
- [ ] Add method to extract texture from preview mesh
- [ ] Ensure consistent texture application

### 5.2 Texture Scaling
- [ ] Remove preview scale for actual export
- [ ] Validate texture height is reasonable (0.1-2.0mm)
- [ ] Add warnings for excessive texture height

### 5.3 Mesh Quality Checks
- [ ] Check for self-intersections after texture
- [ ] Verify watertight property maintained
- [ ] Add option to locally smooth problem areas

## Phase 6: Testing

### 6.1 Unit Tests
- [ ] Test gradient computation on known patterns
- [ ] Test texture generation for each colormap type
- [ ] Test order of operations (scaling then texture)
- [ ] Test edge cases (poles, domain boundaries)

### 6.2 Visual Tests
- [ ] Create gallery of textured previews
- [ ] Compare with corresponding colormaps
- [ ] Verify texture aligns with color boundaries
- [ ] Test with various functions and domains

### 6.3 Export Tests
- [ ] Generate test STL files
- [ ] Verify in STL viewer (MeshLab, etc.)
- [ ] Check file size and complexity
- [ ] Validate printability metrics

## Phase 7: Documentation and Examples

### 7.1 Update Documentation
- [ ] Add texture section to STL export guide
- [ ] Update riemann_pv docstring
- [ ] Create texture parameter guide
- [ ] Add to API reference

### 7.2 Create Examples
- [ ] Basic phase portrait with ridges
- [ ] Chessboard with binary height
- [ ] Polar pattern with grooves
- [ ] Comparison notebook showing all modes
- [ ] Gallery of textured ornaments

### 7.3 Best Practices Guide
- [ ] Recommended texture heights by ornament size
- [ ] Colormap selection for best tactile effect
- [ ] Printer-specific recommendations
- [ ] Post-processing tips

## Phase 8: Performance and Polish

### 8.1 Optimization
- [ ] Profile texture computation
- [ ] Optimize gradient calculation
- [ ] Consider caching for repeated previews
- [ ] Parallelize where beneficial

### 8.2 User Experience
- [ ] Add input validation for texture parameters
- [ ] Provide helpful error messages
- [ ] Add progress indicator for large meshes
- [ ] Consider interactive texture adjustment

### 8.3 Advanced Features (Future)
- [ ] Multiple texture layers
- [ ] Adaptive texture based on local curvature
- [ ] Custom texture patterns
- [ ] Texture preview in matplotlib (2D projection)

## Testing Checklist

### Function Coverage
- [ ] `f(z) = z` (identity, simple test)
- [ ] `f(z) = (z^2 - 1)/(z^2 + 1)` (poles and zeros)
- [ ] `f(z) = exp(z)` (essential singularity)
- [ ] `f(z) = z^n - 1` (multiple poles)

### Colormap Coverage
- [ ] Phase (basic) → ridges
- [ ] Phase (enhanced) → ridges at sectors
- [ ] Chessboard → binary height
- [ ] PolarChessboard → grooves
- [ ] LogRings → ridges

### Scaling Coverage
- [ ] Constant scaling + texture
- [ ] Arctan scaling + texture
- [ ] Log scaling + texture
- [ ] Linear scaling + texture

### Domain Coverage
- [ ] Full sphere (no domain)
- [ ] Disk domain
- [ ] Annulus domain
- [ ] Composite domains

## Success Criteria

1. **Visual Verification**: Texture ridges/grooves appear exactly where colors change
2. **Consistent Height**: 0.8mm ridge is 0.8mm everywhere on the ornament
3. **Print Quality**: Generated STLs are printable without supports
4. **Performance**: Texture adds <10% to preview generation time
5. **User Experience**: Clear, intuitive parameters with good defaults