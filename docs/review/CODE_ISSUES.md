# Code Issues - Complexplorer Project Review

## Overview
This document catalogs technical implementation problems found in the Complexplorer codebase, including hard-coded values, code duplication, backwards compatibility shims, and logical inconsistencies.

---

## 1. Backwards Compatibility Shims (PRIORITY: HIGH - REMOVE THESE)

### 1.1 `quick_plot` Alias
**Location:** `complexplorer/__init__.py:71`

```python
plot as quick_plot,  # Keep backward compat name for now
```

**Issue:** Backwards compatibility alias serving no purpose with small user base.

**Action:**
- ❌ **REMOVE** this alias entirely
- Update any internal uses to `plot`
- Breaking change is acceptable

**Effort:** Low (5 minutes)

---

### 1.2 `stereographic` Alias
**Location:** `complexplorer/core/functions.py:326-327`

```python
# Backward compatibility
stereographic = stereographic_projection
```

**Issue:** Legacy alias for renamed function.

**Action:**
- ❌ **REMOVE** this alias
- Update all imports to use `stereographic_projection`
- Add note in migration guide

**Effort:** Low (10 minutes, includes search/replace)

**Files to update:**
```bash
grep -r "import.*stereographic[^_]" complexplorer/
grep -r "from.*stereographic[^_]" complexplorer/
```

---

### 1.3 Legacy Mesh API
**Location:** `complexplorer/utils/mesh.py:237-240`

```python
# Legacy API compatibility
# ... legacy wrappers
```

**Issue:** Maintains old API patterns.

**Action:**
- ❌ **REMOVE** entire legacy compatibility section
- Update any code still using old patterns
- Document breaking changes

**Effort:** Medium (1-2 hours including testing)

---

## 2. Hard-Coded Magic Numbers

### 2.1 Colormap Hard-Coded Values

#### Phase Colormap Defaults
**Location:** `complexplorer/core/colormap.py` (multiple instances)

```python
# Line ~145
v_base=0.6  # Hard-coded default brightness
v_mult=0.3  # Hard-coded contrast

# Line ~200
default_scale_radius = 0.8  # Magic number for auto-scaling
```

**Issue:** These values aren't documented or explained. Why 0.6? Why 0.8?

**Action:**
- Define as module-level constants with clear names:
  ```python
  # At top of file
  DEFAULT_BASE_BRIGHTNESS = 0.6  # Base brightness for phase portraits
  DEFAULT_BRIGHTNESS_CONTRAST = 0.3  # Contrast factor for modulus variations
  DEFAULT_AUTO_SCALE_RADIUS = 0.8  # Scale factor for square cell auto-sizing
  ```
- Add docstring explaining the rationale
- Make these easily configurable

**Effort:** Low (30 minutes)

---

#### Stereographic Projection Tolerance
**Location:** `complexplorer/core/functions.py:417`

```python
tol = 1e-8  # Hard-coded tolerance
psi_axis = np.linspace(tol, np.pi - tol, resolution)
```

**Issue:** Hard-coded tolerance without explanation.

**Action:**
- Define constant:
  ```python
  STEREOGRAPHIC_POLE_TOLERANCE = 1e-8  # Avoid division by zero at poles
  ```
- Consider making it a parameter for extreme precision needs

**Effort:** Low (15 minutes)

---

### 2.2 Validation Hard-Coded Limits

#### Resolution Limits
**Location:** `complexplorer/utils/validation.py:198`

```python
def validate_resolution(resolution: Any,
                       param_name: str = 'resolution',
                       min_val: int = 10,     # Hard-coded
                       max_val: int = 1000):  # Hard-coded
```

**Issue:** These limits are arbitrary and embedded in function signature.

**Action:**
- Define module constants:
  ```python
  # Rational limits for performance vs quality
  MIN_RESOLUTION = 10    # Below this, artifacts are severe
  MAX_RESOLUTION = 1000  # Above this, memory/time issues on most systems
  RECOMMENDED_2D_RESOLUTION = 500
  RECOMMENDED_3D_RESOLUTION = 100
  ```

**Effort:** Low (20 minutes)

---

#### STL Size Limits
**Location:** `complexplorer/utils/validation.py:150-152`

```python
def validate_stl_parameters(size_mm: float,
                          wall_thickness: float,
                          min_size: float = 1.0,    # Magic
                          max_size: float = 500.0): # Magic
```

**Issue:** Embedded in function - hard to change, undocumented why these values.

**Action:**
```python
# Physical printing constraints
MIN_PRINTABLE_SIZE_MM = 1.0    # Below this, most printers struggle
MAX_PRINTABLE_SIZE_MM = 500.0  # Above this, exceeds most build volumes
MIN_WALL_THICKNESS_MM = 0.8    # Typical minimum for FDM printers
```

**Effort:** Low (20 minutes)

---

### 2.3 PyVista Hard-Coded Parameters

#### Lighting and Material Constants
**Location:** `complexplorer/plotting/pyvista/plot_3d.py:289-295`

```python
specular=0.5,         # Hard-coded
specular_power=15,    # Hard-coded
diffuse=0.7,          # Hard-coded
ambient=0.3,          # Hard-coded
```

**Issue:** These affect visual quality significantly but aren't configurable.

**Action:**
```python
# At module level - PBR-like material defaults
DEFAULT_MATERIAL_PARAMS = {
    'specular': 0.5,       # Specular reflection strength
    'specular_power': 15,  # Specular highlight sharpness (Phong exponent)
    'diffuse': 0.7,        # Diffuse reflection (matte appearance)
    'ambient': 0.3,        # Ambient light contribution
}
```
- Make these accessible as function parameters with sensible defaults
- Add preset configurations ('matte', 'glossy', 'metallic')

**Effort:** Medium (1 hour)

---

### 2.4 Scaling Preset Hard-Coded Values

**Location:** `complexplorer/core/scaling.py:321-347`

```python
SCALING_PRESETS = {
    'balanced': {
        'method': 'sigmoid',
        'params': {'steepness': 2.0, 'center': 1.0, 'r_min': 0.2, 'r_max': 1.0},
        # Why these exact values? ^^^^^
    },
    # ... more presets
}
```

**Issue:** Magic numbers in preset definitions without justification.

**Action:**
- Add comprehensive comments explaining each value choice
- Create a "tuning guide" documenting how these were selected
- Consider allowing easy user preset definition

**Effort:** Medium (2 hours for documentation)

---

## 3. Code Duplication

### 3.1 Massive Colormap Duplication

**Locations:**
- `complexplorer/core/colormap.py:Phase` class
- `complexplorer/core/colormap.py:Chessboard` class
- `complexplorer/core/colormap.py:PolarChessboard` class
- `complexplorer/core/colormap.py:LogRings` class
- And more...

**Issue:** Each colormap class reimplements nearly identical:
- HSV to RGB conversion logic (~50 lines each)
- Input validation (~30 lines each)
- Mask handling (~20 lines each)
- Parameter handling (~40 lines each)

**Total duplication:** ~800+ lines across 6-8 classes

**Example Duplication:**
```python
# In Phase class (lines 200-220)
if outmask is not None:
    # Ensure mask is boolean array
    outmask = np.asarray(outmask, dtype=bool)
    if outmask.shape != H.shape:
        raise ValueError(...)
    # Apply mask to HSV
    H[outmask] = 0
    S[outmask] = 0
    V[outmask] = 0.5

# IDENTICAL code in Chessboard class (lines 450-470)
if outmask is not None:
    outmask = np.asarray(outmask, dtype=bool)
    if outmask.shape != H.shape:
        raise ValueError(...)
    H[outmask] = 0
    S[outmask] = 0
    V[outmask] = 0.5
```

**Action:**
1. Extract common functionality to base class methods:
   ```python
   class Colormap:
       def _apply_mask(self, H, S, V, outmask):
           """Apply outmask to HSV components."""
           # Single implementation

       def _validate_complex_input(self, z):
           """Validate and normalize complex input."""
           # Single implementation

       def _hsv_to_rgb_standard(self, H, S, V):
           """Standard HSV to RGB conversion."""
           # Single implementation
   ```

2. Refactor each colormap to use base methods:
   ```python
   class Phase(Colormap):
       def hsv(self, z, outmask=None):
           z = self._validate_complex_input(z)
           H, S, V = self._compute_phase_hsv(z)  # Unique part
           return self._apply_mask(H, S, V, outmask)
   ```

3. Remove 600+ lines of duplication

**Effort:** High (4-6 hours, requires careful testing)
**Priority:** HIGH - This is the single biggest code smell

---

### 3.2 Plot Function Duplication

**Locations:**
- `complexplorer/plotting/matplotlib/plot_2d.py:plot()`
- `complexplorer/plotting/matplotlib/plot_2d.py:pair_plot()`
- `complexplorer/plotting/matplotlib/plot_3d.py:plot_landscape()`
- `complexplorer/plotting/matplotlib/plot_3d.py:pair_plot_landscape()`
- PyVista equivalents

**Issue:** Nearly identical input validation and mesh generation logic duplicated across functions.

**Example:**
```python
# Repeated in 8+ functions:
if domain is None and z is None:
    raise ValidationError("Either domain or z must be provided")
if f is None and func is None:
    raise ValidationError("Either f or func must be provided")
if z is None:
    z = domain.mesh(resolution)
    mask = domain.outmask(resolution)
else:
    mask = None
if f is None:
    f = func(z)
f = np.asarray(f)
if f.ndim == 0:
    f = np.full_like(z, f)
```

**Action:**
- Create utility function in plotting utils:
  ```python
  def prepare_mesh_and_values(domain, func, z, f, resolution):
      """Prepare mesh and function values for plotting."""
      # Single implementation
      return z, f, mask
  ```
- Replace all duplicated code with single function call

**Effort:** Medium (2-3 hours)
**Priority:** MEDIUM-HIGH

---

### 3.3 Riemann Sphere Mesh Generation Duplication

**Locations:**
- `complexplorer/plotting/matplotlib/plot_3d.py:riemann()` lines 416-428
- `complexplorer/utils/mesh.py:RectangularSphereGenerator`
- Possibly more instances

**Issue:** Sphere mesh generation logic exists in multiple places.

**Action:**
- Consolidate into single authoritative implementation
- All other code should call the canonical version

**Effort:** Medium (2 hours)

---

## 4. Inconsistent Parameter Naming

### 4.1 Resolution Parameter Chaos

**Different names for same concept:**
| Location | Parameter Name | Meaning |
|----------|---------------|---------|
| `plot_2d.py` | `resolution` | Points along longest edge |
| `plot_3d.py` | `resolution` | Same |
| Some functions | `n` | Same thing! |
| Riemann functions | `n_theta`, `n_phi` | Two different resolutions |
| Old code | `N` | Capital N! |

**Action:**
- **STANDARDIZE** on `resolution` everywhere
- For sphere: keep explicit `n_theta` and `n_phi` where needed
- Remove all uses of `n` and `N`

**Effort:** Medium (1-2 hours, global search/replace)
**Priority:** MEDIUM

---

### 4.2 Phase Sectors Naming

**Current:** `n_phi` (confusing - sounds like resolution)
**Better:** `phase_sectors` or `num_phase_sectors`

**Locations:**
- `complexplorer/core/colormap.py` - All Phase-based colormaps
- Examples and documentation

**Action:**
```python
# Old
cmap = Phase(n_phi=6)

# New
cmap = Phase(phase_sectors=6)
```

**Effort:** Medium (2 hours including docs)
**Priority:** MEDIUM - improves clarity significantly

---

### 4.3 Modulus Mode vs Scaling Method

**Current chaos:**
| Location | Parameter Name |
|----------|---------------|
| `plot_3d.py` | `modulus_mode` |
| `scaling.py` | `method` |
| Some places | `scaling` |

**Action:**
- **STANDARDIZE** on `modulus_scaling` everywhere
- Rename `modulus_mode` → `modulus_scaling`
- Update all docs

**Effort:** Low-Medium (1 hour)

---

## 5. Logical Errors & Bugs

### 5.1 Inconsistent `return_ax` Logic

**Location:** `complexplorer/plotting/matplotlib/plot_3d.py:268-287`

```python
return_ax = ax is not None
if ax is None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ...

if return_ax:
    return ax
else:
    return plt.gca()  # BUG: This returns ax anyway!
```

**Issue:** The logic is confused. If we created the axes, we return `plt.gca()` which is the same axes we just created. This accomplishes nothing.

**Fix:**
```python
# Option 1: Always return axes
return ax

# Option 2: Return None if we showed the plot
if ax is None:
    plt.show()
    return None
return ax
```

**Effort:** Low (30 minutes + testing)
**Priority:** LOW (doesn't break anything, just confusing)

---

### 5.2 Mask Application Inconsistency

**Location:** Multiple colormap files

**Issue:** Some colormaps set masked regions to:
- `H=0, S=0, V=0.5` (gray)
- `H=0, S=0, V=0` (black)
- Leave NaN values

**Action:**
- **STANDARDIZE** mask behavior across all colormaps
- Document what masked regions should look like
- Consider making it configurable: `mask_color='gray'` parameter

**Effort:** Medium (2 hours)
**Priority:** MEDIUM

---

### 5.3 OkLCh Gamut Clipping Loses Information

**Location:** `complexplorer/core/color_utils.py:72-75`

```python
# Clip to valid range
R = np.clip(R, 0, 1)
G = np.clip(G, 0, 1)
B = np.clip(B, 0, 1)
```

**Issue:** Simple clipping can cause hue shifts for out-of-gamut colors. There's a `clip_to_gamut()` function in the same file that does this properly!

**Fix:**
```python
# Use the proper gamut clipping function
R, G, B = clip_to_gamut(R, G, B, preserve='hue')
```

**Effort:** Low (15 minutes)
**Priority:** MEDIUM - affects color accuracy

---

## 6. Dead/Unused Code

### 6.1 Unused PlotConfig Class

**Location:** `complexplorer/plotting/base.py` (if it exists)

**Issue:** Base class architecture is partially implemented but never used.

**Evidence:**
```python
# plot_2d.py:96
config = None) -> Figure:  # config: Optional[PlotConfig] = None
```
Type hint commented out, parameter never used.

**Action:**
- ❌ **DELETE** entire `base.py` if it exists
- Remove all `config` parameters from function signatures
- Remove commented-out type hints

**Effort:** Low-Medium (1 hour)
**Priority:** HIGH - reduces confusion

---

### 6.2 Incomplete Preset Functions

**Location:** `complexplorer/api.py` (lines mentioned in TASKS_V2.md)

```python
def publication_ready():
    # TODO: Implement
    pass

def interactive():
    # TODO: Implement
    pass

def high_contrast():
    # TODO: Implement
    pass
```

**Action:**
- Either **IMPLEMENT** these properly OR
- ❌ **DELETE** them entirely

**Effort:**
- Delete: Low (30 minutes)
- Implement: High (4-6 hours)

**Priority:** MEDIUM - document decision

---

## 7. Performance Issues

### 7.1 Redundant Array Copies

**Example:** `complexplorer/core/colormap.py` multiple locations

```python
z = np.asarray(z)  # Copy 1
# ... later ...
z = np.asarray(z)  # Copy 2 (redundant!)
```

**Action:**
- Audit all array creation
- Use views where possible: `z.ravel()` instead of `z.flatten()`
- Only copy when necessary

**Effort:** Medium (3 hours for full audit)
**Priority:** LOW (only matters for huge datasets)

---

### 7.2 Inefficient Riemann Sphere Mesh

**Location:** Multiple files, documented in CLAUDE.md

> The Riemann sphere plot uses a rectangular mesh which is inefficient at poles

**Issue:** Rectangular (theta, phi) grid creates many tiny triangles near poles.

**Current status:** Known issue, marked for future improvement

**Action:**
- Document this limitation clearly
- Add issue to track improvement
- Consider icosphere-based approach for v3.0

**Effort:** High (major refactor)
**Priority:** LOW (works, just not optimal)

---

## 8. Missing Validation

### 8.1 No Validation in Colormap Constructors

**Example:** `complexplorer/core/colormap.py:Phase.__init__`

```python
def __init__(self, n_phi=20, scale_radius=None, ...):
    self.n_phi = n_phi  # No validation!
    # What if n_phi = -5?
    # What if n_phi = 0? (division by zero later!)
```

**Action:**
- Add validation in all __init__ methods:
  ```python
  if n_phi < 1:
      raise ValidationError("phase_sectors must be positive")
  if not isinstance(n_phi, int):
      raise ValidationError("phase_sectors must be integer")
  ```

**Effort:** Medium (2 hours across all classes)
**Priority:** MEDIUM-HIGH

---

### 8.2 No Filename Validation Before STL Export

**Location:** Export functions

**Issue:** Functions accept any filename without checking:
- Directory exists
- Write permissions
- Valid extension
- Disk space

**Action:**
- Use `validate_file_extension()` from validation module
- Add directory existence check
- Add helpful error messages

**Effort:** Low (1 hour)
**Priority:** MEDIUM

---

## 9. Type Hinting Issues

### 9.1 Missing Return Types

**From TASKS_V2.md:** 23 functions missing return types

**Examples:**
```python
# complexplorer/api.py
def quick_plot(...):  # Missing -> Optional[Axes]
def analyze_function(...):  # Missing -> Dict[str, Any]
def create_animation(...):  # Missing -> Any
```

**Action:**
- Add return type hints to all 23 functions
- Add `from __future__ import annotations` for forward references
- Create `py.typed` marker file

**Effort:** Medium (2-3 hours)
**Priority:** MEDIUM - helps IDE users

---

### 9.2 Inconsistent Optional Usage

**Issue:** Some functions use `Optional[T]` while others use `T | None` (Python 3.10+)

**Action:**
- Pick ONE style and stick to it
- Recommendation: `Optional[T]` for Python 3.9 compatibility
- Or require Python 3.10+ and use `T | None`

**Effort:** Low (1 hour)

---

## Summary Statistics

| Category | Count | Priority |
|----------|-------|----------|
| Backwards Compatibility Shims | 3 | **HIGH** - Remove all |
| Hard-Coded Magic Numbers | 15+ | MEDIUM |
| Code Duplication (lines) | 800+ | **HIGH** |
| Parameter Naming Issues | 8+ | MEDIUM |
| Logical Errors | 3 | MEDIUM |
| Dead Code Sections | 5+ | HIGH |
| Performance Issues | 2 | LOW |
| Missing Validation | 10+ | MEDIUM-HIGH |
| Type Hint Issues | 23+ | MEDIUM |

## Total Estimated Effort
- **Quick Wins** (< 1 hour each): ~15 items = ~8 hours
- **Medium Tasks** (1-3 hours): ~12 items = ~24 hours
- **Large Tasks** (3+ hours): ~5 items = ~20 hours
- **Total**: ~52 hours

## Recommended Order
1. ✅ Remove backwards compatibility shims (breaking changes OK)
2. ✅ Fix colormap duplication (biggest code smell)
3. ✅ Delete unused code
4. ✅ Standardize parameter naming
5. ✅ Add missing validation
6. ⚠️ Fix hard-coded values
7. ⚠️ Address type hints
8. ⏸️ Performance optimizations (nice-to-have)

---

**Review Date:** 2025-10-14
**Reviewer:** Claude Code Assistant
**Next Review:** After refactoring Phase 1
