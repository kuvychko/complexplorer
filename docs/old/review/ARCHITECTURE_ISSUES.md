# Architecture Issues - Complexplorer Project Review

## Overview
This document examines structural and architectural problems in the Complexplorer codebase, including excessive abstraction, inconsistent API patterns, module organization issues, and design decisions that hinder maintainability.

---

## 1. Excessive Abstraction

### 1.1 Unused Base Class Architecture

**Locations:**
- `complexplorer/plotting/base.py` (mentioned in TASKS_V2.md but possibly deleted)
- `complexplorer/plotting/matplotlib/plot_2d.py:96`
- `complexplorer/plotting/matplotlib/plot_3d.py:36`

**Issue:** Evidence of planned base class hierarchy that was never completed:

```python
# plot_2d.py:96
config = None) -> Figure:  # config: Optional[PlotConfig] = None

# plot_3d.py:36
config = None) -> Axes3D:  # config: Optional[PlotConfig] = None
```

Commented-out type hints suggest PlotConfig was intended but abandoned.

**From TASKS_V2.md:**
```markdown
### Base Class Architecture
- [ ] Decision: Keep or remove base classes in `plotting/base.py`
- [ ] If keeping: Implement inheritance for `Matplotlib2DPlotter`
```

**Analysis:**
- ❌ **Abandoned abstraction is worse than no abstraction**
- Adds cognitive load ("What was PlotConfig supposed to do?")
- Creates false expectations
- Half-implemented design is confusing

**Action:**
1. ❌ **DELETE** `plotting/base.py` if it exists
2. ❌ **REMOVE** all `config` parameters from plot functions
3. ❌ **REMOVE** all commented-out `PlotConfig` type hints
4. **DOCUMENT** decision: "Flat is better than nested" - functions over class hierarchy

**Rationale:**
- Current functional approach works well
- Adding inheritance would complicate without clear benefit
- If config is needed later, use simple dict or dataclass

**Effort:** Low-Medium (1-2 hours)
**Priority:** HIGH - remove cruft

---

### 1.2 Over-Engineered Colormap Base Class

**Location:** `complexplorer/core/colormap.py`

**Current Structure:**
```python
class Colormap:
    """Abstract base with required methods"""
    def hsv(self, z, outmask=None):
        raise NotImplementedError

    def rgb(self, z, outmask=None):
        # Calls self.hsv() then converts
        # Duplicated in every subclass anyway!
```

**Issue:**
- Base class defines `rgb()` method
- But every subclass **reimplements** it anyway
- Base class provides no actual shared code
- Just enforces interface (which type hints could do better)

**Better Design:**

**Option A: Use Protocol (Python 3.8+)**
```python
from typing import Protocol

class ColormapProtocol(Protocol):
    """Interface definition without inheritance"""
    def hsv(self, z, outmask=None) -> Tuple[np.ndarray, ...]:
        ...
    def rgb(self, z, outmask=None) -> np.ndarray:
        ...
```

**Option B: Proper Shared Implementation**
```python
class Colormap:
    """Base class with ACTUAL shared functionality"""

    def _apply_mask(self, H, S, V, outmask):
        """Shared mask application logic"""
        # Single implementation used by all

    def _hsv_to_rgb(self, H, S, V):
        """Shared HSV conversion logic"""
        # Single implementation

    def rgb(self, z, outmask=None):
        """Final implementation - don't override"""
        H, S, V = self.hsv(z, outmask)
        return self._hsv_to_rgb(H, S, V)
```

**Action:**
- Choose Option B (provides real value)
- Move all shared logic to base class
- Make `rgb()` final (document "don't override")
- Reduces code by ~600 lines (see CODE_ISSUES.md section 3.1)

**Effort:** High (4-6 hours)
**Priority:** HIGH - solves code duplication

---

## 2. API Inconsistency

### 2.1 The z/f vs domain/func Parameter Mess

**Problem:** Every plot function accepts FOUR ways to specify the same thing:

```python
def plot(domain=None, func=None, z=None, f=None, ...):
    """Which parameters do I use?!"""
```

**Current Combinations:**
1. `plot(domain=rect, func=lambda z: z**2)`  - High level, clean
2. `plot(z=mesh, f=values)`  - Pre-computed, advanced
3. `plot(domain=rect, f=values)`  - Mixed level (weird!)
4. `plot(z=mesh, func=lambda z: z**2)`  - Mixed level (weird!)

**Issues:**
- **16 possible combinations** of which parameters to provide
- Validation complexity in every function
- User confusion: "Which style should I use?"
- Code duplication handling all cases

**Example from `plot_2d.py:190-195`:**
```python
if domain is None and z is None:
    raise ValidationError("Either domain or z must be provided")
if f is None and func is None:
    raise ValidationError("Either f or func must be provided")
# ... this is in EVERY plot function
```

**Better Design:**

**Option A: Simplify to Two Clear Functions**
```python
def plot(domain: Domain, func: Callable, resolution: int = 500, ...):
    """High-level API: provide domain and function"""
    z = domain.mesh(resolution)
    f = func(z)
    return plot_arrays(z, f, ...)

def plot_arrays(z: np.ndarray, f: np.ndarray, ...):
    """Low-level API: provide pre-computed arrays"""
    # Direct plotting
```

**Benefits:**
- Clear separation of concerns
- Each function does ONE thing
- No validation spaghetti
- Easy to understand and document

**Option B: Use Overloads (Type-Checked)**
```python
@overload
def plot(domain: Domain, func: Callable, ...) -> Axes: ...

@overload
def plot(*, z: np.ndarray, f: np.ndarray, ...) -> Axes: ...

def plot(domain=None, func=None, z=None, f=None, ...):
    if domain is not None:
        # High-level path
    else:
        # Low-level path
```

**Recommendation:** Option A
- Simpler implementation
- Clearer to users
- Better documentation
- Easier testing

**Migration:**
```python
# Old code
plot(domain=rect, func=f)  # Still works!

# Old code (low-level)
plot(z=mesh, f=values)  # Now: plot_arrays(mesh, values)

# Update: Easy to find with grep
```

**Effort:** High (8-10 hours including tests and docs)
**Priority:** HIGH - core API design
**Breaking Change:** Yes (but small user base, acceptable)

---

### 2.2 Inconsistent Colormap Parameter

**Current Chaos:**

| Function | Parameter Name | Type |
|----------|---------------|------|
| `plot()` | `cmap` | `Colormap` |
| `Matplotlib2DPlotter.plot_single()` | `colormap` | `Colormap` |
| `plot_landscape()` | `cmap` | `Colormap` |
| Some internal | `color_map` | `Colormap` |

**Issue:** Same concept, three different names!

**Action:**
- **STANDARDIZE** on `cmap` (shorter, matches matplotlib convention)
- Or `colormap` (more explicit)
- Pick ONE and use EVERYWHERE

**Effort:** Low (1 hour, search/replace)
**Priority:** MEDIUM

**Recommendation:** `cmap`
- Shorter
- Matches matplotlib (`plt.imshow(..., cmap='viridis')`)
- Common in scientific Python

---

### 2.3 Resolution Parameter Chaos

See CODE_ISSUES.md Section 4.1 - architectural impact:

**Current:** `resolution`, `n`, `N`, `n_theta`, `n_phi` all mean different things (or the same thing!)

**Architectural Decision Needed:**

**For 2D meshes:**
- Use `resolution` = points along longest edge
- Let domain calculate actual grid size

**For Riemann sphere:**
- Use explicit `n_theta` and `n_phi`
- Or: `resolution` = both, `aspect_ratio` = theta/phi ratio

**Action:**
- Define resolution strategy document
- Implement consistently everywhere
- Update all docs

**Effort:** Medium (3-4 hours)
**Priority:** MEDIUM-HIGH

---

## 3. Module Organization

### 3.1 Utils Package Is a Junk Drawer

**Location:** `complexplorer/utils/`

**Current Contents:**
```
utils/
├── validation.py       # Input validation
├── mesh.py            # Mesh generation
├── mesh_distortion.py # Riemann sphere scaling
├── color_utils.py     # Color space conversion
└── ???                # What else is in here?
```

**Issues:**
- `mesh.py` and `mesh_distortion.py` - aren't these related?
- `color_utils.py` - shouldn't this be in `core/`?
- `validation.py` - used by everyone, should be more prominent

**Better Organization:**

**Option A: Flatten into core**
```
core/
├── domain.py
├── colormap.py
├── color_spaces.py    # was utils/color_utils.py
├── functions.py
├── scaling.py
├── mesh.py            # merge mesh + mesh_distortion
└── validation.py      # promote from utils
```

**Option B: Organize by purpose**
```
core/              # Core data structures
├── domain.py
├── colormap.py
└── validation.py

geometry/          # Geometric computations
├── mesh.py
└── riemann.py     # Riemann sphere specific

color/             # Color handling
├── spaces.py
└── utils.py
```

**Recommendation:** Option A (simpler)
- Fewer directories
- Everything important in `core/`
- Clear import paths: `from complexplorer.core import mesh`

**Effort:** Low-Medium (2-3 hours)
**Priority:** MEDIUM

---

### 3.2 Plotting Subpackage Structure

**Current:**
```
plotting/
├── base.py              # Unused!
├── matplotlib/
│   ├── plot_2d.py
│   └── plot_3d.py
└── pyvista/
    ├── plot_3d.py
    └── utils.py
```

**Issues:**
- `base.py` - delete it (see 1.1)
- Too deep: `from complexplorer.plotting.matplotlib.plot_2d import plot`

**Better:**
```
plotting/
├── plot_2d.py         # matplotlib 2D (90% of users)
├── plot_3d.py         # matplotlib 3D
└── pyvista/           # keep separate (optional dep)
    ├── landscape.py
    ├── riemann.py
    └── utils.py
```

**Benefits:**
- Shallower imports
- matplotlib is default (no subdir needed)
- PyVista clearly optional

**Effort:** Low (1 hour + git moves)
**Priority:** LOW-MEDIUM

---

### 3.3 Export Module Feels Bolted-On

**Current:**
```
export/
├── stl/
│   ├── ornament_generator.py
│   ├── mesh_repair.py
│   └── utils.py
└── ... (other formats?)
```

**Issues:**
- Only STL export exists
- Feels separate from main package
- Could be better integrated

**Better Integration:**

**Option A: First-class feature**
```
plotting/
└── pyvista/
    ├── landscape.py
    ├── riemann.py
    └── ornament.py    # STL export integrated here
```

**Option B: Keep separate but rename**
```
export_3d/             # Clearer name
├── stl_ornament.py    # Flatten
├── mesh_repair.py
└── utils.py
```

**Recommendation:** Depends on vision
- If STL is core feature → Option A
- If it's experimental/advanced → Option B

**Effort:** Low (1-2 hours)
**Priority:** LOW

---

## 4. Cross-Cutting Concerns

### 4.1 No Consistent Logging Strategy

**Current:**
```python
# Some functions
if verbose:
    print("Doing stuff...")

# Other functions
# Silent, no progress info

# Some functions
import warnings
warnings.warn("This might be bad")
```

**Issues:**
- `print()` statements can't be suppressed
- No structured logging
- Can't redirect to file
- No log levels (DEBUG, INFO, WARNING, ERROR)

**Better Approach:**
```python
import logging

logger = logging.getLogger(__name__)

# In code
logger.debug("Mesh generation details")
logger.info("Generating ornament...")
logger.warning("Values clipped to gamut")
logger.error("Invalid input")
```

**Benefits:**
- Users can configure: `logging.basicConfig(level=logging.WARNING)`
- Can write to files
- Can customize format
- Professional applications expect this

**Action:**
- Add logging throughout
- Keep `verbose` params for backward compat (map to log level)
- Document logging configuration

**Effort:** Medium (4-5 hours)
**Priority:** MEDIUM

---

### 4.2 No Configuration System

**Current:** Every parameter in every function call

```python
plot(domain, func,
     resolution=500,
     cmap=Phase(n_phi=6, auto_scale_r=True, scale_radius=0.8),
     ...)  # Verbose!
```

**Better:** Configuration system

```python
# In user's script
import complexplorer as cp

cp.config.set_defaults(
    resolution=500,
    cmap='phase',  # String key to preset
    phase_sectors=12,
)

# Now
plot(domain, func)  # Uses configured defaults!

# Or override
plot(domain, func, resolution=1000)  # Just this one
```

**Implementation:**
```python
# complexplorer/config.py
class Config:
    def __init__(self):
        self.resolution = 500
        self.default_cmap = 'phase'
        # ...

    def set_defaults(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

config = Config()  # Singleton
```

**Benefits:**
- Less repetition
- Easier to maintain consistent style
- Can save/load presets
- Better notebook experience

**Effort:** Medium-High (5-6 hours)
**Priority:** LOW (nice-to-have for v3.0)

---

## 5. Dependency Management

### 5.1 Optional Dependencies Handling

**Current:**
```python
# In multiple files
try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    pv = None
```

**Issues:**
- Duplicated in every file that uses PyVista
- Error messages vary
- No central check

**Better:**
```python
# complexplorer/utils/deps.py
class OptionalDependency:
    def __init__(self, name, import_name=None):
        self.name = name
        self.import_name = import_name or name
        self._module = None
        self._checked = False
        self._available = False

    def check(self):
        if not self._checked:
            try:
                self._module = __import__(self.import_name)
                self._available = True
            except ImportError:
                self._available = False
            self._checked = True
        return self._available

    def require(self):
        if not self.check():
            raise ImportError(
                f"{self.name} is required for this feature. "
                f"Install with: pip install complexplorer[{self.name.lower()}]"
            )
        return self._module

# Usage
pyvista = OptionalDependency('PyVista', 'pyvista')

# In code
def plot_landscape_pv(...):
    pv = pyvista.require()  # Clean error if not installed
    # ...
```

**Effort:** Low-Medium (2-3 hours)
**Priority:** LOW-MEDIUM

---

## 6. Testing Architecture

### 6.1 Test Organization Mirrors Code

**Current:**
```
tests/
├── unit/
│   ├── core/
│   ├── plotting/
│   └── utils/
└── integration/
```

**Good:** Mirrors src structure
**Missing:**
- Property-based tests
- Performance benchmarks
- Visual regression tests (for plots)

**Additions Needed:**

```
tests/
├── unit/              # Existing
├── integration/       # Existing
├── properties/        # NEW: hypothesis tests
│   └── test_colormap_invariants.py
├── benchmarks/        # NEW: performance tracking
│   └── test_mesh_generation_speed.py
└── visual/            # NEW: plot comparisons
    └── test_plot_regression.py
```

**Example Property Test:**
```python
from hypothesis import given
from hypothesis import strategies as st

@given(st.complex_numbers())
def test_phase_always_in_range(z):
    """Phase should always be in [0, 2π)"""
    from complexplorer.core.functions import phase
    phi = phase(z)
    assert 0 <= phi < 2 * np.pi
```

**Effort:** Medium-High (varies by test type)
**Priority:** MEDIUM

---

## 7. Import Strategy Issues

### 7.1 Circular Import Risk

**Current:** Imports scattered throughout

**Risk Example:**
```python
# domain.py
from complexplorer.core.validation import validate_resolution

# validation.py
from complexplorer.core.colormap import Phase  # Default cmap

# colormap.py
from complexplorer.core.domain import Rectangle  # For examples?
```

**Circular dependency!** (if it exists)

**Prevention:**
1. **Layer architecture**
   ```
   Layer 1: validation, functions (no internal imports)
   Layer 2: domain, colormap (can import Layer 1)
   Layer 3: plotting (can import Layer 1-2)
   ```

2. **Lazy imports in functions**
   ```python
   def validate_colormap(cmap=None):
       if cmap is None:
           from complexplorer.core.colormap import Phase
           cmap = Phase()
       return cmap
   ```

**Action:**
- Audit all imports
- Create import dependency graph
- Enforce layering with tests

**Effort:** Medium (3-4 hours)
**Priority:** MEDIUM

---

### 7.2 Relative vs Absolute Imports

**Current:** Mix of both

```python
# Some files
from complexplorer.core import domain

# Other files
from . import domain

# Even others
from ..core import domain
```

**TASKS_V2.md confirms:**
```markdown
### Import Standardization
- [ ] Convert all relative imports to absolute imports
```

**Action:**
- **STANDARDIZE** on absolute imports everywhere
  ```python
  # Always this
  from complexplorer.core.domain import Rectangle
  ```

- Never use relative imports
- Clearer, less error-prone, works in all contexts

**Effort:** Low-Medium (1-2 hours)
**Priority:** MEDIUM

---

## 8. API Surface

### 8.1 What Should Be Public?

**Current `__init__.py`:**
```python
from complexplorer.plotting.matplotlib.plot_2d import (
    plot,
    plot as quick_plot,  # Duplicate!
    pair_plot,
    riemann_chart,
    # ... too many?
)
```

**Issue:** Exposing too much? Or too little?

**Guideline:** Public API should be:
1. **Small** - Easy to learn
2. **Powerful** - Can do everything needed
3. **Stable** - Doesn't change often

**Recommended Public API:**

**Core User API:**
```python
import complexplorer as cp

# Main functions (80% of users)
cp.plot()
cp.plot_landscape()
cp.riemann()

# Domains
cp.Rectangle()
cp.Disk()
cp.Annulus()

# Colormaps
cp.Phase()
cp.Chessboard()
```

**Advanced API:**
```python
from complexplorer.plotting import pyvista
from complexplorer.export import stl

pyvista.plot_landscape_pv()
stl.create_ornament()
```

**Internal (not public):**
```python
complexplorer.utils.*  # Implementation details
complexplorer.core.color_utils  # Used by colormaps internally
```

**Action:**
1. Define clear public API in docs
2. Mark internal with leading underscore or in `_internal` package
3. Test public API stability

**Effort:** Medium (3-4 hours)
**Priority:** MEDIUM

---

## 9. Error Handling Strategy

### 9.1 Inconsistent Exception Types

**Current:**
```python
# Some places
raise ValueError("Bad input")

# Other places
raise ValidationError("Bad input")

# Some places
raise RuntimeError("Something went wrong")

# Some places
return None  # Silent failure!
```

**Better:** Exception hierarchy

```python
class ComplexplorerError(Exception):
    """Base exception"""

class ValidationError(ComplexplorerError):
    """Input validation failed"""

class ComputationError(ComplexplorerError):
    """Numerical computation failed"""

class ExportError(ComplexplorerError):
    """File export failed"""
```

**Benefits:**
- Users can `except ComplexplorerError` to catch all
- Or catch specific types
- Clear error messages

**Action:**
- Define exception hierarchy
- Use consistently
- Never return None on error (unless documented)

**Effort:** Medium (2-3 hours)
**Priority:** MEDIUM-HIGH

---

## 10. Performance Architecture

### 10.1 No Caching Strategy

**Current:** Recompute everything every time

```python
# Every call recomputes
z = domain.mesh(500)  # Creates 500x500 = 250k points
f = func(z)          # Evaluates function 250k times
```

**If user calls twice with same params → wasted work!**

**Better:** Optional caching

```python
from functools import lru_cache

class Rectangle:
    @lru_cache(maxsize=4)
    def mesh(self, resolution):
        # Cached per (self, resolution)
        # ...
```

**Caution:** Caching is complex
- Memory usage
- Cache invalidation
- Hash-ability of parameters

**Recommendation:**
- Add caching for expensive operations
- Make it opt-in: `cp.config.enable_caching = True`
- Document memory implications

**Effort:** Medium-High (varies)
**Priority:** LOW (premature optimization)

---

## Summary & Recommendations

| Issue | Priority | Effort | Breaking Change? |
|-------|----------|--------|------------------|
| Remove unused base classes | HIGH | Low | No |
| Fix colormap duplication via base class | HIGH | High | Minor |
| Simplify domain/func vs z/f API | HIGH | High | **YES** |
| Standardize parameter names | MEDIUM-HIGH | Medium | Minor |
| Reorganize utils package | MEDIUM | Medium | No |
| Add logging framework | MEDIUM | Medium | No |
| Fix import strategy | MEDIUM | Low-Medium | No |
| Define clear public API | MEDIUM | Medium | Possibly |
| Exception hierarchy | MEDIUM-HIGH | Low-Medium | Minor |

## Architectural Principles for Refactor

1. **Simplicity over Flexibility**
   - Flat is better than nested
   - Explicit is better than implicit
   - Remove unused abstractions

2. **Consistency**
   - One way to do things
   - Same patterns everywhere
   - Predictable behavior

3. **Separation of Concerns**
   - Clear module boundaries
   - Layered dependencies
   - No circular imports

4. **User Experience First**
   - Simple common cases
   - Power when needed
   - Clear error messages

5. **Breaking Changes Are OK**
   - Small user base
   - Clean design more important
   - Provide migration guide

---

**Review Date:** 2025-10-14
**Next Steps:** See REFACTORING_ROADMAP.md for implementation plan
