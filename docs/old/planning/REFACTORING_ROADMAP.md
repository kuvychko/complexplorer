# Refactoring Roadmap - Complexplorer v2.0

## Overview
This document provides a prioritized, actionable plan for addressing all issues identified in the project review. Based on comprehensive analysis in:
- `CODE_ISSUES.md` - Technical implementation problems
- `ARCHITECTURE_ISSUES.md` - Structural and design issues
- `DOCUMENTATION_ISSUES.md` - Documentation gaps and quality

**Key Principle:** Breaking changes are ACCEPTABLE. Small user base means we can prioritize code clarity over backwards compatibility.

---

## Quick Wins First (Total: ~12 hours)

These provide immediate value with minimal effort. Complete these before major refactoring.

### QW-1: Remove Backwards Compatibility Shims (1 hour)
**Files:** `__init__.py`, `core/functions.py`, `utils/mesh.py`

```bash
# Remove these aliases immediately
- __init__.py:71     → Delete "plot as quick_plot" alias
- functions.py:327   → Delete "stereographic = stereographic_projection"
- utils/mesh.py:237+ → Delete entire "Legacy API compatibility" section
```

**Impact:** Reduces confusion, removes dead code
**Risk:** Low (breaking change acceptable)

---

### QW-2: Delete Unused Base Classes (1.5 hours)
**Files:** `plotting/base.py`, all plot functions

```bash
# 1. Delete base.py entirely (if exists)
rm complexplorer/plotting/base.py

# 2. Remove all config parameters
grep -r "config\s*=" complexplorer/plotting/
# Delete these parameters from all function signatures

# 3. Remove commented type hints
# Find: config = None) -> Figure:  # config: Optional[PlotConfig] = None
# Replace: ) -> Figure:
```

**Impact:** Removes 200+ lines of confusing dead code
**Risk:** None (wasn't being used)

---

### QW-3: Standardize Absolute Imports (1.5 hours)
**Files:** All Python files

```bash
# Find all relative imports
grep -r "^from \." complexplorer/
grep -r "^from \.\." complexplorer/

# Convert to absolute:
# from . import domain  →  from complexplorer.core import domain
# from ..core import X  →  from complexplorer.core import X
```

**Impact:** Clearer imports, prevents circular dependencies
**Risk:** None

---

### QW-4: Organize Root Directory (30 min)
**Files:** Root directory

```bash
# Create reviews directory for analysis docs
mkdir reviews/
mv CODE_ISSUES.md ARCHITECTURE_ISSUES.md DOCUMENTATION_ISSUES.md REFACTORING_ROADMAP.md reviews/

# Merge changelogs
cat CHANGELOG_V2.md >> CHANGELOG.md
rm CHANGELOG_V2.md

# Move development docs
mkdir -p docs/dev/
mv MIGRATION_GUIDE_V2.md docs/dev/migration-guide.md
mv TASKS_V2.md IMPLEMENTATION_TASKS.md .github/project/ 2>/dev/null || mkdir -p .github/project && mv TASKS_V2.md IMPLEMENTATION_TASKS.md .github/project/
```

**Impact:** Cleaner root, clearer organization
**Risk:** None

---

### QW-5: Define Magic Number Constants (2 hours)
**Files:** `core/colormap.py`, `core/functions.py`, `utils/validation.py`

**Create:** `core/constants.py`
```python
"""Shared constants for Complexplorer."""

# Colormap defaults
DEFAULT_BASE_BRIGHTNESS = 0.6
DEFAULT_BRIGHTNESS_CONTRAST = 0.3
DEFAULT_AUTO_SCALE_RADIUS = 0.8

# Stereographic projection
STEREOGRAPHIC_POLE_TOLERANCE = 1e-8

# Validation limits
MIN_RESOLUTION = 10
MAX_RESOLUTION = 1000
RECOMMENDED_2D_RESOLUTION = 500
RECOMMENDED_3D_RESOLUTION = 100

# STL export
MIN_PRINTABLE_SIZE_MM = 1.0
MAX_PRINTABLE_SIZE_MM = 500.0
MIN_WALL_THICKNESS_MM = 0.8

# PyVista material
DEFAULT_MATERIAL_PARAMS = {
    'specular': 0.5,
    'specular_power': 15,
    'diffuse': 0.7,
    'ambient': 0.3,
}
```

Then replace all hard-coded values with imports from this file.

**Impact:** Easier to tune, self-documenting
**Risk:** None

---

### QW-6: Fix Return Type Hints (2 hours)
**Files:** 23 functions across `api.py`, `plotting/`, `export/`

```python
# Add missing return types
def plot(...) -> Optional[Axes]:
def analyze_function(...) -> Dict[str, Any]:
def create_animation(...) -> Any:
# ... etc for all 23 functions

# Create py.typed marker
touch complexplorer/py.typed
```

**Impact:** Better IDE support, clearer contracts
**Risk:** None

---

### QW-7: Standardize Exception Hierarchy (1.5 hours)
**Files:** `core/validation.py`, all files using exceptions

```python
# In validation.py or new exceptions.py
class ComplexplorerError(Exception):
    """Base exception for all Complexplorer errors."""

class ValidationError(ComplexplorerError):
    """Input validation failed."""

class ComputationError(ComplexplorerError):
    """Numerical computation failed."""

class ExportError(ComplexplorerError):
    """File export operation failed."""

class DependencyError(ComplexplorerError):
    """Required dependency not available."""
```

Replace all `ValueError`, `RuntimeError`, etc. with appropriate subclass.

**Impact:** Better error handling, clearer error types
**Risk:** Low (breaking if users catch specific ValueError, but unlikely)

---

### QW-8: Add Validation to Colormap Constructors (2 hours)
**Files:** All colormap classes

```python
class Phase(Colormap):
    def __init__(self, phase_sectors=20, ...):
        if not isinstance(phase_sectors, int):
            raise ValidationError("phase_sectors must be an integer")
        if phase_sectors < 1:
            raise ValidationError("phase_sectors must be positive")
        # ... validate all parameters
        self.phase_sectors = phase_sectors
```

**Impact:** Fail fast with clear errors
**Risk:** None (better user experience)

---

## Major Refactoring - Phase 1: Code Quality (Total: ~25 hours)

### P1-1: Eliminate Colormap Duplication (6-8 hours)
**Priority:** CRITICAL - This is the biggest code smell

**Current:** 800+ lines duplicated across 6-8 colormap classes

**Plan:**
1. **Audit shared code** (1 hour)
   - List all duplicated blocks
   - Identify truly common vs slightly different

2. **Design base class methods** (1 hour)
   ```python
   class Colormap:
       def _validate_input(self, z):
           """Validate and normalize complex input."""

       def _apply_mask(self, H, S, V, outmask):
           """Apply outmask to HSV arrays."""

       def _hsv_to_rgb(self, H, S, V):
           """Standard HSV to RGB conversion."""

       def rgb(self, z, outmask=None):
           """Final RGB output. DONT OVERRIDE."""
           z = self._validate_input(z)
           H, S, V = self.hsv(z, outmask)  # Subclass implements
           return self._hsv_to_rgb(H, S, V)
   ```

3. **Refactor Phase class first** (2 hours)
   - Move shared code to base
   - Simplify Phase to only unique logic
   - Test thoroughly

4. **Refactor remaining colormaps** (2-3 hours)
   - One at a time
   - Test after each

5. **Delete duplicated code** (30 min)
   - Should remove ~600 lines

6. **Update tests** (1 hour)

**Impact:** Massive reduction in code, easier maintenance
**Risk:** Medium (could break colormap behavior if not careful)
**Test Strategy:** Visual regression tests - compare plots before/after

---

### P1-2: Standardize Parameter Naming (3-4 hours)
**Priority:** HIGH - User-facing API

**Changes:**
| Old | New | Files Affected |
|-----|-----|----------------|
| `n_phi` | `phase_sectors` | All Phase colormaps |
| `n`, `N` | `resolution` | All plot functions |
| `modulus_mode` | `modulus_scaling` | 3D plot functions |
| `colormap` | `cmap` | Some plot functions |

**Process:**
1. Create parameter mapping document
2. Use global search/replace with caution
3. Update all docstrings
4. Update all examples
5. Update migration guide
6. Test everything

**Breaking Changes:** Yes (document in migration guide)

---

### P1-3: Consolidate Plot Function Validation (2-3 hours)
**Priority:** HIGH - Reduces duplication

**Create:** `plotting/utils.py`
```python
def prepare_plot_inputs(domain=None, func=None, z=None, f=None, resolution=500):
    """Validate and prepare inputs for all plot functions.

    Returns
    -------
    z, f, mask : tuple
        Prepared mesh, function values, and mask.
    """
    # Validation
    if domain is None and z is None:
        raise ValidationError("Either domain or z must be provided")
    if f is None and func is None:
        raise ValidationError("Either f or func must be provided")

    # Get mesh
    if z is None:
        z = domain.mesh(resolution)
        mask = domain.outmask(resolution)
    else:
        mask = None

    # Evaluate function
    if f is None:
        f = func(z)

    # Ensure array
    f = np.asarray(f)
    if f.ndim == 0:
        f = np.full_like(z, f)

    return z, f, mask
```

**Usage in all plot functions:**
```python
def plot(domain=None, func=None, z=None, f=None, ...):
    z, f, mask = prepare_plot_inputs(domain, func, z, f, resolution)
    # ... rest of plotting logic
```

**Impact:** Removes 50-80 lines per function, 200+ total
**Risk:** Low

---

### P1-4: Fix OkLCh Gamut Clipping (30 min)
**Priority:** MEDIUM - Affects color accuracy

**File:** `core/color_utils.py:72-75`

**Change:**
```python
# Old
R = np.clip(R, 0, 1)
G = np.clip(G, 0, 1)
B = np.clip(B, 0, 1)

# New
R, G, B = clip_to_gamut(R, G, B, preserve='hue')
```

**Impact:** Better color accuracy
**Risk:** Low (improvement)

---

### P1-5: Add Logging Framework (4-5 hours)
**Priority:** MEDIUM - Better debugging and user feedback

**Create:** `core/logging.py`
```python
"""Logging configuration for Complexplorer."""
import logging

# Create logger
logger = logging.getLogger('complexplorer')
logger.setLevel(logging.INFO)

# Default handler (can be customized by user)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(name)s] %(levelname)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Convenience
def set_log_level(level):
    """Set complexplorer log level.

    Parameters
    ----------
    level : str or int
        'DEBUG', 'INFO', 'WARNING', 'ERROR' or logging constant.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    logger.setLevel(level)
```

**Usage throughout codebase:**
```python
from complexplorer.core.logging import logger

# Replace print() statements
# Old: if verbose: print("Generating mesh...")
# New: logger.info("Generating mesh...")

# Old: print(f"Warning: {msg}")
# New: logger.warning(msg)
```

**Keep verbose params for backwards compat:**
```python
def plot(..., verbose=False):
    if verbose:
        logger.setLevel(logging.DEBUG)
    # ...
```

**Impact:** Professional logging, better debugging
**Risk:** Low

---

### P1-6: Clean Up Mask Application (2 hours)
**Priority:** MEDIUM - Consistency

**Current:** Some colormaps use different mask colors

**Decision:** Standardize on gray (H=0, S=0, V=0.5)

**Update all colormaps:**
```python
def _apply_mask(self, H, S, V, outmask):
    """Apply mask to HSV components (base class method)."""
    if outmask is not None:
        outmask = np.asarray(outmask, dtype=bool)
        if outmask.shape != H.shape:
            raise ValidationError(...)
        H[outmask] = 0
        S[outmask] = 0
        V[outmask] = 0.5  # Gray
    return H, S, V
```

**Impact:** Consistent behavior
**Risk:** Low (visual change only)

---

## Major Refactoring - Phase 2: Architecture (Total: ~18 hours)

### P2-1: Simplify Plot API (8-10 hours)
**Priority:** HIGH - Core API design
**Breaking Change:** YES

**Current Problem:** 4 ways to call every plot function

**New Design:**
```python
def plot(domain: Domain,
         func: Callable,
         resolution: int = 500,
         cmap: Optional[Colormap] = None,
         ...) -> Axes:
    """High-level plotting: provide domain and function."""
    z = domain.mesh(resolution)
    f = func(z)
    mask = domain.outmask(resolution)
    return plot_arrays(z, f, mask, cmap, ...)

def plot_arrays(z: np.ndarray,
                f: np.ndarray,
                mask: Optional[np.ndarray] = None,
                cmap: Optional[Colormap] = None,
                ...) -> Axes:
    """Low-level plotting: provide pre-computed arrays."""
    # Direct plotting implementation
```

**Migration:**
```python
# Old: plot(domain=d, func=f)  → Still works!
# Old: plot(z=mesh, f=vals)    → plot_arrays(mesh, vals)
```

**Process:**
1. Implement new `plot_arrays()` functions (3 hours)
2. Refactor existing `plot()` to call `plot_arrays()` (2 hours)
3. Update all 13+ plot functions (3 hours)
4. Update examples and docs (2 hours)
5. Test thoroughly

**Impact:** Cleaner API, easier to understand
**Risk:** Medium (breaking change, significant refactor)

---

### P2-2: Reorganize Utils Package (2-3 hours)
**Priority:** MEDIUM - Better organization

**Current:**
```
utils/
├── validation.py
├── mesh.py
├── mesh_distortion.py
└── color_utils.py
```

**New:**
```
core/
├── domain.py
├── colormap.py
├── functions.py
├── scaling.py
├── mesh.py          # merge mesh + mesh_distortion
├── color_spaces.py  # was color_utils
└── validation.py    # promoted
```

**Process:**
1. Move files
2. Update all imports
3. Test

**Impact:** Clearer organization
**Risk:** Low (just moving files)

---

### P2-3: Flatten Plotting Subpackage (1 hour)
**Priority:** LOW-MEDIUM - Simpler imports

**Current:**
```
plotting/
├── matplotlib/
│   ├── plot_2d.py
│   └── plot_3d.py
└── pyvista/
    └── plot_3d.py
```

**New:**
```
plotting/
├── plot_2d.py    # matplotlib 2D
├── plot_3d.py    # matplotlib 3D
└── pyvista/      # keep separate (optional dep)
    └── plot_3d.py
```

**Impact:** Shorter import paths
**Risk:** Low (breaking import paths)

---

### P2-4: Optional Dependency Management (2-3 hours)
**Priority:** MEDIUM - Better error messages

**Create:** `utils/deps.py`
```python
class OptionalDependency:
    def __init__(self, name, package_name=None, install_extra=None):
        self.name = name
        self.package = package_name or name.lower()
        self.install_extra = install_extra or name.lower()
        self._module = None
        self._available = None

    def is_available(self) -> bool:
        if self._available is None:
            try:
                self._module = __import__(self.package)
                self._available = True
            except ImportError:
                self._available = False
        return self._available

    def require(self):
        if not self.is_available():
            raise DependencyError(
                f"{self.name} is required for this feature.\n"
                f"Install with: pip install complexplorer[{self.install_extra}]"
            )
        return self._module

# Instances
pyvista = OptionalDependency('PyVista', 'pyvista')
pyqt = OptionalDependency('PyQt6', 'PyQt6', install_extra='qt')
```

**Usage:**
```python
from complexplorer.utils.deps import pyvista

def plot_landscape_pv(...):
    pv = pyvista.require()  # Clean error if not installed
    # ... use pv
```

**Impact:** Better UX for optional deps
**Risk:** Low

---

### P2-5: Import Layer Enforcement (1-2 hours)
**Priority:** MEDIUM - Prevent circular imports

**Create:** `tests/test_imports.py`
```python
def test_no_circular_imports():
    """Ensure no circular import dependencies."""
    import complexplorer
    # If this passes, no circular imports

def test_layer_violations():
    """Enforce layered architecture."""
    # Layer 1: validation, functions (no internal imports)
    import complexplorer.core.validation
    import complexplorer.core.functions

    # Layer 2: domain, colormap (can import Layer 1)
    import complexplorer.core.domain
    import complexplorer.core.colormap

    # Layer 3: plotting (can import Layer 1-2)
    import complexplorer.plotting.plot_2d
```

**Create dependency graph:**
```bash
# Use pydeps to visualize
pip install pydeps
pydeps complexplorer --max-bacon=2 -o docs/dependency-graph.png
```

**Impact:** Prevents architectural degradation
**Risk:** Low

---

## Documentation Overhaul (Total: ~40-90 hours depending on scope)

### D-1: Decision: Documentation Strategy (30 min)
**Choose:** Minimal (20-30 hrs) vs Full MkDocs (60-90 hrs)

**Recommendation:** Full MkDocs Material
- Professional appearance
- Necessary for academic adoption
- Future-proof

**See:** `DOCUMENTATION_ISSUES.md` for complete plan

---

### D-2: Set Up MkDocs (If Chosen) (4-6 hours)
1. Install and configure MkDocs Material
2. Create basic page structure
3. Set up GitHub Actions for auto-deploy
4. Migrate existing content

**See:** `DOCUMENTATION_ISSUES.md` Section 10.1

---

### D-3: Core Documentation Content (20-30 hours)
**Priority:** HIGH - Essential for users

1. Improve README (2-3 hrs)
2. Getting Started guide (3-4 hrs)
3. Colormap guide (6-8 hrs)
4. API reference setup (3-4 hrs)
5. Migration guide expansion (3-4 hrs)
6. Performance guide (4-5 hrs)
7. Troubleshooting guide (3-4 hrs)

---

### D-4: Examples and Tutorials (25-35 hours)
**Priority:** HIGH - Showcase capability

1. Clean existing notebooks (5-6 hrs)
2. Create tutorial sequence (10-12 hrs)
3. Build examples gallery (10-15 hrs)

---

### D-5: Docstring Improvements (15-20 hours)
**Priority:** HIGH - API usability

1. Standardize on NumPy style (5-6 hrs)
2. Add examples to all public functions (8-10 hrs)
3. Add mathematical context (4-5 hrs)

---

## Testing Improvements (Total: ~15 hours)

### T-1: Visual Regression Tests (6-8 hours)
**Priority:** HIGH - Catch colormap changes

**Setup pytest-mpl:**
```bash
pip install pytest-mpl
```

**Create:** `tests/visual/test_plot_regression.py`
```python
@pytest.mark.mpl_image_compare
def test_phase_portrait():
    """Ensure Phase colormap output doesn't change."""
    domain = Rectangle(4, 4)
    plot(domain, lambda z: z**2, cmap=Phase(phase_sectors=6))
    return plt.gcf()
```

**Generate baselines:**
```bash
pytest --mpl-generate-path=tests/visual/baseline
```

**Run tests:**
```bash
pytest --mpl
```

---

### T-2: Property-Based Tests (4-5 hours)
**Priority:** MEDIUM - Find edge cases

```python
from hypothesis import given, strategies as st

@given(st.complex_numbers(allow_nan=False, allow_infinity=False))
def test_phase_range(z):
    """Phase always in [0, 2π)."""
    from complexplorer.core.functions import phase
    phi = phase(z)
    assert 0 <= phi < 2 * np.pi

@given(st.complex_numbers())
def test_colormap_no_nan(z):
    """Colormaps never produce NaN for finite input."""
    cmap = Phase()
    rgb = cmap.rgb(np.array([[z]]))
    assert not np.any(np.isnan(rgb))
```

---

### T-3: Performance Benchmarks (3-4 hours)
**Priority:** LOW - Track performance regression

**Setup pytest-benchmark:**
```bash
pip install pytest-benchmark
```

**Create:** `tests/benchmarks/test_performance.py`
```python
def test_mesh_generation_speed(benchmark):
    domain = Rectangle(4, 4)
    result = benchmark(domain.mesh, 500)
    assert result.shape == (500, 500)

def test_plot_speed(benchmark):
    domain = Rectangle(4, 4)
    func = lambda z: z**2
    benchmark(plot, domain, func, resolution=500)
```

---

## Release Preparation (Total: ~10 hours)

### R-1: Update Version and Metadata (1 hour)
**Files:** `pyproject.toml`, `__init__.py`

```toml
[project]
name = "complexplorer"
version = "2.0.0"
description = "Domain coloring for complex functions with 3D visualization and STL export"
authors = [{name = "Your Name", email = "your@email.com"}]
readme = "README.md"
requires-python = ">=3.9"
# ...
```

---

### R-2: Finalize Changelog (2 hours)
Merge all changes into `CHANGELOG.md` following Keep a Changelog format.

---

### R-3: Complete Migration Guide (2 hours)
Ensure all breaking changes documented with examples.

---

### R-4: Review and Test Everything (4-5 hours)
1. Run full test suite
2. Test all examples
3. Build documentation
4. Test installation from source
5. Test on fresh environment

---

### R-5: Create Release (1 hour)
```bash
# Tag release
git tag -a v2.0.0 -m "Version 2.0: Clean API and enhanced features"
git push origin v2.0.0

# Build distribution
python -m build

# Upload to PyPI
python -m twine upload dist/*
```

---

## Timeline and Effort Summary

### By Priority

| Priority Level | Total Effort | Description |
|----------------|--------------|-------------|
| **CRITICAL** | 6-8 hours | Colormap duplication |
| **HIGH** | ~95-125 hours | API fixes, docs, examples |
| **MEDIUM** | ~30-40 hours | Logging, organization, testing |
| **LOW** | ~5-10 hours | Nice-to-have improvements |

### By Phase

| Phase | Effort | Description |
|-------|--------|-------------|
| **Quick Wins** | 12 hrs | Immediate improvements |
| **Code Quality** | 25 hrs | Duplication, naming, validation |
| **Architecture** | 18 hrs | API, organization, structure |
| **Documentation** | 40-90 hrs | Minimal to full redesign |
| **Testing** | 15 hrs | Regression, property, benchmarks |
| **Release** | 10 hrs | Final prep and release |
| **TOTAL** | **120-170 hrs** | Complete refactor |

### Recommended Phased Approach

#### Sprint 1: Quick Wins + Critical (2 weeks, ~20 hours)
1. All quick wins
2. Colormap duplication fix
3. Basic docs cleanup

**Deliverable:** v2.0-alpha with major code improvements

---

#### Sprint 2: API and Architecture (2 weeks, ~25 hours)
1. Standardize parameter naming
2. Plot API simplification
3. Package reorganization
4. Logging framework

**Deliverable:** v2.0-beta with clean API

---

#### Sprint 3: Documentation - Minimal (1 week, ~20 hours) OR
#### Sprint 3-4: Documentation - Full (3-4 weeks, ~60-90 hours)
1. Set up MkDocs (if chosen)
2. Core content
3. Examples cleanup
4. Docstring improvements

**Deliverable:** v2.0-rc1 with documentation

---

#### Sprint 4/5: Testing and Release (1 week, ~15 hours)
1. Visual regression tests
2. Property tests
3. Benchmarks
4. Final review
5. Release v2.0

---

## Risk Management

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Breaking existing users | Medium | High | Migration guide, clear changelog |
| Introducing bugs in refactor | Medium | High | Visual regression tests, comprehensive testing |
| Documentation effort underestimated | High | Medium | Start with minimal, expand later |
| Colormap refactor changes behavior | Low-Medium | Medium | Pixel-perfect comparison tests |

---

## Success Metrics

- [ ] Code reduction: > 600 lines removed (duplicated colormap code)
- [ ] No TODO/FIXME comments remain
- [ ] All public functions have docstrings with examples
- [ ] Test coverage > 85%
- [ ] No backwards compatibility shims
- [ ] All hard-coded values extracted to constants
- [ ] All parameters consistently named
- [ ] Documentation site deployed (if doing full docs)
- [ ] Migration guide complete
- [ ] All examples execute without errors

---

## Post-Release (v2.1+)

### Future Enhancements
1. Configuration system (from ARCHITECTURE_ISSUES.md 4.2)
2. Caching system for expensive operations
3. Animation support (partially implemented)
4. Icosphere-based Riemann sphere (better than rectangular)
5. Interactive widget for Jupyter
6. More colormap families
7. Performance optimizations

### Community Building
1. Create examples gallery website
2. Blog posts about features
3. Academic paper about implementation
4. Conference presentation
5. YouTube tutorials

---

**Review Date:** 2025-10-14
**Next Steps:**
1. Agree on timeline (full refactor vs minimal)
2. Agree on documentation strategy (minimal vs MkDocs)
3. Start with Quick Wins
4. Proceed through sprints

**Breaking Changes Are Acceptable** - small user base, clean code more important.
