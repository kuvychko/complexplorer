# Detailed Implementation Plan for Complexplorer Refactoring

## Overview

This document provides a detailed, step-by-step implementation plan for refactoring the complexplorer codebase based on the improvement plan. Each step includes specific tasks, validation criteria, and rollback strategies.

## Phase 0: Preparation and Legacy Migration (Days 1-2)

### Step 0.1: Create Legacy Directory Structure
```bash
mkdir -p complexplorer/legacy
mkdir -p tests/legacy
```

### Step 0.2: Move Current Files to Legacy
**Files to move:**
```bash
# Core files
mv complexplorer/domain.py complexplorer/legacy/
mv complexplorer/cmap.py complexplorer/legacy/
mv complexplorer/funcs.py complexplorer/legacy/
mv complexplorer/plots_2d.py complexplorer/legacy/
mv complexplorer/plots_3d.py complexplorer/legacy/
mv complexplorer/plots_3d_pyvista.py complexplorer/legacy/
mv complexplorer/mesh_utils.py complexplorer/legacy/
mv complexplorer/utils.py complexplorer/legacy/

# STL export files
mv complexplorer/stl_export complexplorer/legacy/

# Tests
mv tests/unit/test_*.py tests/legacy/
```

### Step 0.3: Create Temporary Compatibility Layer
```python
# complexplorer/__init__.py (temporary)
"""Temporary compatibility layer during migration."""
import warnings

# Import from legacy to maintain compatibility
from .legacy.domain import *
from .legacy.cmap import *
from .legacy.funcs import *
from .legacy.plots_2d import *
from .legacy.plots_3d import *
from .legacy.plots_3d_pyvista import *
from .legacy.mesh_utils import *
from .legacy.utils import *
from .legacy.stl_export import *

warnings.warn("Using legacy imports. Please update to new API.", DeprecationWarning)
```

**Validation:**
- Run existing tests: `pytest tests/legacy/`
- All tests should pass
- Example notebooks should still work

## Phase 1: Foundation - Core Infrastructure (Days 3-5)

### Step 1.1: Create New Directory Structure
```bash
# Create new structure
mkdir -p complexplorer/core
mkdir -p complexplorer/plotting/base
mkdir -p complexplorer/plotting/matplotlib
mkdir -p complexplorer/plotting/pyvista
mkdir -p complexplorer/export/base
mkdir -p complexplorer/export/stl
mkdir -p complexplorer/utils
mkdir -p tests/unit/core
mkdir -p tests/unit/plotting
mkdir -p tests/unit/export
mkdir -p tests/unit/utils
```

### Step 1.2: Extract ModulusScaling to Core
**Task:** Create `complexplorer/core/scaling.py`
```python
# complexplorer/core/scaling.py
"""Modulus scaling methods for complex function visualization."""

import numpy as np
from typing import Callable

class ModulusScaling:
    """Collection of modulus scaling methods for visualization."""
    
    # Copy all methods from legacy/mesh_utils.py ModulusScaling class
    # Add new methods (sigmoid, adaptive, hybrid) from recent implementation
    
SCALING_PRESETS = {
    'balanced': {...},
    'detail_near_zero': {...},
    'auto': {...}
}

def get_scaling_preset(name: str) -> dict:
    """Get a predefined scaling configuration."""
    # Implementation
```

**Tests to migrate:**
- Create `tests/unit/core/test_scaling.py`
- Extract relevant tests from `tests/legacy/test_mesh_utils.py`
- Add new tests for sigmoid, adaptive, hybrid methods
- Add tests for presets

**Validation:**
```bash
pytest tests/unit/core/test_scaling.py -v
```

### Step 1.3: Create Validation Module
**Task:** Create `complexplorer/utils/validation.py`
```python
# complexplorer/utils/validation.py
"""Centralized validation utilities."""

from typing import Optional, Callable
import numpy as np

class ValidationError(ValueError):
    """Custom exception for validation errors."""
    pass

def validate_domain_or_mesh(domain: Optional['Domain'], 
                          mesh: Optional[np.ndarray]) -> None:
    """Validate that either domain or mesh is provided."""
    # Extract validation logic from legacy plotting functions

def validate_function(func: Callable) -> Callable:
    """Validate and wrap function for safe evaluation."""
    # Implementation

def validate_colormap(cmap: Optional['Cmap']) -> 'Cmap':
    """Validate colormap with sensible defaults."""
    # Implementation

def validate_stl_parameters(size_mm: float, wall_thickness: float) -> None:
    """Validate STL export parameters."""
    # Implementation
```

**Tests:**
- Create `tests/unit/utils/test_validation.py`
- Test each validation function with valid and invalid inputs
- Test custom exceptions

### Step 1.4: Create Abstract Base Classes
**Task:** Create `complexplorer/plotting/base.py`
```python
# complexplorer/plotting/base.py
"""Abstract base classes for plotting."""

from abc import ABC, abstractmethod
from typing import Optional, Callable
from ..utils.validation import validate_domain_or_mesh, validate_function, validate_colormap

class BasePlotter(ABC):
    """Abstract base class for all plotters."""
    
    def __init__(self, domain: Optional['Domain'], 
                 func: Callable, 
                 cmap: Optional['Cmap']):
        self.domain = domain
        self.func = validate_function(func)
        self.cmap = validate_colormap(cmap)
        self._data = None
    
    @abstractmethod
    def plot(self, **kwargs):
        """Main plotting method to be implemented by subclasses."""
        pass
    
    def _prepare_data(self):
        """Common data preparation logic."""
        # Extract common logic from legacy plotting functions

class Base2DPlotter(BasePlotter):
    """Base class for 2D plotting."""
    pass

class Base3DPlotter(BasePlotter):
    """Base class for 3D plotting."""
    
    def get_mesh_data(self):
        """Get mesh data for export."""
        if self._data is None:
            self._prepare_data()
        return self._data
```

**Task:** Create `complexplorer/export/base.py`
```python
# complexplorer/export/base.py
"""Base export functionality."""

from abc import ABC, abstractmethod

class BaseExporter(ABC):
    """Base class for export functionality."""
    
    @abstractmethod
    def export(self, data, filename: str, **kwargs):
        """Export data to file."""
        pass
```

**Tests:**
- Create `tests/unit/plotting/test_base.py`
- Test abstract base class behavior
- Test common methods

## Phase 2: Core Module Migration (Days 6-10)

### Step 2.1: Migrate Domain Classes
**Task:** Create `complexplorer/core/domain.py`
```python
# complexplorer/core/domain.py
"""Domain classes for complex plane regions."""

# Copy from legacy/domain.py
# Update imports to use new structure
# Standardize parameter names (e.g., 're_bounds' → 'real_bounds')
```

**Tests to migrate:**
- Copy `tests/legacy/test_domain.py` to `tests/unit/core/test_domain.py`
- Update imports
- Add any missing test cases

**Validation:**
```bash
pytest tests/unit/core/test_domain.py -v
```

### Step 2.2: Migrate Colormap Classes
**Task:** Create `complexplorer/core/colormap.py`
```python
# complexplorer/core/colormap.py
"""Color mapping classes for complex visualization."""

# Copy from legacy/cmap.py
# Rename 'Cmap' to 'Colormap' for clarity
# Update all references
```

**Tests to migrate:**
- Copy `tests/legacy/test_cmap.py` to `tests/unit/core/test_colormap.py`
- Update class names and imports

### Step 2.3: Migrate Mathematical Functions
**Task:** Create `complexplorer/core/functions.py`
```python
# complexplorer/core/functions.py
"""Mathematical utility functions."""

# Copy from legacy/funcs.py
# Move stereographic projection functions from mesh_utils.py here
# Organize functions logically
```

**Tests to migrate:**
- Copy relevant tests to `tests/unit/core/test_functions.py`

### Step 2.4: Create Mesh Utilities
**Task:** Create `complexplorer/utils/mesh.py`
```python
# complexplorer/utils/mesh.py
"""Mesh generation utilities."""

# Copy mesh generation functions from legacy/mesh_utils.py
# EXCLUDE ModulusScaling (already in core/scaling.py)
# Update imports to use core.scaling
```

## Phase 3: Plotting Module Refactoring (Days 11-15)

### Step 3.1: Implement 2D Matplotlib Plotting
**Task:** Create `complexplorer/plotting/matplotlib/plot_2d.py`
```python
# complexplorer/plotting/matplotlib/plot_2d.py
"""2D plotting with matplotlib."""

from ..base import Base2DPlotter
from ...core.domain import Domain
from ...core.colormap import Colormap

class Plot2D(Base2DPlotter):
    """2D complex function plotter."""
    
    def plot(self, **kwargs):
        """Create 2D plot."""
        # Migrate logic from legacy/plots_2d.py
        # Use base class methods
        # Remove duplicate validation

# Function wrappers for backward compatibility
def plot(func: Callable, domain: Domain = None, 
         colormap: Colormap = None, **kwargs):
    """Plot complex function in 2D."""
    plotter = Plot2D(domain, func, colormap)
    return plotter.plot(**kwargs)

def pair_plot(func: Callable, domain: Domain = None, 
              colormap: Colormap = None, **kwargs):
    """Create side-by-side domain/codomain plot."""
    # Implementation
```

**Tests to migrate:**
- Split `tests/legacy/test_plots_2d.py` into focused test files
- Create `tests/unit/plotting/matplotlib/test_plot_2d.py`

### Step 3.2: Implement 3D Matplotlib Plotting
**Task:** Create `complexplorer/plotting/matplotlib/plot_3d.py`
```python
# complexplorer/plotting/matplotlib/plot_3d.py
"""3D plotting with matplotlib."""

from ..base import Base3DPlotter

class Plot3D(Base3DPlotter):
    """3D complex function plotter using matplotlib."""
    
    def plot(self, **kwargs):
        """Create 3D plot."""
        # Migrate from legacy/plots_3d.py
```

### Step 3.3: Split PyVista Plotting
**Task:** Split `legacy/plots_3d_pyvista.py` into:

1. `complexplorer/plotting/pyvista/plot_3d.py` - Landscape plots
2. `complexplorer/plotting/pyvista/riemann.py` - Riemann sphere
3. `complexplorer/plotting/pyvista/utils.py` - PyVista utilities

```python
# complexplorer/plotting/pyvista/plot_3d.py
"""3D landscape plotting with PyVista."""

from ..base import Base3DPlotter
from ...core.scaling import ModulusScaling

class LandscapePlotter(Base3DPlotter):
    """3D landscape plotter using PyVista."""
    
    def plot(self, **kwargs):
        """Create 3D landscape plot."""
        # Migrate plot_landscape_pv function
```

```python
# complexplorer/plotting/pyvista/riemann.py
"""Riemann sphere visualization with PyVista."""

from ..base import Base3DPlotter
from ...core.scaling import ModulusScaling, get_scaling_preset

class RiemannSpherePlotter(Base3DPlotter):
    """Riemann sphere plotter using PyVista."""
    
    def __init__(self, func: Callable, scaling: str = 'constant',
                 scaling_params: dict = None, scaling_preset: str = None):
        super().__init__(None, func, None)
        self.scaling = scaling
        self.scaling_params = scaling_params or {}
        
        # Handle presets
        if scaling_preset:
            preset = get_scaling_preset(scaling_preset)
            self.scaling = preset['method']
            self.scaling_params.update(preset['params'])
    
    def plot(self, **kwargs):
        """Create Riemann sphere visualization."""
        # Migrate riemann_pv function
```

**Tests:**
- Create separate test files for each module
- Ensure all PyVista functionality is tested

## Phase 4: STL Export Migration (Days 16-18)

### Step 4.1: Migrate STL Export
**Task:** Move STL export to new structure

```bash
# Move files
mv complexplorer/legacy/stl_export/ornament_generator.py complexplorer/export/stl/generator.py
mv complexplorer/legacy/stl_export/spherical_healing.py complexplorer/export/stl/healing.py
```

**Update imports in moved files:**
```python
# complexplorer/export/stl/generator.py
from ..base import BaseExporter
from ...core.scaling import ModulusScaling
from ...core.domain import Domain
from ...utils.mesh import RectangularSphereGenerator

class OrnamentGenerator(BaseExporter):
    """Generate 3D-printable ornaments from complex functions."""
    # Update to use new imports
```

### Step 4.2: Complete STL TODOs
**Task:** Fix the boundary loop extraction
```python
# complexplorer/export/stl/healing.py
def extract_boundary_loops(mesh: pv.PolyData) -> List[np.ndarray]:
    """Extract properly ordered boundary loops from mesh edges."""
    # Complete the implementation
    # Remove TODO comment
```

### Step 4.3: Add STL Tests
**Task:** Create comprehensive STL tests
```python
# tests/unit/export/stl/test_generator.py
# tests/unit/export/stl/test_healing.py
# Implement all tests from improvement plan
```

## Phase 5: API Integration (Days 19-21)

### Step 5.1: Create Main API
**Task:** Update `complexplorer/__init__.py`
```python
# complexplorer/__init__.py
"""Complexplorer - Visualize complex functions."""

# Core concepts
from .core.domain import Domain, Rectangle, Disk, Annulus
from .core.colormap import Colormap, Phase, Chessboard, PolarChessboard
from .core.scaling import ModulusScaling, get_scaling_preset
from .core.functions import sawtooth, stereographic_projection

# Plotting functions
from .plotting.matplotlib.plot_2d import plot, pair_plot
from .plotting.matplotlib.plot_3d import plot_landscape, pair_plot_landscape
from .plotting.pyvista.plot_3d import plot_landscape_pv, pair_plot_landscape_pv
from .plotting.pyvista.riemann import riemann_pv as riemann_sphere

# Export functionality
from .export.stl import OrnamentGenerator

# Unified interfaces
def generate_ornament(func, filename: str, **kwargs):
    """Generate 3D-printable ornament from complex function."""
    if 'scaling_preset' in kwargs:
        preset = get_scaling_preset(kwargs.pop('scaling_preset'))
        kwargs.update(preset)
    
    gen = OrnamentGenerator(func, **kwargs)
    return gen.generate_ornament(filename)

def export(plot_object, filename: str, **kwargs):
    """Unified export interface."""
    # Implementation
```

### Step 5.2: Remove Legacy Compatibility Layer
**Task:** Clean up after successful migration
```python
# Remove temporary imports from __init__.py
# Add proper deprecation warnings if needed
```

## Phase 6: Test Migration and Validation (Days 22-24)

### Step 6.1: Migrate All Tests
**Process for each test file:**
1. Copy test from `tests/legacy/` to appropriate location in `tests/unit/`
2. Update imports to use new module structure
3. Run test to ensure it passes
4. Add any missing test cases
5. Delete legacy test file

**Test migration checklist:**
- [ ] `test_domain.py` → `tests/unit/core/test_domain.py`
- [ ] `test_cmap.py` → `tests/unit/core/test_colormap.py`
- [ ] `test_funcs.py` → `tests/unit/core/test_functions.py`
- [ ] `test_mesh_utils.py` → Split into:
  - [ ] `tests/unit/core/test_scaling.py`
  - [ ] `tests/unit/utils/test_mesh.py`
- [ ] `test_plots_2d.py` → `tests/unit/plotting/matplotlib/test_plot_2d.py`
- [ ] `test_plots_3d.py` → `tests/unit/plotting/matplotlib/test_plot_3d.py`
- [ ] `test_plots_3d_pyvista.py` → Split into:
  - [ ] `tests/unit/plotting/pyvista/test_plot_3d.py`
  - [ ] `tests/unit/plotting/pyvista/test_riemann.py`
- [ ] Add `tests/unit/export/stl/` tests

### Step 6.2: Validate Test Coverage
```bash
# Run all tests with coverage
pytest tests/unit/ --cov=complexplorer --cov-report=html

# Ensure coverage > 90%
# Add tests for any uncovered code
```

### Step 6.3: Integration Testing
**Create integration tests:**
```python
# tests/integration/test_full_pipeline.py
def test_2d_plot_pipeline():
    """Test complete 2D plotting pipeline."""
    domain = Rectangle(4, 4)
    plot(lambda z: z**2, domain=domain)

def test_3d_stl_pipeline():
    """Test complete 3D to STL pipeline."""
    gen = OrnamentGenerator(lambda z: (z-1)/(z+1))
    gen.generate_ornament("test.stl")
    # Verify STL is valid

def test_scaling_integration():
    """Test ModulusScaling integration."""
    riemann_sphere(lambda z: z**2, scaling_preset='balanced')
```

## Phase 7: Example Migration (Days 25-26)

### Step 7.1: Create Migration Script
**Task:** Create automated notebook migration
```python
# scripts/migrate_notebooks.py
"""Migrate example notebooks to new API."""

import nbformat
import re

def migrate_notebook(input_path, output_path):
    """Migrate a notebook to use new API."""
    nb = nbformat.read(input_path, as_version=4)
    
    for cell in nb.cells:
        if cell.cell_type == 'code':
            # Update imports
            cell.source = re.sub(
                r'from complexplorer import (\w+)',
                r'import complexplorer as cp\n# Updated: \1 is now cp.\1',
                cell.source
            )
            
            # Update function calls
            replacements = {
                'riemann_pv': 'cp.riemann_sphere',
                'plot': 'cp.plot',
                # etc.
            }
            
            for old, new in replacements.items():
                cell.source = cell.source.replace(old, new)
    
    nbformat.write(nb, output_path)
```

### Step 7.2: Migrate Each Example
**Notebooks to migrate:**
1. `examples/basic_examples.ipynb`
2. `examples/riemann_sphere_demo.ipynb`
3. `examples/stl_ornament_demo.ipynb`
4. Any other examples

**For each notebook:**
1. Run migration script
2. Manually review and fix any issues
3. Run notebook to ensure it works
4. Add comments explaining API changes

### Step 7.3: Create New Examples
**Create examples showcasing new features:**
```python
# examples/new_api_demo.ipynb
# Demonstrate:
# - Unified export interface
# - ModulusScaling as first-class concept
# - Scaling presets
# - Builder pattern (if implemented)
```

## Phase 8: Documentation and Cleanup (Days 27-28)

### Step 8.1: Update Documentation
- Update README.md with new API
- Create migration guide
- Document breaking changes
- Add API reference

### Step 8.2: Clean Up Legacy Code
```bash
# After all tests pass and examples work
rm -rf complexplorer/legacy/
rm -rf tests/legacy/
```

### Step 8.3: Final Validation
1. Run full test suite
2. Check all examples work
3. Build and test package
4. Verify imports work as expected

## Rollback Strategy

At each phase, if issues arise:

1. **Phase 0-1**: Simply restore original files from legacy/
2. **Phase 2-4**: Keep both implementations, fix issues incrementally
3. **Phase 5-7**: Maintain compatibility layer longer if needed
4. **Phase 8**: Only proceed when fully confident

## Success Criteria

Each phase is complete when:
- All tests pass (old and new)
- No regression in functionality
- Code coverage maintained or improved
- Examples work correctly
- Performance benchmarks show no regression

## Daily Checklist

For each implementation day:
- [ ] Write new code following the plan
- [ ] Migrate corresponding tests
- [ ] Run tests to ensure they pass
- [ ] Update imports in dependent modules
- [ ] Document any deviations from plan
- [ ] Commit working state before moving on

## Risk Mitigation

1. **Always work in a branch**: Never modify main directly
2. **Test continuously**: Run tests after each change
3. **Keep legacy code**: Don't delete until fully migrated
4. **Document decisions**: Note any deviations from plan
5. **Regular backups**: Commit working states frequently

This detailed plan provides a systematic approach to refactoring complexplorer while maintaining functionality and improving code quality throughout the process.