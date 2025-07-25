# Complexplorer Codebase Improvement Plan

## Executive Summary

This document outlines a comprehensive plan to improve the modularity, maintainability, and code quality of the complexplorer library. The improvements are organized by priority and include specific action items with estimated complexity.

**Key Goals:**
- Improve modularity through better separation of concerns
- Eliminate code duplication
- Standardize patterns and conventions
- Enhance maintainability and extensibility
- Simplify complex functions and modules
- Preserve and enhance STL export functionality for 3D printing

## Priority 1: Critical Architectural Improvements (High Impact, High Priority)

### 1.1 Create Abstract Base Classes for Plotting [★★★★★]

**Problem:** Significant code duplication between matplotlib and PyVista plotting modules.

**Solution:** Introduce abstract base classes to share common logic.

**Action Items:**
```python
# complexplorer/plotting/base.py
from abc import ABC, abstractmethod

class BasePlotter(ABC):
    """Abstract base class for all plotters."""
    
    def __init__(self, domain, func, cmap):
        self.domain = self._validate_domain(domain)
        self.func = self._validate_function(func)
        self.cmap = self._validate_colormap(cmap)
    
    @abstractmethod
    def plot(self, **kwargs):
        """Main plotting method to be implemented by subclasses."""
        pass
    
    def _prepare_data(self):
        """Common data preparation logic."""
        pass
    
    def _validate_domain(self, domain):
        """Common domain validation."""
        pass

class Base2DPlotter(BasePlotter):
    """Base class for 2D plotting."""
    pass

class Base3DPlotter(BasePlotter):
    """Base class for 3D plotting."""
    pass

class BaseExporter(ABC):
    """Base class for export functionality."""
    
    @abstractmethod
    def export(self, data, filename, **kwargs):
        """Export data to file."""
        pass
```

**Benefits:**
- Eliminates ~40% code duplication
- Enforces consistent interfaces
- Simplifies testing
- Provides base for STL export

### 1.2 Extract Validation Module [★★★★★]

**Problem:** Validation logic is scattered and duplicated across all plotting functions.

**Solution:** Create a centralized validation module.

**Action Items:**
```python
# complexplorer/validation.py
from typing import Optional, Union, Callable
import numpy as np

class ValidationError(ValueError):
    """Custom exception for validation errors."""
    pass

def validate_domain_or_mesh(domain: Optional[Domain], 
                          mesh: Optional[np.ndarray]) -> None:
    """Validate that either domain or mesh is provided."""
    if domain is None and mesh is None:
        raise ValidationError("Either domain or mesh parameter must be provided")

def validate_function(func: Callable) -> Callable:
    """Validate and wrap function for safe evaluation."""
    if not callable(func):
        raise ValidationError("Function parameter must be callable")
    
    def safe_func(z):
        try:
            return func(z)
        except Exception as e:
            raise ValidationError(f"Function evaluation failed: {e}")
    
    return safe_func

def validate_colormap(cmap: Optional[Cmap]) -> Cmap:
    """Validate colormap with sensible defaults."""
    if cmap is None:
        from .cmap import Phase
        return Phase(n_sectors=6, scale_radius=0.6)
    return cmap

def validate_stl_parameters(size_mm: float, wall_thickness: float) -> None:
    """Validate STL export parameters."""
    if size_mm <= 0:
        raise ValidationError("Size must be positive")
    if wall_thickness <= 0 or wall_thickness >= size_mm / 2:
        raise ValidationError("Invalid wall thickness")
```

**Benefits:**
- Centralizes validation logic
- Improves error messages consistency
- Reduces code by ~200 lines
- Supports STL export validation

### 1.3 Extract ModulusScaling to Core Module [★★★★★]

**Problem:** ModulusScaling class is buried in mesh_utils.py but is a core concept used across multiple visualization types.

**Solution:** Move ModulusScaling to its own module in core.

**Action Items:**
```python
# complexplorer/core/scaling.py
"""Modulus scaling methods for complex function visualization."""

import numpy as np
from typing import Callable
from abc import ABC, abstractmethod

class ModulusScaler(ABC):
    """Abstract base for modulus scaling methods."""
    
    @abstractmethod
    def scale(self, moduli: np.ndarray, **kwargs) -> np.ndarray:
        """Apply scaling to modulus values."""
        pass

class ModulusScaling:
    """Collection of modulus scaling methods for visualization.
    
    These methods map the modulus |f(z)| to a radius value, allowing
    visualizations to show both phase and magnitude information.
    """
    
    @staticmethod
    def constant(moduli: np.ndarray, radius: float = 1.0) -> np.ndarray:
        """Constant radius regardless of modulus."""
        return np.full_like(moduli, radius, dtype=float)
    
    @staticmethod
    def linear(moduli: np.ndarray, scale: float = 0.1) -> np.ndarray:
        """Linear scaling: r = 1 + scale * |f(z)|."""
        return 1.0 + scale * moduli
    
    @staticmethod
    def arctan(moduli: np.ndarray, r_min: float = 0.5, r_max: float = 1.5) -> np.ndarray:
        """Smooth scaling using arctangent."""
        normalized = (2/np.pi) * np.arctan(moduli)
        return r_min + (r_max - r_min) * normalized
    
    # ... other scaling methods ...

# Optional: Scaling presets
SCALING_PRESETS = {
    'balanced': {
        'method': 'sigmoid',
        'params': {'steepness': 2.0, 'center': 1.0, 'r_min': 0.2, 'r_max': 1.0}
    },
    'detail_near_zero': {
        'method': 'hybrid',
        'params': {'transition': 0.5, 'r_min': 0.2, 'r_max': 1.0}
    },
    'auto': {
        'method': 'adaptive',
        'params': {'low_percentile': 5, 'high_percentile': 95, 'r_min': 0.2, 'r_max': 1.0}
    }
}

def get_scaling_preset(name: str) -> dict:
    """Get a predefined scaling configuration."""
    if name not in SCALING_PRESETS:
        raise ValueError(f"Unknown preset: {name}")
    return SCALING_PRESETS[name].copy()
```

**Benefits:**
- Makes ModulusScaling a first-class concept
- Easier to find and use
- Can be extended without modifying mesh_utils
- Supports preset system

### 1.4 Implement Data Transformation Pipeline [★★★★★]

**Problem:** Data transformation (domain → function evaluation → colors) is implicit and scattered.

**Solution:** Create an explicit data transformation pipeline.

**Action Items:**
```python
# complexplorer/pipeline.py
from typing import Protocol, Tuple
import numpy as np
from .core.scaling import ModulusScaling

class TransformationPipeline:
    """Manages the data transformation pipeline."""
    
    def __init__(self):
        self.transforms = []
    
    def add_transform(self, transform: Callable):
        """Add a transformation step."""
        self.transforms.append(transform)
        return self
    
    def execute(self, domain: Domain, func: Callable, cmap: Cmap) -> dict:
        """Execute the full pipeline."""
        # Step 1: Generate mesh points
        z = domain.mesh()
        
        # Step 2: Evaluate function
        w = self._safe_evaluate(func, z)
        
        # Step 3: Handle infinities
        w = self._handle_infinities(w)
        
        # Step 4: Generate colors
        colors = cmap.rgb(w)
        
        # Step 5: Apply custom transforms
        data = {'z': z, 'w': w, 'colors': colors}
        for transform in self.transforms:
            data = transform(data)
        
        return data

class MeshTransformationPipeline(TransformationPipeline):
    """Extended pipeline for 3D mesh generation (including STL)."""
    
    def __init__(self, scaling_method: str = 'constant'):
        super().__init__()
        self.scaling_method = scaling_method
    
    def execute_3d(self, domain: Domain, func: Callable, 
                   scaling_params: dict = None) -> dict:
        """Execute pipeline for 3D mesh generation."""
        data = super().execute(domain, func, None)
        
        # Apply modulus scaling using the extracted ModulusScaling class
        moduli = np.abs(data['w'])
        scaler = getattr(ModulusScaling, self.scaling_method)
        data['radii'] = scaler(moduli, **(scaling_params or {}))
        
        # Generate 3D mesh
        data['mesh'] = self._generate_3d_mesh(data)
        
        return data
```

**Benefits:**
- Makes data flow explicit
- Enables custom transformations
- Improves testability
- Supports both visualization and STL export
- Uses the extracted ModulusScaling class

## Priority 2: Module Restructuring (High Impact, Medium Complexity)

### 2.1 Split Large Modules [★★★★☆]

**Problem:** plots_3d_pyvista.py is 983 lines - too large and handling too many responsibilities.

**Solution:** Split into focused modules.

**New Structure:**
```
complexplorer/
├── plotting/
│   ├── __init__.py
│   ├── base.py              # Abstract base classes
│   ├── matplotlib/
│   │   ├── __init__.py
│   │   ├── plot_2d.py       # 2D plotting functions
│   │   ├── plot_3d.py       # 3D plotting functions
│   │   └── utils.py         # Matplotlib-specific utilities
│   └── pyvista/
│       ├── __init__.py
│       ├── plot_3d.py       # 3D landscape plots
│       ├── riemann.py       # Riemann sphere visualization
│       └── utils.py         # PyVista-specific utilities
├── export/
│   ├── __init__.py
│   ├── base.py              # Base export functionality
│   ├── image.py             # Image export (PNG, JPG, etc.)
│   ├── vector.py            # Vector export (SVG, PDF)
│   ├── interactive.py       # Interactive export (HTML)
│   └── stl/
│       ├── __init__.py
│       ├── generator.py     # STL generation (from current ornament_generator.py)
│       ├── healing.py       # Mesh healing (from spherical_healing.py)
│       └── utils.py         # STL-specific utilities
├── core/
│   ├── __init__.py
│   ├── domain.py            # Domain classes
│   ├── colormap.py          # Color mapping classes (from cmap.py)
│   ├── scaling.py           # Modulus scaling methods (extracted from mesh_utils.py)
│   └── functions.py         # Mathematical functions (from funcs.py)
├── utils/
│   ├── __init__.py
│   ├── validation.py        # Validation utilities
│   ├── mesh.py             # Mesh generation utilities (remaining from mesh_utils.py)
│   └── transforms.py        # Data transformations
└── __init__.py              # Clean public API
```

**Migration Notes:**
- `ModulusScaling` class moves from `mesh_utils.py` to `core/scaling.py`
- Mesh generation functions remain in `utils/mesh.py`
- Stereographic projection functions move to `core/functions.py`

**Benefits:**
- Better organization
- Easier navigation
- Focused modules
- Clear separation of export functionality
- ModulusScaling becomes a core concept

### 2.2 Enhance STL Export Integration [★★★★☆]

**Problem:** STL export is isolated and not well-integrated with the rest of the codebase.

**Solution:** Better integrate STL export with the plotting pipeline.

**Action Items:**
```python
# complexplorer/export/stl/generator.py
from ..base import BaseExporter
from ...plotting.base import Base3DPlotter
from ...core.scaling import ModulusScaling

class STLExporter(BaseExporter):
    """Enhanced STL exporter integrated with plotting pipeline."""
    
    def __init__(self, plotter: Base3DPlotter):
        self.plotter = plotter
        self.healer = MeshHealer()
    
    def export(self, filename: str, size_mm: float = 50, 
               wall_thickness: float = 2, **kwargs):
        """Export plot to STL file."""
        # Use data from plotter
        mesh_data = self.plotter.get_mesh_data()
        
        # Apply modulus scaling if needed
        if 'scaling' in kwargs:
            scaling_method = getattr(ModulusScaling, kwargs['scaling'])
            mesh_data = self._apply_scaling(mesh_data, scaling_method)
        
        # Heal mesh for 3D printing
        healed_mesh = self.healer.heal(mesh_data)
        
        # Scale and export
        scaled_mesh = self._scale_to_size(healed_mesh, size_mm)
        scaled_mesh.save(filename, binary=kwargs.get('binary', True))

# Usage
from complexplorer import RiemannSpherePlotter, STLExporter

plotter = RiemannSpherePlotter(func=lambda z: z**2)
plotter.plot()

exporter = STLExporter(plotter)
exporter.export("output.stl", size_mm=60)
```

### 2.3 Complete STL Export Features [★★★☆☆]

**Problem:** STL export has incomplete features (TODO in boundary loop extraction).

**Solution:** Complete the implementation and add missing features.

**Action Items:**
```python
# complexplorer/export/stl/healing.py
def extract_boundary_loops(mesh: pv.PolyData) -> List[np.ndarray]:
    """Extract properly ordered boundary loops from mesh edges."""
    # Complete the TODO implementation
    edges = mesh.extract_feature_edges(
        boundary_edges=True,
        non_manifold_edges=False,
        feature_edges=False
    )
    
    # Use connectivity to extract ordered loops
    connectivity = edges.connectivity()
    loops = []
    
    for region_id in range(connectivity.n_arrays):
        region = connectivity.threshold(region_id, region_id)
        # Order points to form a proper loop
        ordered_points = order_boundary_points(region.points)
        loops.append(ordered_points)
    
    return loops
```

## Priority 3: Testing and Documentation (High Impact, Low Complexity)

### 3.1 Add STL Export Tests [★★★★★]

**Problem:** No tests exist for STL export functionality.

**Solution:** Create comprehensive test suite.

**Action Items:**
```python
# tests/unit/test_stl_export.py
import pytest
import numpy as np
from complexplorer.stl_export import OrnamentGenerator
from complexplorer.core.scaling import ModulusScaling
from complexplorer import Rectangle, Disk
import pyvista as pv

class TestSTLExport:
    """Test suite for STL export functionality."""
    
    def test_basic_generation(self, tmp_path):
        """Test basic STL generation."""
        gen = OrnamentGenerator(lambda z: z**2)
        output = tmp_path / "test.stl"
        gen.generate_ornament(str(output))
        
        # Verify file exists and is valid
        assert output.exists()
        mesh = pv.read(str(output))
        assert mesh.n_points > 0
        assert mesh.is_manifold
    
    def test_with_domain(self, tmp_path):
        """Test STL generation with domain restriction."""
        gen = OrnamentGenerator(
            lambda z: 1/z,
            domain=Disk(2)
        )
        output = tmp_path / "test_domain.stl"
        gen.generate_ornament(str(output))
        
        mesh = pv.read(str(output))
        assert mesh.is_manifold
    
    def test_scaling_methods(self, tmp_path):
        """Test different scaling methods."""
        for scaling in ['constant', 'arctan', 'adaptive']:
            gen = OrnamentGenerator(
                lambda z: (z-1)/(z+1),
                scaling=scaling
            )
            output = tmp_path / f"test_{scaling}.stl"
            gen.generate_ornament(str(output))
            
            mesh = pv.read(str(output))
            assert mesh.is_manifold
    
    def test_modulus_scaling_integration(self):
        """Test that STL export uses the core ModulusScaling class."""
        moduli = np.array([0, 1, 2, 10])
        
        # Test that scaling methods produce expected results
        constant = ModulusScaling.constant(moduli, radius=1.0)
        assert np.all(constant == 1.0)
        
        arctan = ModulusScaling.arctan(moduli)
        assert arctan[0] < arctan[1] < arctan[2] < arctan[3]
        assert np.all(arctan >= 0.5) and np.all(arctan <= 1.5)
    
    def test_mesh_healing(self):
        """Test mesh healing algorithms."""
        # Create mesh with holes
        mesh_with_holes = create_test_mesh_with_holes()
        
        healer = SphericalMeshHealer()
        healed = healer.heal_mesh(mesh_with_holes)
        
        assert healed.is_manifold
        assert healed.n_open_edges == 0
    
    def test_size_validation(self):
        """Test size parameter validation."""
        gen = OrnamentGenerator(lambda z: z)
        
        with pytest.raises(ValueError):
            gen.generate_ornament("test.stl", size_mm=-10)
        
        with pytest.raises(ValueError):
            gen.generate_ornament("test.stl", wall_thickness=30, size_mm=50)
```

**Benefits:**
- Ensures STL export reliability
- Catches regressions
- Documents expected behavior
- Tests integration with core modules

### 3.2 Update Documentation [★★★☆☆]

**Problem:** README.md has outdated STL export API documentation.

**Solution:** Update all documentation.

**Action Items:**
1. Update README.md with correct API
2. Add STL export section to main documentation
3. Create STL export tutorial
4. Document PyVista dependency requirement
5. Document ModulusScaling as a core concept

## Priority 4: API Design Improvements (Medium Impact, High Value)

### 4.1 Unified Export Interface [★★★★☆]

**Problem:** Different export types have different interfaces.

**Solution:** Create unified export API.

**Action Items:**
```python
# complexplorer/__init__.py
from .core.scaling import ModulusScaling, get_scaling_preset

def export(plot, filename: str, **kwargs):
    """Unified export interface."""
    ext = filename.split('.')[-1].lower()
    
    if ext == 'stl':
        from .export.stl import STLExporter
        exporter = STLExporter(plot)
    elif ext in ['png', 'jpg', 'jpeg']:
        from .export.image import ImageExporter
        exporter = ImageExporter(plot)
    elif ext == 'html':
        from .export.interactive import HTMLExporter
        exporter = HTMLExporter(plot)
    else:
        raise ValueError(f"Unsupported export format: {ext}")
    
    return exporter.export(filename, **kwargs)

# Usage
import complexplorer as cp

plot = cp.riemann_sphere(lambda z: z**2)
cp.export(plot, "ornament.stl", size_mm=60)
cp.export(plot, "image.png", dpi=300)
cp.export(plot, "interactive.html")
```

### 4.2 Expose Core Concepts at Package Level [★★★☆☆]

**Problem:** Important concepts like ModulusScaling require deep imports.

**Solution:** Add to main package API.

**Action Items:**
```python
# complexplorer/__init__.py
# Core concepts
from .core.domain import Domain, Rectangle, Disk, Annulus
from .core.colormap import Cmap, Phase, Chessboard
from .core.scaling import ModulusScaling, get_scaling_preset
from .core.functions import sawtooth, stereographic_projection

# Plotting functions
from .plotting import plot, plot3d, riemann_sphere

# Export functionality
from .export.stl import OrnamentGenerator, generate_ornament
from .export import export

# Convenience function for STL
def generate_ornament(func, filename: str, **kwargs):
    """Generate 3D-printable ornament from complex function."""
    # Can use scaling presets
    if 'scaling_preset' in kwargs:
        preset = get_scaling_preset(kwargs.pop('scaling_preset'))
        kwargs.update(preset)
    
    gen = OrnamentGenerator(func, **kwargs)
    return gen.generate_ornament(filename)
```

## Priority 5: Performance and Quality Improvements (Low Priority, High Value)

### 5.1 Optimize Mesh Generation [★★☆☆☆]

**Problem:** Mesh generation could be more efficient.

**Solution:** Cache and reuse base meshes.

**Action Items:**
```python
# complexplorer/utils/mesh_cache.py
class MeshCache:
    """Cache for commonly used base meshes."""
    
    def __init__(self):
        self._cache = {}
    
    def get_sphere_mesh(self, resolution: int) -> pv.PolyData:
        """Get or create sphere mesh with given resolution."""
        if resolution not in self._cache:
            self._cache[resolution] = pv.Sphere(
                theta_resolution=resolution,
                phi_resolution=resolution
            )
        return self._cache[resolution].copy()
```

## Implementation Timeline

### Phase 1 (Weeks 1-2): Foundation
1. Extract ModulusScaling to core/scaling.py
2. Create validation module (including STL validation)
3. Implement abstract base classes (including BaseExporter)
4. Set up new directory structure

### Phase 2 (Weeks 3-4): Core Refactoring
1. Split large modules
2. Move STL export to new structure
3. Implement data pipeline with ModulusScaling integration

### Phase 3 (Week 5): STL Export Enhancement
1. Complete boundary loop extraction
2. Add comprehensive STL tests
3. Update STL documentation

### Phase 4 (Week 6): Integration
1. Create unified export interface
2. Integrate STL with plotting pipeline
3. Add core concepts to main API

### Phase 5 (Week 7): Quality Improvements
1. Fix naming conventions
2. Optimize mesh generation
3. Implement caching

### Phase 6 (Week 8): Testing and Documentation
1. Update all tests
2. Create migration guide
3. Update examples including STL demos

## Breaking Changes

The following changes will break backward compatibility:

1. **Module structure**: Import paths will change
   - Old: `from complexplorer.mesh_utils import ModulusScaling`
   - New: `from complexplorer.core.scaling import ModulusScaling`
   - Or: `from complexplorer import ModulusScaling` (recommended)

2. **STL imports**:
   - Old: `from complexplorer.stl_export import OrnamentGenerator`
   - New: `from complexplorer.export.stl import OrnamentGenerator`
   - Or: `from complexplorer import OrnamentGenerator` (recommended)

3. **Function signatures**: Parameter names will be standardized
4. **Error types**: New exception hierarchy
5. **API organization**: Some functions will move to submodules

## Migration Guide

```python
# Old way
from complexplorer.stl_export import OrnamentGenerator
from complexplorer.mesh_utils import ModulusScaling
gen = OrnamentGenerator(f)
gen.generate_ornament("output.stl")

# New way (option 1 - direct)
import complexplorer as cp
cp.generate_ornament(f, "output.stl", size_mm=60, scaling_preset='balanced')

# New way (option 2 - integrated)
plot = cp.riemann_sphere(f)
cp.export(plot, "output.stl", size_mm=60)

# New way (option 3 - advanced with ModulusScaling)
from complexplorer import OrnamentGenerator, ModulusScaling
gen = OrnamentGenerator(f, scaling='adaptive')
gen.generate_ornament("output.stl")

# New way (option 4 - custom scaling)
from complexplorer import ModulusScaling
custom_scaling = lambda m: ModulusScaling.sigmoid(m, steepness=3.0)
plot = cp.riemann_sphere(f, scaling='custom', scaling_func=custom_scaling)
```

## Success Metrics

1. **Code reduction**: ~30% fewer lines through deduplication
2. **Module size**: No module larger than 300 lines
3. **Test coverage**: Maintain >90% coverage (including STL export)
4. **Performance**: No regression in speed
5. **API simplicity**: Reduce required imports by 50%
6. **STL quality**: All exported STLs pass manifold checks
7. **Core concepts**: ModulusScaling easily accessible and extendable

## Conclusion

This improvement plan addresses the major architectural and code quality issues in complexplorer while preserving and enhancing the valuable STL export functionality. The modular structure will enable future enhancements while the simplified API will improve user experience. Key improvements include:

- ModulusScaling becomes a first-class core concept
- STL export is better integrated with the plotting pipeline
- Clear separation of concerns across modules
- Unified export interface for all file types
- Comprehensive testing including STL functionality

The plan ensures that the sophisticated mesh healing capabilities are preserved while making the codebase more maintainable and extensible.