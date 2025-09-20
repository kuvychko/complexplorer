# Complexplorer v2.0 Migration Guide

## Overview

Complexplorer v2.0 is a major release that prioritizes API cleanliness and simplicity. This guide will help you migrate your code from v1.x to v2.0.

## Breaking Changes

### 1. API Function Renames

#### Main plotting function
```python
# Old (v1.x)
import complexplorer as cp
cp.quick_plot(func, domain=rect, mode='2d')

# New (v2.0)
import complexplorer as cp
cp.plot(func, rect, mode='2d')
```

### 2. Removed Functions

The following incomplete functions have been removed:
- `analyze_function()` - Use `plot()` directly with appropriate colormap
- `create_animation()` - Will be added in a future release  
- `compare_functions()` - Will be added in a future release
- `visualize` and `explore` aliases - Use `plot()` directly

### 3. Preset API Changes

```python
# Old (v1.x)
from complexplorer import Presets
config = Presets.publication_ready()

# New (v2.0)
from complexplorer import publication_preset
config = publication_preset()
```

Available presets:
- `publication_preset()` - High-quality publication figures
- `interactive_preset()` - Fast interactive exploration
- `high_contrast_preset()` - Maximum phase contrast

### 4. Import Changes

All imports are now absolute for clarity:

```python
# Old (relative imports in internal code)
from ..core.domain import Domain

# New (absolute imports everywhere)
from complexplorer.core.domain import Domain
```

### 5. Base Classes Removed

The unused base plotter classes have been removed. This is an internal change that shouldn't affect most users.

### 6. Export API Changes

```python
# Old (v1.x) 
from complexplorer.export import STLExporter

# New (v2.0)
from complexplorer.export import OrnamentGenerator, create_ornament
```

## Common Migration Patterns

### Basic Plotting

```python
# v1.x
import complexplorer as cp

# Quick plot
cp.quick_plot(lambda z: z**2, mode='2d')

# With custom domain
rect = cp.Rectangle(4, 4)
cp.quick_plot(lambda z: (z-1)/(z+1), domain=rect)

# Using aliases
cp.visualize(func, domain=rect)
```

```python
# v2.0
import complexplorer as cp

# Simple plot (name is clearer)
cp.plot(lambda z: z**2, mode='2d')

# With custom domain (cleaner parameter order)
rect = cp.Rectangle(4, 4) 
cp.plot(lambda z: (z-1)/(z+1), rect)

# No aliases - one clear function name
cp.plot(func, rect)
```

### Using Presets

```python
# v1.x
from complexplorer import Presets, quick_plot

config = Presets.publication_ready()
quick_plot(func, **config)
```

```python
# v2.0
from complexplorer import publication_preset, plot

config = publication_preset()
plot(func, **config)
```

### Direct Module Imports

```python
# v1.x - May have worked but not guaranteed
from complexplorer.plotting.matplotlib.plot_2d import plot

# v2.0 - Clear, absolute imports
from complexplorer.plotting import plot
# or
from complexplorer.plotting.matplotlib.plot_2d import plot
```

## Features Postponed to Future Releases

The following features were incomplete in v1.x and have been removed from v2.0. They will be properly implemented in future releases:

1. **Animation support** - Creating animations of parametric families
2. **Function comparison** - Side-by-side comparison plots  
3. **Automatic zero/pole detection** - Mathematical feature analysis

## New in v2.0

### Cleaner API Surface
- Fewer functions, each doing one thing well
- Clear, descriptive function names
- No redundant aliases

### Better Type Safety
- Complete type hints for all public functions
- Better IDE support and autocomplete

### Performance Improvements
- Optimized imports and module loading
- Cleaner internal structure

## Getting Help

If you encounter issues during migration:

1. Check this guide for the specific function/feature
2. Review the updated examples in the `examples/` directory
3. Report issues at: https://github.com/anthropics/claude-code/issues

## Version Compatibility

To check your version:
```python
import complexplorer
print(complexplorer.__version__)  # Should be 2.0.0
```

## Summary

The main philosophy of v2.0 is **"less is more"** - a cleaner, simpler API that does complex function visualization excellently. While some features have been removed, the core functionality is stronger and more maintainable.