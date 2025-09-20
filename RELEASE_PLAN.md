# Complexplorer Release Plan v2.0

## Executive Summary
Major release focusing on API cleanliness, architectural consistency, and removing technical debt. Breaking changes are acceptable to achieve a clean, maintainable codebase.

## Core Principles
- **Clean API over backward compatibility** - Remove deprecated patterns, simplify interfaces
- **Explicit over implicit** - Clear function names, obvious parameter meanings
- **Consistency throughout** - Unified patterns across all modules
- **Performance by design** - Remove inefficient patterns, optimize critical paths

## Phase 1: Architectural Cleanup (Priority: Critical)

### 1.1 Complete Base Class Implementation
- **Remove unused base classes** if inheritance adds no value
- **OR implement properly** with shared functionality extracted
- Delete `PlotConfig` if not needed, or fully implement it
- Remove all TODO comments - either implement or delete the feature

### 1.2 Simplify Module Structure
- **Flatten unnecessary nesting** - if submodules have single files, promote them
- **Consolidate related functionality** - merge small related modules
- **Clear separation of concerns** - one module, one responsibility

### 1.3 Clean API Surface
- **Remove incomplete functions** (`analyze_function`, `compare_functions`) 
- **OR complete them properly** with full implementation
- **Rename for clarity** - e.g., `quick_plot` → `plot` (make it the default)
- **Remove redundant parameters** - audit all functions for unused kwargs

## Phase 2: Type Safety & Validation (Priority: High)

### 2.1 Complete Type Annotations
- Add all missing return types (23 functions identified)
- Use proper generic types (not just `Any`)
- Add `py.typed` marker for mypy support
- Consider using `TypedDict` for complex return values

### 2.2 Consistent Validation
- Create validation decorators for common patterns
- Fail fast with clear error messages
- Remove defensive programming where types guarantee safety
- Use `assert` for programmer errors, exceptions for user errors

### 2.3 Import Structure Reform
- **Single import style**: Use absolute imports everywhere
- **Clean public API**: Define `__all__` in every `__init__.py`
- **No circular imports**: Restructure if needed
- **Clear module boundaries**: No cross-subpackage imports

## Phase 3: Performance & Quality (Priority: High)

### 3.1 Performance Optimization
- **Remove mesh generation bottleneck** in Riemann sphere
- **Vectorize all operations** - no Python loops over arrays
- **Lazy evaluation** where appropriate
- **Memory efficiency** - avoid unnecessary copies

### 3.2 Testing Strategy
- **Property-based testing** for mathematical invariants
- **Snapshot testing** for visual outputs
- **Benchmark suite** to prevent performance regressions
- **Remove redundant tests** - quality over quantity

### 3.3 Documentation
- **Docstrings**: Every public function, consistent format
- **Examples**: One clear example per major function
- **Mathematical notation**: LaTeX in docstrings
- **No external docs initially** - code should be self-documenting

## Phase 4: Feature Completion (Priority: Medium)

### 4.1 Core Features Only
- **Remove half-implemented features** entirely
- **Complete only essential functionality**
- Animation can wait for v2.1 if not critical
- Focus on doing fewer things excellently

### 4.2 Colormap Consistency
- Ensure all 13 colormaps follow same pattern
- Consistent parameter names across all
- Same initialization style
- Unified approach to out-of-domain handling

### 4.3 Export Functionality
- **STL export**: Complete or remove
- **Focus on one format done well** rather than many poorly
- Clear API: `export_stl()` not generic `export()`

## Breaking Changes (Embraced)

### API Changes
- `quick_plot()` → `plot()` - make simple things simple
- Remove `z` and `f` parameters - use `domain` and `func` consistently  
- Rename unclear parameters (e.g., `n_phi` → `phase_sectors`)
- Remove optional parameters that are rarely used

### Structural Changes
- Flatten module structure where beneficial
- Move functions to more logical locations
- Rename modules for clarity (e.g., `funcs.py` → `math_functions.py`)

### Behavioral Changes
- Strict validation by default (no silent failures)
- Consistent coordinate systems throughout
- Unified approach to infinity handling

## Implementation Strategy

### Week 1: Slash and Burn
- Delete all TODO comments (implement or remove)
- Remove incomplete features
- Clean up base class situation
- Simplify module structure

### Week 2: Consistency Pass  
- Standardize all imports
- Complete type hints
- Unify parameter names
- Consistent validation

### Week 3: Quality & Performance
- Optimize critical paths
- Add essential tests only
- Clean, minimal documentation
- Performance benchmarks

### Week 4: Polish
- Final API review
- Update all examples
- Ensure all tests pass
- Performance validation

## Success Metrics

- **Code reduction**: Target 20% fewer lines through cleanup
- **API surface**: Fewer public functions, each doing one thing well
- **Performance**: 2x faster Riemann sphere generation
- **Type coverage**: 100% of public API typed
- **Test quality**: Fewer, better tests with higher coverage
- **Zero TODOs**: No incomplete code in release

## Version Strategy

### Version 2.0.0
- Major version bump signals breaking changes
- Clean slate for future development
- Sets precedent for quality over compatibility

### Future Versions
- 2.1: Animation support (if needed)
- 2.2: Additional export formats
- 2.3: Performance optimizations
- 3.0: Next major architectural change

## Migration Guide

### For Users
```python
# Old (v1.x)
cp.quick_plot(func, domain=rect, mode='2d')

# New (v2.0) 
cp.plot(func, rect)  # mode inferred from function
```

### Key Principles
- Explicit imports: `from complexplorer.core import Domain`
- Clear function names: `plot_landscape()` not `plot(..., mode='3d')`
- Consistent parameters: always `func` and `domain`, never `f` and `z`

## Non-Goals

- Backward compatibility
- Feature parity with similar libraries
- Supporting every use case
- Extensive documentation (initially)

## Philosophy

> "Perfection is achieved not when there is nothing more to add, but when there is nothing left to take away." - Antoine de Saint-Exupéry

The goal is a clean, fast, maintainable library that does complex function visualization excellently, not a library that does everything adequately.