# Complexplorer v2.0 Task Checklist

## Phase 1: Slash and Burn (Clean up technical debt)

### TODO Comments Cleanup
- [ ] Remove TODO in `api.py:112` - Add automatic zero/pole detection
- [ ] Remove TODO in `api.py:152` - Implement animation functionality  
- [ ] Remove TODO in `api.py:186` - Implement comparison plots
- [ ] Remove TODO in `plot_2d.py:21,24` - Implement base class inheritance
- [ ] Remove TODO in `plot_3d.py:22` - Implement base class inheritance

### Incomplete API Functions
- [ ] Delete or complete `analyze_function()` in api.py
- [ ] Delete or complete `create_animation()` in api.py
- [ ] Delete or complete `compare_functions()` in api.py  
- [ ] Fix incomplete preset functions (`publication_ready`, `interactive`, `high_contrast`)
- [ ] Rename `quick_plot()` to `plot()` - make it the default

### Base Class Architecture
- [ ] Decision: Keep or remove base classes in `plotting/base.py`
- [ ] If keeping: Implement inheritance for `Matplotlib2DPlotter`
- [ ] If keeping: Implement inheritance for `Matplotlib3DPlotter`
- [ ] If keeping: Implement inheritance for PyVista plotters
- [ ] Delete or implement `PlotConfig` class properly

## Phase 2: Type Safety & Consistency

### Type Hints (23 functions missing return types)
- [ ] `api.py:27:quick_plot` - add return type
- [ ] `api.py:70:analyze_function` - add return type
- [ ] `api.py:119:create_animation` - add return type
- [ ] `api.py:156:compare_functions` - add return type
- [ ] `api.py:195:publication_ready` - add return type
- [ ] `api.py:203:interactive` - add return type
- [ ] `api.py:211:high_contrast` - add return type
- [ ] `color_utils.py:61:linear_to_srgb` - add return type
- [ ] `domain.py:506:tight_bounds` - add return type
- [ ] `export/base.py:82:set_metadata` - add return type
- [ ] Add remaining 13 return type hints
- [ ] Add `py.typed` marker file

### Import Standardization
- [ ] Convert all relative imports to absolute imports
- [ ] Fix mixed import patterns in `core/` modules
- [ ] Fix mixed import patterns in `plotting/` modules  
- [ ] Fix mixed import patterns in `export/` modules
- [ ] Update all `__init__.py` files with proper `__all__` definitions

### Parameter Consistency
- [ ] Rename `n_phi` → `phase_sectors` throughout
- [ ] Rename `n_theta` → `latitude_divisions` for sphere
- [ ] Remove `z` and `f` parameters - use only `domain` and `func`
- [ ] Standardize colormap parameter names across all 13 implementations
- [ ] Audit and remove unused **kwargs throughout

## Phase 3: Module Structure Simplification

### Flatten Structure
- [ ] Rename `funcs.py` → `math_functions.py` for clarity
- [ ] Consider merging small submodules with single files
- [ ] Move validation utilities to core if used everywhere
- [ ] Consolidate export functionality if minimal

### Clean Separation
- [ ] Ensure no cross-subpackage imports
- [ ] Clear module boundaries - one responsibility per module
- [ ] Remove circular import possibilities

## Phase 4: Performance Optimization

### Critical Path Optimization
- [ ] Profile Riemann sphere mesh generation
- [ ] Optimize rectangular mesh generation bottleneck
- [ ] Implement mesh caching where appropriate
- [ ] Vectorize any remaining Python loops over arrays

### Memory Efficiency
- [ ] Avoid unnecessary array copies
- [ ] Use views where possible
- [ ] Implement lazy evaluation for expensive computations

## Phase 5: Validation & Error Handling

### Validation Framework
- [ ] Create `@validate_domain` decorator
- [ ] Create `@validate_function` decorator  
- [ ] Create `@validate_colormap` decorator
- [ ] Implement consistent validation across all entry points
- [ ] Clear, actionable error messages

### Error Strategy
- [ ] Use exceptions for user errors
- [ ] Use assertions for programmer errors
- [ ] Remove defensive programming where types guarantee safety
- [ ] Fail fast with clear messages

## Phase 6: Documentation & Testing

### Documentation
- [ ] Ensure every public function has a docstring
- [ ] Add mathematical notation in LaTeX where needed
- [ ] One clear example per major function
- [ ] Consistent docstring format (NumPy style)

### Testing Strategy  
- [ ] Remove redundant tests
- [ ] Add property-based tests for invariants
- [ ] Create performance benchmark suite
- [ ] Ensure 100% coverage of public API

## Phase 7: Final Cleanup

### Code Quality
- [ ] Remove all dead code
- [ ] No commented-out code blocks
- [ ] Consistent naming conventions
- [ ] PEP 8 compliance throughout

### API Surface
- [ ] Review and minimize public API
- [ ] Ensure each function does one thing well
- [ ] Remove rarely-used optional parameters
- [ ] Document all breaking changes

### Release Preparation
- [ ] Update version to 2.0.0
- [ ] Write comprehensive CHANGELOG
- [ ] Create migration guide with examples
- [ ] Update all example notebooks
- [ ] Ensure all tests pass
- [ ] Performance validation against v1.x

## Metrics to Track

- [ ] Lines of code reduced by ≥20%
- [ ] Zero TODO comments remaining
- [ ] 100% type hint coverage on public API
- [ ] Riemann sphere generation 2x faster
- [ ] All tests passing
- [ ] No circular imports
- [ ] Clean mypy output