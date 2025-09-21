# Complexplorer Technical Implementation Tasks

## Implementation Status: ✅ COMPLETED (2025-09-21)

The following high-priority features have been successfully implemented:
- ✅ Phase 1: API Simplification (all tasks)
- ✅ Phase 2: Colormap Enhancements (OklabPhase, unit circle emphasis)
- ✅ Phase 4: Utility Functions (all tasks)
- ✅ Comprehensive unit tests (55 tests passing)

Lower priority features (Phase 2 contours, Phase 3) were not implemented and remain for future work.

## Overview
This document provides a detailed task breakdown for implementing **technical improvements** identified in the complexplorer vs cplot analysis. Documentation and tutorial tasks have been separated into DOCUMENTATION_TASKS.md and should be addressed after technical implementation is complete and tested.

## Phase 1: API Simplification (Week 1)
**Goal:** Lower barrier to entry with simpler API while maintaining backward compatibility

### Task 1.1: Implement `show()` Convenience Function
**Priority:** HIGH  
**Estimated Time:** 2 hours  
**Location:** `complexplorer/api.py` (new section)

**Subtasks:**
- [ ] Create `show()` function signature matching cplot's simplicity
- [ ] Accept x_range and y_range tuples with resolution as third element
- [ ] Convert range tuples to Rectangle domain internally
- [ ] Pass through all kwargs to underlying `plot()` function
- [ ] Add default colormap (Phase) if none specified
- [ ] Write docstring with clear examples

**Acceptance Criteria:**
- Function works with minimal arguments: `cp.show(lambda z: z**2)`
- Supports range specification: `cp.show(f, (-2, 2, 400), (-2, 2, 400))`
- All existing plot kwargs work transparently
- No breaking changes to existing API

**Implementation:**
```python
def show(func, x_range=(-2, 2, 400), y_range=None, **kwargs):
    """
    Quick plot function for simple use cases, similar to cplot.
    
    Parameters
    ----------
    func : callable
        Complex function to visualize
    x_range : tuple
        (min, max, resolution) for real axis
    y_range : tuple, optional
        (min, max, resolution) for imaginary axis. If None, uses x_range
    **kwargs
        Additional arguments passed to plot()
    
    Examples
    --------
    >>> cp.show(lambda z: z**2)
    >>> cp.show(lambda z: 1/z, (-3, 3, 500), (-3, 3, 500))
    """
```

### Task 1.2: Add Range-Based Domain Creation Helper
**Priority:** HIGH  
**Estimated Time:** 1 hour  
**Location:** `complexplorer/core/domain.py`

**Subtasks:**
- [ ] Add `Rectangle.from_ranges()` class method
- [ ] Support (min, max) and (min, max, resolution) tuples
- [ ] Calculate appropriate center and dimensions
- [ ] Add input validation
- [ ] Write unit tests

**Acceptance Criteria:**
- Creates Rectangle from range specifications
- Handles both 2-tuple and 3-tuple inputs
- Validates input ranges (min < max)
- Covered by unit tests

### Task 1.3: Improve Default Parameters
**Priority:** MEDIUM  
**Estimated Time:** 2 hours  
**Location:** Multiple files

**Subtasks:**
- [ ] Review all plot function defaults
- [ ] Set sensible defaults for resolution (400-500)
- [ ] Default to Phase colormap with auto_scale_r=True for better initial experience
- [ ] Document default choices in docstrings
- [ ] Update examples to show both simple and advanced usage

**Acceptance Criteria:**
- New users get good results with minimal configuration
- Defaults produce publication-quality output
- Documentation clearly states defaults

## Phase 2: Colormap Enhancements (Week 2)
**Goal:** Enhance mathematical rigor with better contours and pure OKLAB implementation

### Task 2.1: Implement Pure OKLAB Colormap
**Priority:** HIGH  
**Estimated Time:** 4 hours  
**Location:** `complexplorer/core/colormap.py`

**Subtasks:**
- [ ] Create `OklabPhase` class inheriting from `Colormap`
- [ ] Implement cylindrical OKLAB (not OkLCh) conversion
- [ ] Match cplot's OKLAB implementation approach
- [ ] **CRITICAL: Implement enhanced phase/modulus visualization using sawtooth function**
  - [ ] Support modulus variation through Value (brightness) modulation
  - [ ] Use sawtooth function to create discontinuous edges for better perception
  - [ ] Allow toggling enhanced mode on/off (default: on)
  - [ ] Follow Phase colormap implementation pattern for consistency
- [ ] Support all standard enhanced phase features (n_phi, auto_scale_r)
- [ ] Add comprehensive docstring with mathematical background
- [ ] Create unit tests comparing against expected values
- [ ] Add to __init__.py exports

**Acceptance Criteria:**
- Produces perceptually uniform colors
- Matches cplot's OKLAB behavior for direct comparison
- **Enhanced mode creates clear visual discontinuities at modulus boundaries**
- **Sawtooth-based Value modulation matches other complexplorer colormaps**
- Supports all enhanced phase portrait features
- Can toggle between smooth (cplot-like) and enhanced (complexplorer) modes
- Passes unit tests for color conversion accuracy

**Implementation Notes:**
- Use direct OKLAB color space, not OkLCh
- **IMPORTANT: The enhanced phase/modulus visualization is a key differentiator**
  - Human perception struggles with smoothly varying colors
  - Discontinuous Value edges (via sawtooth) dramatically improve structure visibility
  - This is a major advantage of complexplorer over cplot
  - Refer to Phase colormap implementation for sawtooth usage pattern
- Ensure smooth transitions at phase boundaries
- Consider performance optimizations for large arrays
- Enhanced mode should be default but easily disabled for pure OKLAB comparison

### Task 2.2: Add Unit Circle Emphasis to Phase Colormap
**Priority:** HIGH  
**Estimated Time:** 3 hours  
**Location:** `complexplorer/core/colormap.py`

**Subtasks:**
- [ ] Add `emphasize_unit_circle` parameter to Phase class
- [ ] Implement brightness boost/reduction near |z|=1
- [ ] Implement particular color emphasis near |z|=1
- [ ] Make emphasis strength configurable
- [ ] Default to True for mathematical functions
- [ ] Update existing enhanced phase logic to incorporate emphasis
- [ ] Write tests for unit circle detection

**Acceptance Criteria:**
- Unit circle clearly visible in phase portraits
- Emphasis strength is adjustable
- Works with auto_scale_r feature
- No visual artifacts at emphasis boundaries

### Task 2.3: Enhance Default Contour Visibility
**Priority:** HIGH  
**Estimated Time:** 2 hours  
**Location:** `complexplorer/core/colormap.py`

**Subtasks:**
- [ ] Review current contour generation algorithm
- [ ] Increase default contour line width/contrast
- [ ] Add adaptive contrast based on background
- [ ] Implement anti-aliasing for smoother lines
- [ ] Test with various colormaps

**Acceptance Criteria:**
- Contours clearly visible against all backgrounds
- No jarring visual artifacts
- Maintains mathematical accuracy
- Performance impact < 10%

### Task 2.4: Smart Contour Defaults
**Priority:** MEDIUM  
**Estimated Time:** 3 hours  
**Location:** `complexplorer/core/colormap.py`

**Subtasks:**
- [ ] Implement heuristics for contour spacing based on domain size
- [ ] Add special handling for common mathematical points (0, 1, i, -1, -i)
- [ ] Auto-detect appropriate logarithmic vs linear spacing
- [ ] Create preset configurations for common use cases
- [ ] Document smart default behavior

**Acceptance Criteria:**
- Contours automatically adjust to domain scale
- Important mathematical features highlighted
- User can override smart defaults
- Clear documentation of heuristic rules

## Phase 3: Enhanced Contour System (Week 3)
**Goal:** Improve contour generation and configurability

### Task 3.1: Refactor Contour Generation
**Priority:** MEDIUM  
**Estimated Time:** 4 hours  
**Location:** `complexplorer/core/colormap.py`

**Subtasks:**
- [ ] Extract contour logic into separate module
- [ ] Create `ContourGenerator` class
- [ ] Support multiple contour types (arg, abs, mixed)
- [ ] Implement efficient contour detection algorithm
- [ ] Add caching for repeated calculations

**Acceptance Criteria:**
- Cleaner code organization
- Faster contour generation
- Supports complex contour patterns
- Backward compatible with existing API

### Task 3.2: Add Configurable Contour Levels
**Priority:** MEDIUM  
**Estimated Time:** 2 hours  
**Location:** `complexplorer/core/colormap.py`

**Subtasks:**
- [ ] Add `arg_levels` parameter for specific phase contours
- [ ] Add `abs_levels` parameter for specific magnitude contours
- [ ] Support both absolute values and relative spacing
- [ ] Implement level validation and sorting
- [ ] Update documentation

**Acceptance Criteria:**
- Users can specify exact contour locations
- Both linear and custom spacing supported
- Clear error messages for invalid inputs
- Examples show common use cases

### Task 3.3: Special Mathematical Contours
**Priority:** LOW  
**Estimated Time:** 3 hours  
**Location:** `complexplorer/core/colormap.py`

**Subtasks:**
- [ ] Add preset for branch cut visualization
- [ ] Add preset for Riemann sheets
- [ ] Implement critical point emphasis
- [ ] Add flow line visualization option
- [ ] Document mathematical significance

**Acceptance Criteria:**
- Mathematical features clearly highlighted
- Presets accessible via simple API
- Documentation explains mathematical concepts
- Visual output matches mathematical expectations

## Phase 4: Utility Functions (Week 4)
**Goal:** Add helpful utility functions for common tasks

### Task 4.1: Direct Color Export Function
**Priority:** LOW  
**Estimated Time:** 1 hour  
**Location:** `complexplorer/utils/color.py` (new file)

**Subtasks:**
- [ ] Create `get_color(z, cmap)` function
- [ ] Support scalar and array inputs
- [ ] Return RGB tuples/arrays
- [ ] Add HSV export option
- [ ] Write unit tests

**Acceptance Criteria:**
- Works with all colormap types
- Handles edge cases (infinity, NaN)
- Performance optimized for arrays
- Clear documentation with examples

### Task 4.2: Add Mathematical Special Functions
**Priority:** LOW  
**Estimated Time:** 2 hours  
**Location:** `complexplorer/special.py` (new file)

**Subtasks:**
- [ ] Wrap scipy special functions with complex support
- [ ] Add error handling for unavailable functions
- [ ] Create function registry
- [ ] Add documentation with mathematical background
- [ ] Include usage examples

**Acceptance Criteria:**
- Graceful fallback if scipy not available
- Clear naming convention
- Proper branch cut handling
- Mathematical documentation included


## Testing Requirements

### Unit Tests
Each new feature requires corresponding unit tests:
- [ ] Test `show()` function with various inputs
- [ ] Test OKLAB color conversion accuracy
- [ ] Test contour generation with edge cases
- [ ] Test range-based domain creation

### Integration Tests
- [ ] Test new API with existing examples
- [ ] Ensure backward compatibility
- [ ] Test colormap combinations

### Performance Tests
- [ ] Benchmark new contour algorithms
- [ ] Measure OKLAB colormap performance
- [ ] Profile memory usage for large domains

## Success Metrics

1. **API Simplicity**: New users can create visualization in < 3 lines of code
2. **Performance**: No degradation in rendering speed
3. **Quality**: OKLAB colormap matches cplot's perceptual uniformity
4. **Code Quality**: Clean, well-structured implementation
5. **Testing**: >95% code coverage maintained
6. **Backward Compatibility**: All existing code continues to work

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| OKLAB implementation differs from cplot | Medium | Create validation tests against cplot output |
| Performance regression | High | Benchmark before/after each change |
| Breaking API changes | High | Maintain compatibility layer |
| Increased complexity | Medium | Keep simple API separate from advanced |

## Dependencies

- No new required dependencies
- Optional scipy for special functions
- Existing matplotlib, numpy, pyvista dependencies

## Timeline Summary

- **Week 1**: API Simplification (Tasks 1.1-1.3)
- **Week 2**: Colormap Enhancements (Tasks 2.1-2.4)
- **Week 3**: Enhanced Contours (Tasks 3.1-3.3)
- **Week 4**: Utility Functions (Tasks 4.1-4.2)

## Notes

- Triangulated mesh support explicitly excluded (causes PyVista artifacts)
- Focus on maintaining complexplorer's unique strengths (3D, STL export)
- API simplification should not compromise advanced features
- Performance optimization is secondary to correctness
- Documentation tasks separated to DOCUMENTATION_TASKS.md for later phase

## Next Steps

1. Review and approve technical task plan
2. Set up development branch for v2.1.0
3. Begin with Phase 1 high-priority tasks
4. Complete all technical implementation
5. Thorough testing of new features
6. Then proceed to documentation phase (see DOCUMENTATION_TASKS.md)