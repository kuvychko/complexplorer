# Composite Domain Viewing Window Fix

## Problem

When creating composite domains through set operations (union, intersection, difference), the viewing window is calculated as the bounding box of all constituent domains' windows. This leads to unnecessarily large viewing areas that don't tightly fit the actual domain.

### Example
```python
# Grid of disks - each disk has radius 0.6, but the viewing window 
# encompasses the entire grid extent
domain_grid = create_disk_grid(n=2, spacing=1, radius=0.6)
```

Current behavior: Window spans from (-2-0.6, -2-0.6) to (2+0.6, 2+0.6) ≈ 5.2 x 5.2
Desired behavior: Window should tightly fit the actual disks

## Root Cause

In `CompositeDomain.__init__()`, the viewing window is calculated as:
```python
# Calculate combined bounds
all_reals = [
    domain1.window_real[0], domain1.window_real[1],
    domain2.window_real[0], domain2.window_real[1]
]
all_imags = [
    domain1.window_imag[0], domain1.window_imag[1],
    domain2.window_imag[0], domain2.window_imag[1]
]

real_bounds = (min(all_reals), max(all_reals))
imag_bounds = (min(all_imags), max(all_imags))
```

This takes the union of all bounds regardless of the actual operation.

## Proposed Solution

### Option 1: Tight Bounds Calculation (Recommended)

Add a method to calculate tight bounds by sampling the domain:

```python
def calculate_tight_bounds(self, sample_density: int = 100) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Calculate tight bounds by sampling the actual domain.
    
    Parameters
    ----------
    sample_density : int
        Number of sample points along each axis.
        
    Returns
    -------
    real_bounds, imag_bounds : tuple of tuple
        Tight bounds for the composite domain.
    """
    # Get loose bounds from constituent domains
    real_min = min(self.domain1.window_real[0], self.domain2.window_real[0])
    real_max = max(self.domain1.window_real[1], self.domain2.window_real[1])
    imag_min = min(self.domain1.window_imag[0], self.domain2.window_imag[0])
    imag_max = max(self.domain1.window_imag[1], self.domain2.window_imag[1])
    
    # Create sample grid
    real_samples = np.linspace(real_min, real_max, sample_density)
    imag_samples = np.linspace(imag_min, imag_max, sample_density)
    real_grid, imag_grid = np.meshgrid(real_samples, imag_samples)
    z_samples = real_grid + 1j * imag_grid
    
    # Find points actually in the domain
    mask = self.contains(z_samples.ravel())
    if not np.any(mask):
        # Empty domain, return zero-size bounds
        return (0, 0), (0, 0)
    
    valid_points = z_samples.ravel()[mask]
    
    # Calculate tight bounds with small margin
    margin = 0.1  # 10% margin
    real_extent = np.real(valid_points)
    imag_extent = np.imag(valid_points)
    
    real_range = real_extent.max() - real_extent.min()
    imag_range = imag_extent.max() - imag_extent.min()
    
    real_bounds = (
        real_extent.min() - margin * real_range,
        real_extent.max() + margin * real_range
    )
    imag_bounds = (
        imag_extent.min() - margin * imag_range,
        imag_extent.max() + margin * imag_range
    )
    
    return real_bounds, imag_bounds
```

### Option 2: Operation-Aware Bounds

Calculate bounds based on the set operation:

```python
def __init__(self, domain1, domain2, operation):
    # ... existing code ...
    
    if operation == 'intersection':
        # Intersection can only be within overlap
        real_bounds = (
            max(domain1.window_real[0], domain2.window_real[0]),
            min(domain1.window_real[1], domain2.window_real[1])
        )
        imag_bounds = (
            max(domain1.window_imag[0], domain2.window_imag[0]),
            min(domain1.window_imag[1], domain2.window_imag[1])
        )
    elif operation == 'difference':
        # Difference is contained within first domain
        real_bounds = domain1.window_real
        imag_bounds = domain1.window_imag
    else:  # union
        # Keep existing behavior for union
        real_bounds = (min(all_reals), max(all_reals))
        imag_bounds = (min(all_imags), max(all_imags))
```

### Option 3: Lazy Evaluation with Caching

Add a property that calculates tight bounds on first access:

```python
@property
def tight_bounds(self):
    """Get tight bounds, calculating if necessary."""
    if not hasattr(self, '_tight_bounds'):
        self._tight_bounds = self.calculate_tight_bounds()
    return self._tight_bounds

def viewing_window(self, tight: bool = True):
    """Get viewing window, optionally using tight bounds."""
    if tight:
        real_bounds, imag_bounds = self.tight_bounds
    else:
        real_bounds = self.window_real
        imag_bounds = self.window_imag
    return real_bounds, imag_bounds
```

## Implementation Steps

1. Add `calculate_tight_bounds()` method to CompositeDomain
2. Add optional `tight_bounds` parameter to Domain base class
3. Update plotting functions to use tight bounds when available
4. Add tests for various composite domain scenarios
5. Document the new behavior

## Usage Example

```python
# After implementation
domain_grid = create_disk_grid(n=2, spacing=1, radius=0.6)

# Automatic tight bounds
cp.plot(domain_grid, func, cmap=cmap)  # Uses tight bounds

# Or explicit control
cp.plot(domain_grid, func, cmap=cmap, use_tight_bounds=True)
```

## Considerations

- Performance: Tight bounds calculation requires sampling, which adds overhead
- Cache invalidation: If domains are mutable, cached bounds need updating
- Empty domains: Need to handle cases where operations result in empty sets
- Backward compatibility: Existing code should continue to work