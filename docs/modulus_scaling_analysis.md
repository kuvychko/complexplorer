# Modulus Scaling Analysis for Riemann Sphere Visualization

## Current Scaling Methods

The complexplorer library currently provides 5 modulus scaling methods for mapping complex function moduli `|f(z)|` from `[0, ∞)` to a finite radius range `[r_min, r_max]`:

### 1. **Constant** ⭐⭐⭐
```python
r = radius  # Fixed radius, no modulus information
```
- **Use case**: Traditional Riemann sphere showing only phase
- **Visual quality**: Clean but loses magnitude information

### 2. **Linear** ⭐
```python
r = 1 + scale * |f(z)|
```
- **Problem**: Unbounded growth, poor for functions with poles
- **Visual quality**: Often produces extreme distortions

### 3. **Arctan** ⭐⭐⭐⭐
```python
normalized = (2/π) * arctan(|f(z)|)
r = r_min + (r_max - r_min) * normalized
```
- **Use case**: General purpose, smooth compression
- **Visual quality**: Good balance, but can be too uniform

### 4. **Logarithmic** ⭐⭐
```python
log_mod = log(|f(z)|) / log(base)
normalized = 1 / (1 + exp(-log_mod))
r = r_min + (r_max - r_min) * normalized
```
- **Use case**: Functions with exponential growth
- **Visual quality**: Can overemphasize small values

### 5. **Linear Clamp** ⭐⭐⭐
```python
clamped = min(|f(z)|, m_max)
normalized = clamped / m_max
r = r_min + (r_max - r_min) * normalized
```
- **Use case**: Focus on specific modulus range
- **Visual quality**: Good for bounded regions, harsh cutoff

### 6. **Power** (Exists but not exposed) ⭐⭐⭐
```python
normalized = (|f(z)| / max(|f|))^exponent
r = r_min + (r_max - r_min) * normalized
```
- **Use case**: Fine-tuning compression/expansion
- **Visual quality**: Flexible but requires max calculation

### 7. **Custom** (Referenced but not implemented) ⭐⭐⭐⭐⭐
- Allows user-defined scaling functions

## Problems with Current Methods

1. **Lack of artistic control**: Most methods are purely mathematical
2. **Poor handling of zeros/poles**: Extreme values dominate visualization
3. **No perceptual uniformity**: Linear changes in modulus don't map to perceptually uniform changes
4. **Limited dynamic range control**: Hard to visualize functions with both small and large values

## Proposed New Scaling Methods

### High Usefulness (⭐⭐⭐⭐⭐)

#### 1. **Sigmoid Family**
```python
def sigmoid_scaling(moduli, steepness=1.0, center=1.0, r_min=0.2, r_max=1.0):
    """Smooth S-curve scaling with adjustable transition."""
    normalized = 1 / (1 + np.exp(-steepness * (moduli - center)))
    return r_min + (r_max - r_min) * normalized
```
- **Benefits**: Smooth transition, adjustable center and steepness
- **Use case**: Functions with clear "interesting" modulus range

#### 2. **Adaptive Percentile**
```python
def adaptive_percentile_scaling(moduli, low_percentile=10, high_percentile=90, r_min=0.2, r_max=1.0):
    """Scale based on data percentiles, ignoring extreme outliers."""
    p_low = np.percentile(moduli[np.isfinite(moduli)], low_percentile)
    p_high = np.percentile(moduli[np.isfinite(moduli)], high_percentile)
    
    normalized = np.clip((moduli - p_low) / (p_high - p_low), 0, 1)
    return r_min + (r_max - r_min) * normalized
```
- **Benefits**: Automatically adapts to data range, ignores outliers
- **Use case**: General purpose, especially for unknown functions

#### 3. **Hybrid Linear-Logarithmic**
```python
def hybrid_scaling(moduli, transition=1.0, r_min=0.2, r_max=1.0):
    """Linear for small values, logarithmic for large values."""
    small_mask = moduli <= transition
    large_mask = ~small_mask
    
    normalized = np.zeros_like(moduli)
    # Linear part: [0, transition] -> [0, 0.5]
    normalized[small_mask] = 0.5 * moduli[small_mask] / transition
    # Log part: (transition, ∞) -> (0.5, 1]
    normalized[large_mask] = 0.5 + 0.5 * np.tanh(np.log(moduli[large_mask] / transition))
    
    return r_min + (r_max - r_min) * normalized
```
- **Benefits**: Preserves detail at both small and large scales
- **Use case**: Functions with interesting behavior near zeros and poles

### Medium-High Usefulness (⭐⭐⭐⭐)

#### 4. **Smooth Step**
```python
def smooth_step_scaling(moduli, steps=[0.1, 1, 10], r_values=[0.2, 0.5, 0.8, 1.0]):
    """Piecewise smooth scaling with multiple plateaus."""
    # Implementation using cubic Hermite interpolation between steps
    # Creates visually distinct "levels" while maintaining smoothness
```
- **Benefits**: Creates clear visual layers for different magnitude ranges
- **Use case**: Highlighting specific modulus values (e.g., |f|=1)

#### 5. **Gaussian Bump**
```python
def gaussian_bump_scaling(moduli, center=1.0, width=0.5, bump_height=0.3, r_min=0.2, r_max=1.0):
    """Emphasize values near a specific modulus."""
    base = r_min + (r_max - r_min) * np.tanh(0.5 * moduli)
    bump = bump_height * np.exp(-((moduli - center) / width) ** 2)
    return np.clip(base + bump, r_min, r_max)
```
- **Benefits**: Highlights interesting modulus values
- **Use case**: Emphasizing unit circle or other special values

#### 6. **Perceptual Uniform**
```python
def perceptual_uniform_scaling(moduli, r_min=0.2, r_max=1.0):
    """Based on Stevens' power law for magnitude perception."""
    # Psychophysical scaling: perceived = actual^0.3 for brightness
    normalized = np.power(moduli / (1 + moduli), 0.3)
    return r_min + (r_max - r_min) * normalized
```
- **Benefits**: Perceptually linear changes
- **Use case**: Scientific visualization where perception matters

### Medium Usefulness (⭐⭐⭐)

#### 7. **Reciprocal**
```python
def reciprocal_scaling(moduli, offset=1.0, r_min=0.2, r_max=1.0):
    """Inverse scaling: emphasizes small values."""
    normalized = 1 - 1 / (1 + moduli / offset)
    return r_min + (r_max - r_min) * normalized
```
- **Benefits**: Shows detail near zeros
- **Use case**: Functions with interesting behavior near zeros

#### 8. **Sawtooth Modulus**
```python
def sawtooth_scaling(moduli, period=1.0, r_min=0.2, r_max=1.0):
    """Periodic scaling creating rings."""
    normalized = (moduli % period) / period
    return r_min + (r_max - r_min) * normalized
```
- **Benefits**: Shows modulus contours clearly
- **Use case**: Educational visualization of modulus levels

#### 9. **Artistic Wave**
```python
def wave_scaling(moduli, frequency=2.0, amplitude=0.1, r_min=0.2, r_max=1.0):
    """Base scaling with sinusoidal perturbation."""
    base = np.tanh(0.5 * moduli)
    wave = amplitude * np.sin(frequency * np.pi * moduli)
    normalized = np.clip(base + wave, 0, 1)
    return r_min + (r_max - r_min) * normalized
```
- **Benefits**: Creates visually interesting patterns
- **Use case**: Artistic visualizations

### Lower Usefulness (⭐⭐)

#### 10. **Threshold Bands**
```python
def threshold_bands_scaling(moduli, thresholds=[0.1, 1, 10], r_min=0.2, r_max=1.0):
    """Discrete bands based on thresholds."""
    # Creates distinct rings but lacks smoothness
```

#### 11. **Exponential Decay**
```python
def exp_decay_scaling(moduli, decay_rate=1.0, r_min=0.2, r_max=1.0):
    """Exponential decay from maximum radius."""
    normalized = np.exp(-decay_rate * moduli)
    return r_min + (r_max - r_min) * normalized
```

## Implementation Recommendations

### 1. **Fix Missing Custom Implementation**
```python
@staticmethod
def custom(moduli: np.ndarray, scaling_func: Callable[[np.ndarray], np.ndarray],
          r_min: float = 0.2, r_max: float = 1.0) -> np.ndarray:
    """Apply user-defined scaling function."""
    normalized = scaling_func(moduli)
    # Ensure output is in [0, 1]
    normalized = np.clip(normalized, 0, 1)
    return r_min + (r_max - r_min) * normalized
```

### 2. **Add Most Useful Methods**
Priority order for implementation:
1. Custom (fix missing implementation)
2. Sigmoid family
3. Adaptive percentile
4. Hybrid linear-logarithmic
5. Smooth step
6. Perceptual uniform

### 3. **Improve Existing Methods**
- Add `power` method to the exposed interface
- Make `linear` method bounded by default
- Add better parameter defaults based on common use cases

### 4. **Create Scaling Presets**
```python
SCALING_PRESETS = {
    'balanced': {'method': 'sigmoid', 'steepness': 2.0, 'center': 1.0},
    'detail_near_zero': {'method': 'hybrid', 'transition': 0.5},
    'highlight_unit': {'method': 'gaussian_bump', 'center': 1.0, 'width': 0.3},
    'artistic': {'method': 'wave', 'frequency': 3.0, 'amplitude': 0.1},
    'scientific': {'method': 'perceptual_uniform'},
}
```

## Visual Quality Comparison

### Best for General Use:
1. **Adaptive Percentile**: Automatically handles any function well
2. **Sigmoid**: Smooth and adjustable for known ranges
3. **Hybrid**: Good balance between small and large value detail

### Best for Specific Cases:
- **Zeros**: Reciprocal, Hybrid (linear part)
- **Poles**: Arctan, Sigmoid with high center
- **Unit Circle**: Gaussian bump centered at 1
- **Educational**: Sawtooth, Smooth step
- **Artistic**: Wave, Custom with creative functions

## Conclusion

The current scaling methods are functional but limited in their ability to create visually pleasing and informative visualizations. The proposed extensions focus on:

1. **Adaptability**: Methods that automatically adjust to data
2. **Perceptual quality**: Scaling that matches human perception
3. **Artistic control**: Options for creative visualization
4. **Special features**: Highlighting specific mathematical properties

Implementing the top 5-6 proposed methods would significantly enhance the visual quality and usefulness of Riemann sphere visualizations in complexplorer.