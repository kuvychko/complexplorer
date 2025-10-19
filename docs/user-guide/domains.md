# Domains

Domains define the region of the complex plane where you want to visualize a function. Complexplorer provides several built-in domain types and supports custom domains through composition.

## Built-in Domain Types

### Rectangle

The most common domain type, representing a rectangular region in the complex plane.

```python
import complexplorer as cp

# Square domain centered at origin
domain = cp.Rectangle(re_length=4, im_length=4, center=0+0j)

# Rectangular domain (non-square)
domain = cp.Rectangle(re_length=6, im_length=4, center=0+0j)

# Off-center domain
domain = cp.Rectangle(re_length=4, im_length=4, center=2+1j)

# Convenience: automatically square if only one length given
domain = cp.Rectangle(4, 4)  # Square domain
```

**Parameters:**

- `re_length`: Width (real axis extent)
- `im_length`: Height (imaginary axis extent)
- `center`: Complex number for domain center (default: 0+0j)
- `square`: If True, forces im_length = re_length (default: True when im_length not specified)

**Use cases:**

- General purpose visualization
- Functions with rectangular symmetry
- Exploring specific regions of the complex plane

### Disk

A circular region centered at a point.

```python
# Unit disk
domain = cp.Disk(radius=1, center=0)

# Larger disk
domain = cp.Disk(radius=3, center=0)

# Off-center disk
domain = cp.Disk(radius=2, center=1+1j)
```

**Parameters:**

- `radius`: Disk radius (positive float)
- `center`: Complex number for disk center (default: 0+0j)

**Use cases:**

- Functions with rotational symmetry
- Bounded visualizations (avoids infinity for many functions)
- Unit disk is natural for Mobius transformations and conformal maps

### Annulus

A ring-shaped region between two circles.

```python
# Simple annulus
domain = cp.Annulus(inner_radius=0.5, outer_radius=2, center=0)

# Thin ring
domain = cp.Annulus(inner_radius=0.9, outer_radius=1.1, center=0)

# Off-center annulus
domain = cp.Annulus(inner_radius=1, outer_radius=3, center=2j)
```

**Parameters:**

- `inner_radius`: Inner circle radius (positive float)
- `outer_radius`: Outer circle radius (must be > inner_radius)
- `center`: Complex number for annulus center (default: 0+0j)

**Use cases:**

- Functions with poles at origin (exclude singularity)
- Laurent series visualization
- Focusing on specific magnitude ranges
- Avoiding branch cuts

## Domain Properties

All domains provide:

- `contains(z)`: Check if point(s) z are in the domain
- `sample(resolution)`: Generate grid of points for visualization

```python
domain = cp.Disk(radius=2, center=0)

# Check if point is in domain
point = 1 + 1j
if domain.contains(point):
    print("Point is in domain")

# Check multiple points
points = np.array([0+0j, 1+1j, 3+3j])
mask = domain.contains(points)  # Returns boolean array
```

## Choosing the Right Domain

Different functions benefit from different domains:

### Polynomial Functions

For polynomials like z^n, use a **Disk** or **Rectangle**:

```python
f = lambda z: z**3 - 2*z + 1
domain = cp.Disk(radius=2)
cp.plot(domain, f)
```

### Rational Functions

For rational functions with poles, use an **Annulus** to exclude singularities:

```python
# Function has pole at origin
f = lambda z: 1/z
domain = cp.Annulus(inner_radius=0.2, outer_radius=3)
cp.plot(domain, f)
```

### Functions with Branch Cuts

For functions like sqrt(z) or log(z), carefully position the domain to avoid branch cuts:

```python
# sqrt(z) has branch cut on negative real axis
f = lambda z: np.sqrt(z)

# Position domain to avoid branch cut
domain = cp.Rectangle(4, 4, center=1+0j)  # Centered at z=1 (positive real)
cp.plot(domain, f)
```

### Mobius Transformations

For Mobius transformations, the **unit disk** is natural:

```python
# Maps unit disk to itself (if |a| < 1)
a = 0.5 + 0.3j
f = lambda z: (z - a) / (1 - np.conj(a) * z)

domain = cp.Disk(radius=1)
cp.plot(domain, f)
```

## Advanced: Domain Composition

!!! warning "Coming Soon"
    Domain composition through set operations (union, intersection, difference) is planned for a future release.

## Domain Best Practices

### 1. Start with Default Domains

For exploration, use simple domains:

```python
# Quick exploration with show()
cp.show(lambda z: z**2)  # Uses Rectangle(-2, 2) x (-2i, 2i)

# Explicit default
domain = cp.Rectangle(4, 4)
```

### 2. Match Domain to Function Behavior

Consider where your function has:

- **Poles**: Exclude with Annulus
- **Branch cuts**: Position Rectangle carefully
- **Rotational symmetry**: Use Disk
- **Rectangular symmetry**: Use Rectangle

### 3. Avoid Infinity at Domain Boundary

For functions that blow up at infinity, use bounded domains:

```python
# Bad: Rectangle can extend to large values
domain = cp.Rectangle(100, 100)  # Function may be unstable at edges

# Good: Disk naturally bounds the domain
domain = cp.Disk(radius=5)  # More stable visualization
```

### 4. Resolution Considerations

Larger domains need higher resolution for the same detail:

```python
# Small domain - 400 points is plenty
domain = cp.Rectangle(2, 2)
cp.plot(domain, f, resolution=400)

# Large domain - need more points
domain = cp.Rectangle(10, 10)
cp.plot(domain, f, resolution=800)
```

### 5. Domain for STL Export

For 3D printing, carefully choose domains to avoid numerical issues:

```python
from complexplorer.export.stl import OrnamentGenerator

# Exclude singularities with annulus
f = lambda z: 1 / (z**2 + 1)
domain = cp.Annulus(inner_radius=0.1, outer_radius=3)

ornament = OrnamentGenerator(func=f, domain=domain, resolution=150)
ornament.generate_and_save('ornament.stl', size_mm=80)
```

## Domain Visualization Tips

### Comparing Domain and Codomain

Use `pair_plot()` to see both input and output:

```python
f = lambda z: z**2
domain = cp.Disk(radius=2)

# Shows domain on left, codomain on right
cp.pair_plot(domain, f, figsize=(12, 6))
```

### Domain Shape in 3D

The domain shape is more apparent in 3D landscapes:

```python
# Disk domain creates circular base
domain = cp.Disk(radius=2)
cp.plot_landscape_pv(domain, f)

# Annulus domain creates ring-shaped base
domain = cp.Annulus(0.5, 2)
cp.plot_landscape_pv(domain, f)
```

## Examples by Function Type

### Elementary Functions

```python
# Exponential
f = lambda z: np.exp(z)
domain = cp.Rectangle(4, 4)  # Unbounded, but centered view

# Sine/Cosine
f = lambda z: np.sin(z)
domain = cp.Rectangle(6, 6)  # See multiple periods

# Power function
f = lambda z: z**n
domain = cp.Disk(radius=2)  # Rotational symmetry
```

### Rational Functions

```python
# Simple pole
f = lambda z: 1/(z - 1)
domain = cp.Annulus(0.3, 3, center=1)  # Exclude pole at z=1

# Multiple poles
f = lambda z: 1/(z**2 - 1)
domain = cp.Disk(radius=3)  # Poles at +/-1, but domain includes them

# Better: exclude both poles
domain = cp.Rectangle(6, 6)  # Rectangular domain shows both
```

### Multi-valued Functions

```python
# Square root (branch cut on negative real axis)
f = lambda z: np.sqrt(z)
domain = cp.Rectangle(4, 4, center=1+0j)  # Avoid negative reals

# Logarithm (branch cut on negative real axis)
f = lambda z: np.log(z)
domain = cp.Disk(radius=3, center=1+0j)  # Centered away from origin
```

## Common Patterns

### Exploring Function Near Origin

```python
# Small disk for detailed view
domain = cp.Disk(radius=0.5)
cp.plot(domain, f, resolution=600)  # High res for detail
```

### Exploring Function at Infinity

```python
# Large disk or use Riemann sphere
domain = cp.Disk(radius=10)
cp.plot(domain, f)

# Better: Riemann sphere shows infinity explicitly
cp.riemann_pv(f, modulus_mode='arctan')
```

### Asymmetric Functions

```python
# Function interesting in right half-plane
f = lambda z: np.exp(z) * np.sin(z)
domain = cp.Rectangle(4, 4, center=2+0j)  # Centered at z=2
```

## Next Steps

- **[Colormaps](colormaps.md)** - Choose how to visualize your function
- **[2D Plotting](plotting-2d.md)** - Create 2D phase portraits
- **[3D Plotting](plotting-3d.md)** - Build 3D analytic landscapes
- **[Quick Start](../getting-started/quickstart.md)** - Basic usage examples
