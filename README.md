# Complexplorer

*We cannot directly see the minute details of a Dedekind cut, nor is it clear that arbitrarily great or
arbitrarily tiny times or lengths actually exist in nature. One could say that 
the so-called ‘real numbers’ are as much a product of mathematicians’ 
imaginations as are the complex numbers. Yet we shall find that complex 
numbers, as much as reals, and perhaps even more, find a unity with 
nature that is truly remarkable. It is as though Nature herself is as 
impressed by the scope and consistency of the complex-number system 
as we are ourselves, and has entrusted to these numbers the precise 
operations of her world at its minutest scales.* ...

*Moreover, to refer just to the scope and to the consistency of complex 
numbers does not do justice to this system. There is something more 
which, in my view, can only be referred to as ‘magic’.*

[Road to Reality](https://www.ams.org/notices/200606/rev-blank.pdf), Chapter 4 - Magical Complex Numbers, Sir Roger Penrose

Complexplorer is a Python library for visualization of complex functions. 
The library was insipred by Elias Wegert's book ["Visual Complex Functions - An Introduction with Phase Portraits"](https://link.springer.com/book/10.1007/978-3-0348-0180-5) and it greatly benefitted from discussions and feedback that Elias kindly provided. The library supports enhanced phase portraits and 
several other visual styles. 

The library provides classes and functions to:  

* Create complex domains and corresponding complex-valued 2D arrays (meshes)
* Convert complex-valued 2D arrays to HSV and RGB color maps according to various schemes (Enhanced Phase Portrait, Chessboard, PolarChessboard, LogRings)
* Visualize complex-valued 2D arrays as 2D and 3D plots (2D image, 3D analytic landscape, 3D Riemann sphere)

Design choices of this library enable:  

* Simple composability: any domain can be used with any plot and any color map, yielding a multitude of different visualizations.
* Deferred evaluation of domain meshes. Meshing is typically performed during plotting (and not during domain instanciation). This allows for quick iteration of mesh period for best visual results.
* Different domain instances can be composed using union and intersection operations to create complex domains.

One exception from the composability ideal is Riemann sphere 3D plot. It has its own meshing algorithm to balance point density between the poles and equator, so that function does not accept domain (or input z arrays) as its argument.

Complexplorer is designed to be very light-weight in terms of its dependencies. It requires only numpy and matplotlib, which comes at a cost. 
Matplotlib is not a 3D rendering library, so 3D visualizations are painfully slow. This is especially true for Riemann sphere plot which uses 
a custom rectangular mesh that wastes a lot of points at the poles. A triangular mesh would be the right tool here, but I have not figured out how to achieve arbitrary point coloring in matplotlib.

## Library overview

The library contains following classes and functions.

### Domains

* `Domain`: This class serves as the base class for defining complex domains. It encapsulates 
the meshing and masking functionality of a `Domain` instance.

* `Rectangle`: A subclass of `Domain`, the `Rectangle` class allows the creation of rectangular domains centered at a given point. 
It takes the length (real and imaginary) of the rectangle and the center point as input.

* `Disk`: Another subclass of `Domain`, the `Disk` class enables the creation of circular domains (disks) centered at a given point.
It requires specifying the radius of the disk and the center point.

* `Annulus`: The `Annulus` class, also a subclass of `Domain`, enables the creation of annular domains (rings) centered at a given point.
It requires specifying the inner and outer radii and the center point.

### Color maps

* `Cmap`: This class serves as a base class for color maps and defines 
an informal interface for child color map classes. It implements 
the `*.hsv()` and `*.rgb()` methods which are used to convert 
input complex values to HSV and RGB-valued arrays.

* `Phase`: This class implements a phase color map. It can be used
to generate regular phase color maps or enhanced phase color maps.

* `Chessboard`: This class implements a chessboard color map.

* `PolarChessboard`: This class implements a polar chessboard color map.

* `LogRings`: This class implements a logarithmic black and white rings color map.

### 2D plotting functions

* `plot`: plot complex function as pullback of the color map of the co-domain to the domain.

* `pair_plot`: plot color maps of the domain and the pullback of the co-domain of the function.

* `riemann_chart`: plot the phase portrait of a complex function projected from the Riemann hemisphere.

* `riemann_hemispheres`: plot a pair of phase portraits corresponding to the upper and lower hemispheres of the Riemann sphere.

### 3D plotting functions

* `plot_landscape`: plot a complex function as a 3D landscape on the complex plane.

* `pair_plot_landscape`: - plot analytic landscapes of the domain and the pullback of the co-domain of the function.

* `riemann`: plot a complex function as a phase portrait on the Riemann sphere.

### Supporting functions

* `phase`: return a phase of complex input mapped to [0, 2*pi) interval.

* `sawtooth`: return a sawtooth wave of input x.

* `stereographic`: return a (x,y,z) tuple corresponding to stereographic projection of complex input z.

## Installation

Install using pip:

```
pip install complexplorer
```

## Documentation

Every module, class, and function of the library is documented via a docstring. Use `help` or `?` to view them.

## Example notebooks

* [Basic functionality overview](https://github.com/kuvychko/complexplorer/tree/main/examples/plots_example.ipynb)

* [Domains and color maps](https://github.com/kuvychko/complexplorer/tree/main/examples/domains_cmaps_example.ipynb)

## Gallery

Examples below use a test function $f(z) = \frac{z - 1}{z^2 + z + 1}$, a standard example from ["Visual Complex Functions - An Introduction with Phase Portraits"](https://link.springer.com/book/10.1007/978-3-0348-0180-5). Different color maps and plot types are shown. For the code used to generate these plots see this [example Jupyter notebook](https://github.com/kuvychko/complexplorer/tree/main/examples/plot_example.ipynb)

### Phase portraits (domain and co-domain side-by-side)

![Phase portraint](https://github.com/kuvychko/complexplorer/blob/main/examples/gallery/Phase_portrait_2d.png?raw=true)


![Phase portraint phase enhanced](https://github.com/kuvychko/complexplorer/blob/main/examples/gallery/Phase_portrait_phase_enhanced_2d.png?raw=true)

![Phase portraint modulus enhanced](https://github.com/kuvychko/complexplorer/blob/main/examples/gallery/Phase_portrait_modulus_enhanced_2d.png?raw=true)

![Enhanced phase portraint phase and modulus enhanced](https://github.com/kuvychko/complexplorer/blob/main/examples/gallery/Enhanced_phase_portrait_phase_and_modulus_enhanced_2d.png?raw=true)

![Polar chessboard linear](https://github.com/kuvychko/complexplorer/blob/main/examples/gallery/Polar_chessboard_linear_modulus_spacing_2d.png?raw=true)

![Polar chessboard log](https://github.com/kuvychko/complexplorer/blob/main/examples/gallery/Polar_chessboard_log_modulus_spacing_2d.png?raw=true)

![Logarithmic rings](https://github.com/kuvychko/complexplorer/blob/main/examples/gallery/Logarithmic_rings_2d.png?raw=true)

### Analytic landscapes  (domain and co-domain side-by-side)

![Phase portraint](https://github.com/kuvychko/complexplorer/blob/main/examples/gallery/Phase_portrait_3d.png?raw=true)

![Phase portraint phase enhanced](https://github.com/kuvychko/complexplorer/blob/main/examples/gallery/Phase_portrait_phase_enhanced_3d.png?raw=true)

![Phase portraint modulus enhanced](https://github.com/kuvychko/complexplorer/blob/main/examples/gallery/Phase_portrait_modulus_enhanced_3d.png?raw=true)

![Enhanced phase portraint phase and modulus enhanced](https://github.com/kuvychko/complexplorer/blob/main/examples/gallery/Enhanced_phase_portrait_phase_and_modulus_enhanced_3d.png?raw=true)

![Polar chessboard linear](https://github.com/kuvychko/complexplorer/blob/main/examples/gallery/Polar_chessboard_linear_modulus_spacing_3d.png?raw=true)

![Polar chessboard log](https://github.com/kuvychko/complexplorer/blob/main/examples/gallery/Polar_chessboard_log_modulus_spacing_3d.png?raw=true)

![Logarithmic rings](https://github.com/kuvychko/complexplorer/blob/main/examples/gallery/Logarithmic_rings_3d.png?raw=true)

### 2D Riemann chart (projected hemispheres)

![Riemann charts](https://github.com/kuvychko/complexplorer/blob/main/examples/gallery/riemann_chart_2d.png?raw=true)

### Riemann sphere

![Riemann sphere](https://github.com/kuvychko/complexplorer/blob/main/examples/gallery/riemann_sphere_3d.png?raw=true)

## Future work

Open an issue and leave a comment using [Bug Tracker](https://github.com/kuvychko/complexplorer/issues) if you would like to see any of these (or other) features implemented. Collaboration is always welcome!

* Write unit tests. Currently [Basic functionality overview](https://github.com/kuvychko/complexplorer/tree/main/examples/plots_example.ipynb) notebook serves as the unit test set, but it is not a very elegant solution.

* Minimization of viewing window of the intersection of two domains to fit the resulting domain snuggly. Currently this is not implemented, and the viewing window is the same as in the case of a union operation.

* Triangular meshing and corresponding 3D visualization of Riemann sphere (I am not sure if this is doable in matplotlib).

* A modified Riemann sphere projection, where first spherical coordinates of points on the sphere are calculated, then radius is scaled to represent the modulus of the corresponding complex value. It would be nice to implement this using triangular mesh.

* Function to convert modified Riemann sphere projection described above into an *.stl file for 3D printing ("Christmas ornaments" generator). This is Elias Wegert's idea - would be fun!
