# Complexplorer

Complexplorer is a Python library for visualization of complex functions. It is insipred by Elias Wegert's book ["Visual Complex Functions - An Introduction with Phase Portraits"](https://link.springer.com/book/10.1007/978-3-0348-0180-5).

The library provides classes and functions to:
* Create complex domains and corresponding complex-valued 2D arrays (meshes)
* Convert complex-valued 2D arrays to HSV and RGB color maps according to various schemes (Enhanced Phase Portrait, Chessboard, PolarChessboard, LogRings)
* Visualize complex-valued 2D arrays as 2D and 3D plots (2D image, 3D analytic landscape, 3D Riemann sphere)

Design choices of this library enable:
* Simple composability: any domain can be used with any plot and any color map, yielding a multitude of different visualizations.
* Deferred evaluation of domain meshes. Meshing is performed inside plotting functions (and not during domain instanciation). This allows for quick iteration of mesh period for best visual results.
* Different domain instances can be unioned together to create complex domains.

One exception from the composability ideal is Riemann sphere 3D plot. It has its own meshing algorithm to balance point density between the poles and equator.
