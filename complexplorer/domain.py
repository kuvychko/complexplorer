from math import ceil
from functools import reduce
import numpy as np
from typing import Optional, Tuple, List, Callable
from copy import deepcopy

"""
This module contains a set of classes for the construction and manipulation of complex domains in the mathematical sense.
A domain is represented as a rectangular region in the complex plane, defined by a tuple for the real axis range and imaginary axis range.

The key functionality of the module is comprised of the `Domain`, `Rectangle`, `Disk`, and `Annulus` classes.
`Domain`, being the base class, encapsulates the meshing and masking that occur upon specification of a `Domain` instance.
Instances of the domain can be rectangular (`Rectangle`), circular (`Disk`), or annular (`Annulus`).

The meshing mechanism of a `Domain` employs a 'deferred' approach, meaning the mesh and mask are not instantly calculated
upon the creation of a `Domain` instance but are instead calculated as needed based on the number of desired mesh points.
This allows the user to generate instances of meshes with different interval sizes depending on their needs, without having to
re-define a domain. The mesh and mask computations are performed when invoking the `*.mesh` and `*.mask` methods of the instance respectively. 

In essence, a mesh refers to a 2-D grid of complex numbers representing the points that populate a domain.
Conversely, a mask is a 2-D Boolean array of the same size as  points belonging (or not) to the domain.
The `*.inmask()` method returns a mask where the True values represent points belonging to the domain. for the "inmask" and
points outside of the domain for '*.outmask` method output.

The main classes provided are:

- `Domain`: This class serves as the base class for defining complex domains. 
The meshing and masking of the domain are performed dynamically when requested.

- `Rectangle`: A subclass of `Domain`, the `Rectangle` class allows the creation of rectangular domains centered at a given point. 
It takes the dimensions (real and imaginary) of the rectangle and the center point as input.

- `Disk`: Another subclass of `Domain`, the `Disk` class enables the creation of circular domains (disks) centered at a given point.
It requires specifying the radius of the disk and the center point.

- `Annulus`: The `Annulus` class, also a subclass of `Domain`, enables the creation of annular domains (rings) centered at a given point.
It requires specifying the inner and outer radii and the center point.
"""


__all__ = ['Domain', 'Rectangle', 'Disk', 'Annulus']

class Domain():
    """
    Class that defines complex domains. Domain definition is comprised of two parts: 2D mesh of complex numbers 
    which defines a rectangle where the actual domain resides (with sides parallel to real and imaginary axes),
    and a corresponding 2D Boolean mask array which shows which points within the mesh belong to the Domain.
    The mask is required to define curvilinear domains such Disk and Annular.

    The meshing of Domain is deferred and performed in calls of *.mesh and *.mask methods. The idea behind it is
    that mesh grain likely requires iterations to achieve the best look of a plot, so it makes sense to do it from
    the corresponding plot function.
    """
    def __init__(self, real: Tuple[float, float], imag: Tuple[float, float], mask_list: Optional[List[Callable]] = None):
        if real[0] == real[1]:
            msg = f"First and second values of real input cannot be equal"
            raise ValueError(msg)
        if imag[0] == imag[1]:
            msg = f"First and second values of imag input cannot be equal"
            raise ValueError(msg)
        
        self.real_range = (min(real), max(real))
        self.imag_range = (min(imag), max(imag))
        self.mask_list = mask_list

    def spacing(self, n):
        # calculating point spacing for real and imaginary axes
        real_length = self.real_range[1] - self.real_range[0]
        imag_length = self.imag_range[1] - self.imag_range[0]
        return max([real_length, imag_length])/n

    def mesh(self, n):
        """
        Return a rectangular complex mesh.
        
        The distance between axis points is defined by dividing the longer axis by input n.
        """

        # calculating point spacing for real and imaginary axes
        real_length = self.real_range[1] - self.real_range[0]
        imag_length = self.imag_range[1] - self.imag_range[0]
        spacing = self.spacing(n)
        
        real_axis = np.linspace(self.real_range[0], self.real_range[1], ceil(real_length/spacing))
        imag_axis = np.linspace(self.imag_range[0], self.imag_range[1], ceil(imag_length/spacing))
        x, y = np.meshgrid(real_axis, imag_axis)
        z = x + 1j*y
        return z
    
    def inmask(self, n):
        """
        Return a boolean mask which defines a valid domain within the rectangular mesh region.
        """

        z = self.mesh(n)
        mask = np.full_like(z, True, dtype=bool)
        if self.mask_list is None:
            return mask
        else:
            # the loop is needed if mask_dict contains multiple mask functions
            mask_array_list = []
            for func in self.mask_list:
                mask_array_list.append(func(z))
            # by defining "out" argument in np.logical_and numpy doesn't need to allocate a new array for each output
            mask = reduce(lambda x,y: np.logical_or(x, y, out=mask), mask_array_list)
            return mask
    
    def outmask(self, n):
        return ~self.inmask(n)
        
    def domain(self, n):
        """
        Return a domain mesh (rectangular mesh with values outside of the domain set to np.nan)
        """
        mesh = self.mesh(n)
        mesh[~self.outmask(n)] = np.nan
        return mesh
    
    def union(self, a):
        """
        Create a union of Domain with another instance of Domain.
        """

        left = min(list(self.real_range) + list(a.real_range))
        right = max(list(self.real_range) + list(a.real_range))
        bottom = min(list(self.imag_range) + list(a.imag_range))
        top = max(list(self.imag_range) + list(a.imag_range))
        return Domain((left, right), (bottom, top), self.mask_list + a.mask_list)
    
class Rectangle(Domain):
    def __init__(self, real: float, imag: float, center: complex = 0+0.0j):
        """
        Intialize a Domain intance corresponding to Rectangle centered at center.
        """

        real = abs(real)/2
        imag = abs(imag)/2
        real_range = (center.real - real, center.real + real)
        imag_range = (center.imag - imag, center.imag + imag)
        super().__init__(real_range, imag_range)

class Disk(Domain):
    def __init__(self, radius: float, center: complex = 0+0.0j):
        """
        Intialize a Domain instance corresponding to Rectangle centered at center.
        """

        if radius <= 0:
            raise ValueError('Radius must be positive')
        
        real_range = (center.real - radius, center.real + radius)
        imag_range = (center.imag - radius, center.imag + radius)
        mask_func = lambda x: np.less_equal(np.absolute(x - center), radius)
        super().__init__(real_range, imag_range, mask_list=[mask_func])

class Annulus(Domain):
    def __init__(self, radius_inner: float, radius_outer: float, center: complex = 0+0.0j):
        """
        Initialize a Domain instance of Annulus (ring)
        """

        if radius_inner <= 0:
            raise ValueError('radius_inner must be positive')
        if radius_outer <= radius_inner:
            raise ValueError('radius_outer must be greater than radius_inner')
        
        real_range = (center.real - radius_outer, center.real + radius_outer)
        imag_range = (center.imag - radius_outer, center.imag + radius_outer)
        def mask_func(x):
            belongs_inside_outer = np.less_equal(np.absolute(x - center), radius_outer)
            belongs_outside_inner = np.greater_equal(np.absolute(x - center), radius_inner)
            return np.logical_and(belongs_inside_outer, belongs_outside_inner)
        super().__init__(real_range, imag_range, mask_list=[mask_func])
