from math import ceil
from functools import reduce
import numpy as np
from typing import Optional, Tuple, List, Callable
from copy import deepcopy

"""
This module contains a set of classes for the construction and manipulation of complex domains in the mathematical sense.

A domain is represented by a function which accepts numpy arrays of complex values and returns like-shaped arrays of boolean
values. This function is stored in the `Domain.infunc` attribute. True values correspond to input points that belong 
to the domain and False to the ones that do not.
A viewing window determines a rectangular region of complex plane which is meshed and returned by `Domain.mesh(n)` method. 
Integer n defines the number of mesh points of the longer axis of the window region (either real or imaginary).
Note that this method returned a 2D numpy array containing a complex mesh of the viewing window with no regard to 
the mask function. `Domain.domain(n)` method performs a similar meshing operation but returns a 2D complex mesh with
points outside of the domain set to `numpy.nan`. `Domain.inmask(n)` returns a Boolean 2D array corresponding to 
a respective mesh with Boolean values marking points that belong (True) or do not belong (False) to the domain.

Domain representation function is stored in the "mask" attribute of the Domain class. 
Viewing window is defined by `Domain.window_real` and `Domain.window_imag` attributes. Each stores a 2-tuple of real numbers 
which correspond to left/right (for window_real) or bottom/top (for window_imag) coordinates of the viewing window.

Classes:
-------

- `Domain`: This class serves as the base class for defining complex domains. It encapsulates 
the meshing and masking functionality of a `Domain` instance.

- `Rectangle`: A subclass of `Domain`, the `Rectangle` class allows the creation of rectangular domains centered at a given point. 
It takes the length (real and imaginary) of the rectangle and the center point as input.

- `Disk`: Another subclass of `Domain`, the `Disk` class enables the creation of circular domains (disks) centered at a given point.
It requires specifying the radius of the disk and the center point.

- `Annulus`: The `Annulus` class, also a subclass of `Domain`, enables the creation of annular domains (rings) centered at a given point.
It requires specifying the inner and outer radii and the center point.
"""


__all__ = ['Domain', 'Rectangle', 'Disk', 'Annulus']

class Domain():
    """
    Base class that defines complex domains. 
    
    A domain is represented by a function which accepts numpy arrays of complex values and returns like arrays of boolean
    values. True values correspond to input points that belong to the domain and False to the ones that do not.
    A viewing window determines a rectangular region of complex plane which is meshed and returned by Domain.mesh(n) method. 
    Integer n defines the number of mesh points of the longer axis of the window region (either real or imaginary).
    Note that this method returned a 2D numpy array containing a complex mesh of the viewing window with no regard to 
    the infunc function. Domain.domain(n) method performs a similar meshing operation but returns a 2D complex mesh with
    points outside of the domain set to numpy.nan. Domain.inmask(n) returns a Boolean 2D array corresponding to 
    a respective mesh with Boolean values marking points that belong (True) or do not belong (False) to the domain.

    Domain representation function is stored in the "infunc" attribute of the Domain class. 
    Viewing window is defined by Domain.window_real and Domain.window_imag attributes. Each stores a 2-tuple of real numbers 
    which correspond to left/right (for window_real) or bottom/top (for window_imag) coordinates of the viewing window.
    """
    def __init__(self,
                 real: Tuple[float, float],
                 imag: Tuple[float, float],
                 infunc: Optional[Callable] = lambda x: np.full_like(x, True, dtype=bool),
                 square: bool = True
                 ):
        """
        Construct a domain instance.

        Parameters:
        ----------
        real: 2-tuple of floats
            2-tuple defines the left and right edges of the viewing window. Input order of
            values is irrelevant (constructor sorts them).
        imag: 2-tuple of floats
            2-tuple defines the bottom and top edges of the viewing window. Input order of
            values is irrelevant (constructor sorts them).
        infunc: Callable, optional
            Domain representation function which accepts numpy arrays of complex values and 
            returns like arrays of boolean values. True values correspond to input points that 
            belong to the domain and False to the ones that do not. The default value is 
            a function which returns True values for any points of the complex plane.
        square: bool, optional
            If True, the viewing window is constrained to a square (the shorter input axis is
            scaled to be equal to the larger one). If False viewing window may be rectangular.
            The default value is True.
        """

        if real[0] == real[1]:
            msg = f"First and second values of real input cannot be equal"
            raise ValueError(msg)
        if imag[0] == imag[1]:
            msg = f"First and second values of imag input cannot be equal"
            raise ValueError(msg)
        
        self.square = square
        if square:
            real_d = abs(real[0] - real[1])
            imag_d = abs(imag[0] - imag[1])
            delta = (real_d - imag_d)/2
            if delta >= 0:
                self.window_real = (min(real), max(real))
                self.window_imag = (min(imag) - delta, max(imag) + delta)
            else:
                delta *= -1
                self.window_real = (min(real) - delta, max(real) + delta)
                self.window_imag = (min(imag), max(imag))
        else:
            self.window_real = (min(real), max(real))
            self.window_imag = (min(imag), max(imag))
        self.infunc = infunc

    def spacing(self, n: int) -> float:
        """
        Calculate point spacing of the viewing window mesh.

        This spacing is used for meshing of both real and imaginary axes.

        Parameters:
        ----------
        n: int
            Number of sampling points for the longer axis of the viewing window.

        Returns:
        -------
        float
            Distance between adjacent sampling points.
        """

        real_length = self.window_real[1] - self.window_real[0]
        imag_length = self.window_imag[1] - self.window_imag[0]
        return max([real_length, imag_length])/n

    def mesh(self, n: int):
        """
        Generate a rectangular complex mesh corresponding to the viewing window of the domain.
        
        The distance between axis points is defined by invoking the *.spacing(n) method.

        Parameters:
        ----------
        n: int
            Number of sampling points for the longer axis of the viewing window.

        Returns:
        -------
        numpy.array of complex type
            Viewing window mesh as a 2D array of complex values.
        """

        real_length = self.window_real[1] - self.window_real[0]
        imag_length = self.window_imag[1] - self.window_imag[0]
        spacing = self.spacing(n)
        
        real_axis = np.linspace(self.window_real[0], self.window_real[1], ceil(real_length/spacing))
        imag_axis = np.linspace(self.window_imag[0], self.window_imag[1], ceil(imag_length/spacing))
        x, y = np.meshgrid(real_axis, imag_axis)
        return x + 1j*y
    
    def inmask(self, n: int):
        """
        Generate a boolean mask with True values marking mesh points that belong to the domain.

        Parameters:
        ----------
        n: int
            Number of sampling points for the longer axis of the viewing window.

        Returns:
        -------
        numpy.array of bool type
            Boolean mask of viewing window mesh with True values marking points that belong to the domain.
        """

        z = self.mesh(n)
        return self.infunc(z)
    
    def outmask(self, n: int):
        """
        Generate a boolean mask with True values marking mesh points that do NOT belong to the domain.

        Parameters:
        ----------
        n: int
            Number of sampling points for the longer axis of the viewing window.

        Returns:
        -------
        numpy.array of bool type
            Boolean mask of viewing window mesh with True values marking points that do NOT belong to the domain.
        """

        return np.logical_not(self.inmask(n))
        
    def domain(self, n: int):
        """
        Generate a rectangular complex mesh corresponding to the domain.

        The size of the mesh corresponds to the viewing window, but values 
        outside of the domain are set to numpy.nan. The distance between axis 
        points is defined by invoking the *.spacing(n) method.

        Parameters:
        ----------
        n: int
            Number of sampling points for the longer axis of the viewing window.

        Returns:
        -------
        numpy.array of complex type
            Domain mesh as a 2D array of complex values.
        """

        mesh = self.mesh(n)
        mesh[~self.outmask(n)] = np.nan
        return mesh
    
    def union(self, a):
        """
        Create a union of Domain with another instance of Domain.

        Viewing window is updated to contain viewing windows of both parent domains.

        Parameters:
        ----------
        a: complexplorer.Domain
            Domain instance.

        Returns:
        -------
        complexplorer.Domain
            Domain instance corresponding to the union of input domains.
        """

        left = min(list(self.window_real) + list(a.window_real))
        right = max(list(self.window_real) + list(a.window_real))
        bottom = min(list(self.window_imag) + list(a.window_imag))
        top = max(list(self.window_imag) + list(a.window_imag))
        infunc = lambda z: np.logical_or(self.infunc(z), a.infunc(z))
        return Domain((left, right), (bottom, top), infunc=infunc, square=self.square)
    
    def intersection(self, a):
        """
        Create an intersection of Domain with another instance of Domain.

        Viewing window is updated to contain viewing windows of both parent domains.
        This operation may result in a larger viewing window that needed to show 
        the domain, but viewing window can be adjusted by direct manipulation of 
        *.real and *.imag attribute values.

        Parameters:
        ----------
        a: complexplorer.Domain
            Domain instance.

        Returns:
        -------
        complexplorer.Domain
            Domain instance corresponding to the intersection of input domains.
        """

        left = min(self.window_real + a.window_real)
        right = max(self.window_real + a.window_real)
        bottom = min(self.window_imag + a.window_imag)
        top = max(self.window_imag + a.window_imag)
        infunc = lambda z: np.logical_and(self.infunc(z), a.infunc(z))
        return Domain((left, right), (bottom, top), infunc=infunc, square=self.square)
    
class Rectangle(Domain):
    def __init__(self, real: float, imag: float, center: complex = 0+0.0j, square: bool = True):
        """
        Create a Rectangle domain.

        Parameters:
        ----------
        real: float
            Length of the real axis of the domain.
        imag: float
            Length of the imaginary axis of the domain.
        center: complex, optional
            Defines the center of the domain. The default value is the origin of the complex plane.
        square: bool, optional
            If True the size of the viewing window is constrained to a square with a side 
            corresponding to the larger axis of the rectangle.

        Returns:
        -------
        complexplorer.Domain
            Domain instance corresponding to input rectangle.
        """

        real = abs(real)/2
        imag = abs(imag)/2
        window_real = (center.real - real, center.real + real)
        window_imag = (center.imag - imag, center.imag + imag)
        def infunc(z):
            a = np.greater_equal(np.real(z), window_real[0]).astype(int)
            b = np.greater_equal(window_real[1], np.real(z)).astype(int)
            c = np.greater_equal(np.imag(z), window_imag[0]).astype(int)
            d = np.greater_equal(window_imag[1], np.imag(z)).astype(int)
            return np.equal(a + b + c + d, 4)

        super().__init__(window_real, window_imag, infunc=infunc, square=square)

class Disk(Domain):
    def __init__(self, radius: float, center: complex = 0+0.0j):
        """
        Create a Disk domain.

        Parameters:
        ----------
        radius: float
            Radius of the disk.
        center: complex, optional
            Defines the center of the domain. The default value is the origin of the complex plane.

        Returns:
        -------
        complexplorer.Domain
            Domain instance corresponding to input disk.
        """

        if radius <= 0:
            raise ValueError('Radius must be positive')
        
        window_real = (center.real - radius, center.real + radius)
        window_imag = (center.imag - radius, center.imag + radius)
        infunc = lambda x: np.less_equal(np.absolute(x - center), radius)
        super().__init__(window_real, window_imag, infunc=infunc)

class Annulus(Domain):
    def __init__(self, radius_inner: float, radius_outer: float, center: complex = 0+0.0j):
        """
        Create an Annulus domain.

        Parameters:
        ----------
        radius_inner: float
            Inner radius of the annulus region.
        radius_outer: float
            Outer radius of the annulus region.
        center: complex, optional
            Defines the center of the domain. The default value is the origin of the complex plane.

        Returns:
        -------
        complexplorer.Domain
            Domain instance corresponding to input annulus.
        """

        if radius_inner <= 0:
            raise ValueError('radius_inner must be positive')
        if radius_outer <= radius_inner:
            raise ValueError('radius_outer must be greater than radius_inner')
        
        window_real = (center.real - radius_outer, center.real + radius_outer)
        window_imag = (center.imag - radius_outer, center.imag + radius_outer)
        def infunc(x):
            belongs_inside_outer = np.less_equal(np.absolute(x - center), radius_outer)
            belongs_outside_inner = np.greater_equal(np.absolute(x - center), radius_inner)
            return np.logical_and(belongs_inside_outer, belongs_outside_inner)
        super().__init__(window_real, window_imag, infunc=infunc)
