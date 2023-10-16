from math import ceil
from functools import reduce
import numpy as np
from typing import Optional, Tuple


def phase(z):
    "Returns phase of complex number in [0, 2*pi) interval."

    phi = np.angle(z)
    # this line converts angles from [pi, -pi) to [0, 2*pi)
    phi[phi<0] = 2*np.pi + phi[phi<0]
    return phi


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
    def __init__(self, real:Tuple, imag: Tuple, mask_dict: Optional[dict] = None):
        if real[0] == real[1]:
            msg = f"First and second values of real input cannot be equal"
            raise ValueError(msg)
        if imag[0] == imag[1]:
            msg = f"First and second values of imag input cannot be equal"
            raise ValueError(msg)
        
        self._real_range = (min(real), max(real))
        self._imag_range = (min(imag), max(imag))
        self.mask_dict = mask_dict

    def mesh(self, n):
        """
        Return a rectangular complex mesh.
        
        The distance between axis points is defined by dividing the longer axis by input n.
        """

        # calculating point spacing for real and imaginary axes
        real_length = self._real_range[1] - self._real_range[0]
        imag_length = self._imag_range[1] - self._imag_range[0]
        if real_length >= imag_length:
            spacing = real_length / n
        else:
            spacing = imag_length / n
        
        real_axis = np.linspace(self._real_range[0], self._real_range[1], ceil(real_length/spacing))
        imag_axis = np.linspace(self._imag_range[0], self._imag_range[1], ceil(imag_length/spacing))
        x, y = np.meshgrid(real_axis, imag_axis)
        z = x + 1j*y
        return z
    
    def mask(self, n):
        """
        Return a boolean mask which defines a valid domain within the rectangular mesh region.
        """

        z = self.mesh(n)
        mask = np.full_like(z, True, dtype=bool)
        if self.mask_dict is None:
            return mask
        else:
            # the loop is needed if mask_dict contains multiple mask functions
            mask_array_list = []
            for func in self.mask_dict.values():
                mask_array_list.append(func(z))
            # by defining "out" argument in np.logical_and numpy doesn't need to allocate a new array for each output
            mask = reduce(lambda x,y: np.logical_and(x, y, out=mask), mask_array_list)
            return mask
        
    def domain(self, n):
        """
        Return a domain mesh (rectangular mesh with values outside of the domain set to np.nan)
        """
        mesh = self.mesh(n)
        mesh[~self.mask(n)] = np.nan
        return mesh

class Rectangle(Domain):
    def __init__(self, real: float, imag: float, center: complex = 0+0.0j):
        """
        Intialize a Domain intance corresponding to Rectangle centered at center.
        """

        real = abs(real)/2
        imag = abs(imag)/2
        real_range = (center.real - real, center.real + real)
        imag_range = (center.imag - imag, center.imag + imag)
        super.__init__(real_range, imag_range)

class Disk(Domain):
    def __init__(self, radius: float, center: complex = 0+0.0j):
        """
        Intialize a Domain intance corresponding to Rectangle centered at center.
        """

        if radius <= 0:
            raise ValueError('Radius must be positive')
        
        real_range = (center.real - radius, center.real + radius)
        imag_range = (center.imag - radius, center.imag + radius)
        mask_func = lambda x: np.less_equal(np.absolute(x - center), radius)
        super.__init__(real_range, imag_range, mask_dict={'disk_mask': mask_func})
       
