import numpy as np
import matplotlib.colors as colors
from typing import Optional, Tuple
from complexplorer.domain import Domain
from complexplorer.funcs import phase, sawtooth

"""
This module contains a set of classes for the construction of color maps.

A color map is represented by a class which defines a function that 
converts input complex values into numpy arrays of HSV or RGB values, 
with individual H/S/V or R/G/B values mapped to [0, 1] interval.
Input values which correspond to the True values of the like-shaped 
outmask array (values outside of the domain) are set to 
OUT_OF_DOMAIN_COLOR_HSV color.

Classes:
--------

- `Cmap`: This class serves as a base class for color maps and defines 
an informal interface for child color map classes. It implements 
the `*.hsv()` and `*.rgb()` methods which are used to convert 
input complex values to HSV and RGB-valued arrays.

- `Phase`: This class implements a phase color map. It can be used
to generate regular phase color maps or enhanced phase color maps.

- `Chessboard`: This class implements a chessboard color map.

- `PolarChessboard`: This class implements a polar chessboard color map.

- `LogRings`: This class implements a logarithmic black and white rings color map.

"""

# default out of domain color (gray)
OUT_OF_DOMAIN_COLOR_HSV = (0., 0.01, 0.9)

class Cmap():
    "Cmap class defines a function that returns a color map corresponding to input complex matrix"

    def __init__(self, out_of_domain_hsv: Optional[Tuple] = OUT_OF_DOMAIN_COLOR_HSV):
        """
        Base Cmap constructor.
        
        Parameters:
        ----------
        out_of_domain_hsv: 3-tuple, optional
            3-tuple of HSV values corresponding to the color of complex points with values
            outside of the domain. The default is OUT_OF_DOMAIN_COLOR_HSV (gray).
        
        """

        self.out_of_domain_hsv = out_of_domain_hsv

    def hsv_tuple(self, z):
        """
        Return a 3-tuple of (H, S, V) arrays corresponding to the input z array of complex values.

        HSV values are mapped to [0, 1] interval.

        Parameters:
        ----------
        z: numpy.array
            Array of complex values
        
        """

        raise NotImplementedError(".hsv_tuple method not implemented")
    
    def hsv(self, z, outmask=None):
        """
        Return a numpy array of HSV values corresponding to the input z array of complex values.

        HSV values are mapped to [0, 1] interval.

        Parameters:
        ----------
        z: numpy.array
            Array of complex values
        outmask: numpy.array
            Boolean array of the same shape as z input, with True values corresponding to points 
            outside of the domain.

        """

        H, S, V = self.hsv_tuple(z)
        # applying domain outmask
        if outmask is not None:
            H[outmask] = self.out_of_domain_hsv[0]
            S[outmask] = self.out_of_domain_hsv[1]
            V[outmask] = self.out_of_domain_hsv[2]
        HSV = np.dstack((H,S,V))
        return HSV
    
    def rgb(self, z, outmask=None):
        """
        Return a numpy array of RGB values corresponding to the input z array of complex values.

        RGB values are mapped to [0, 1] interval.

        Parameters:
        ----------
        z: numpy.array
            Array of complex values
        outmask: numpy.array
            Boolean array of the same shape as z input, with True values corresponding to points 
            outside of the domain.
            
        """

        HSV = self.hsv(z, outmask=outmask)
        RGB = colors.hsv_to_rgb(HSV)
        return RGB
    
class Phase(Cmap):
    def __init__(self,
                 n_phi: Optional[int] = None,
                 r_linear_step: float = None,
                 r_log_base: Optional[float] = None,
                 v_base: float = 0.5,
                 out_of_domain_hsv=OUT_OF_DOMAIN_COLOR_HSV):
        """
        Phase color map constructor.

        Class implements both regular and enhanced phase color maps.
        If any of the n_phi, r_linear_step, r_log_base are not None, 
        output phase portrait is enhanced. If n_phi value is given,
        the phase is used for enhancement, if either (or both) r_linear_step and 
        r_log_base are given then modulus is used for enhancement.

        Parameters:
        ----------
        n_phi: int, optional
            Number of sectors used for enhanced color mapping of complex phase.
            The default is None (no enhanced color mapping of complex phase).
        r_linear_step: float, optional
            Linear step value is used to divide input complex values and normalize 
            the modulus which is used for corresponding enhanced color mapping via 
            a sawtooth function.
        r_log_base: float, optional
            Logarithm base used to calculate the log of the modulus of the input 
            values prior to evaluating a sawtooth function and generating 
            a logarithmic enhanced color map.
        v_base: float, optional
            Base value for the V (must belog to [0, 1) interval). Smaller values 
            produce darker grades of gray (darker enhanced effect), while larger 
            values give lighter grades of gray.
        out_of_domain_hsv: 3-tuple, optional
            3-tuple of HSV values corresponding to the color of complex points with values
            outside of the domain. The default is OUT_OF_DOMAIN_COLOR_HSV (gray).

        """

        if v_base < 0 or v_base > 1:
            raise ValueError("v_base must be within [0, 1) interval")
        if n_phi is not None:
            self.phi = np.pi / int(n_phi)
        else:
            self.phi = None
        self.r_linear_step = r_linear_step
        self.r_log_base = r_log_base
        self.v_base = v_base
        self.out_of_domain_hsv = out_of_domain_hsv

    def hsv_tuple(self, z):
        """
        Return a 3-tuple of (H, S, V) arrays corresponding to the input z array of complex values.

        HSV values are mapped to [0, 1] interval.

        Parameters:
        ----------
        z: numpy.array
            Array of complex values        

        """
        
        phi = phase(z)
        S = np.ones_like(z, dtype=float)
        if self.phi is not None:
            V_phi = sawtooth(phi/ self.phi)
        else:
            V_phi = np.ones_like(z, dtype=float)
        
        if self.r_linear_step and self.r_log_base is None:
            V_r = sawtooth(np.abs(z)/self.r_linear_step)
        elif self.r_linear_step is None and self.r_log_base:
            V_r = sawtooth(np.abs(z), self.r_log_base)
        elif self.r_linear_step and self.r_log_base:
            V_r = sawtooth(np.abs(z)/self.r_linear_step, self.r_log_base)
        else:
            V_r = np.ones_like(z, dtype=float)
        V_scaler = 1 - self.v_base
        V = (V_phi + V_r) * V_scaler / 2 + self.v_base # scaling sum of V_phi and V_r to [0, 1] interval
        H = phi/(2*np.pi)
        return H, S, V

class Chessboard(Cmap):
    """
    Chessboard color map constructor.
    
    Class implements a chessboard color map with a given spacing and center.
    """

    def __init__(self, spacing: float = 1., center: complex = 0+0.0j, out_of_domain_hsv=OUT_OF_DOMAIN_COLOR_HSV):
        """
        Chessboard color map constructor.
        
        Parameters:
        ----------
        spacing: float, optional
            Spacing between chessboard squares. The default is 1.
        center: complex, optional
            Complex value corresponding to the center of the chessboard. The default is 0+0.0j.
        out_of_domain_hsv: 3-tuple, optional
            3-tuple of HSV values corresponding to the color of complex points with values
            outside of the domain. The default is OUT_OF_DOMAIN_COLOR_HSV (gray).

        """

        self.center = center
        self.spacing = spacing
        self.out_of_domain_hsv = out_of_domain_hsv

    def hsv_tuple(self, z):
        """
        Return a 3-tuple of (H, S, V) arrays corresponding to the input z array of complex values.
        
        Parameters:
        ----------
        z: numpy.array
            Array of complex values

        """

        S = np.zeros_like(z, dtype=float)
        H = np.ones_like(z, dtype=float) # fill values do not matter
        V = np.zeros_like(z, dtype=float) # this defines a white plane
        z = (z - self.center)/self.spacing # adjusting origin and scaling by spacing
        real = np.real(z)
        real_mod = np.mod(real, 2)
        real_bool = np.less_equal(real_mod, 1)
        imag = np.imag(z)
        imag_mod = np.mod(imag, 2)
        imag_bool = np.less_equal(imag_mod, 1)
        V0 = np.logical_and(real_bool, imag_bool)
        V1 = np.logical_and(~real_bool, ~imag_bool)
        V = np.logical_or(V0, V1)
        return H, S, V

class PolarChessboard(Cmap):
    """
    Polar chessboard color map constructor.
    """

    def __init__(
        self,
        n_phi: int = 6,
        spacing: float = 1,
        r_log = None,
        out_of_domain_hsv=OUT_OF_DOMAIN_COLOR_HSV
        ):
        """
        Polar chessboard color map constructor.

        Parameters:
        ----------
        n_phi: int, optional
            Phase spacing between chessboard squares in number of sectors. 
            The default is 6.
        spacing: float, optional
            Modulus spacing between chessboard squares. The default is 1.
        r_log: float, optional
            Logarithm base used to calculate the log of the modulus prior to 
            building of the chessboard. The default is None (no logarithmic scaling of 
            modulus).
        out_of_domain_hsv: 3-tuple, optional
            3-tuple of HSV values corresponding to the color of complex points with values
            outside of the domain. The default is OUT_OF_DOMAIN_COLOR_HSV (gray).

        """

        self.phi = np.pi / int(n_phi)
        self.spacing = spacing
        self.r_log = r_log
        self.out_of_domain_hsv = out_of_domain_hsv

    def hsv_tuple(self, z):
        """
        Return a 3-tuple of (H, S, V) arrays corresponding to the input z array of complex values.

        Parameters:
        ----------
        z: numpy.array
            Array of complex values

        """

        S = np.zeros_like(z, dtype=float)
        H = np.ones_like(z, dtype=float) # fill values do not matter
        V = np.zeros_like(z, dtype=float) # this defines a white plane
        z = z / self.spacing # adjusting origin and scaling by spacing
        angle = phase(z) / self.phi
        angle_mod = np.mod(angle, 2)
        r = np.abs(z)
        if self.r_log is not None:
            # setting numpy to ignore divide by zero and invalid input errors locally
            # this only needed for r=0
            with np.errstate(divide='ignore', invalid='ignore'):
                r = np.log(r) / np.log(self.r_log)
        r_mod = np.mod(r, 2)
        angle_bool = np.less_equal(angle_mod, 1)
        r_bool = np.less_equal(r_mod, 1)
        V0 = np.logical_and(angle_bool, r_bool)
        V1 = np.logical_and(~angle_bool, ~r_bool)
        V = np.logical_or(V0, V1)
        return H, S, V

class LogRings(Cmap):
    """
    Logarithmic black and white rings color map constructor.
    """

    def __init__(self, log_spacing: float = 0.2, out_of_domain_hsv=OUT_OF_DOMAIN_COLOR_HSV):
        """
        Logarithmic black and white rings color map constructor.

        Parameters:
        ----------
        log_spacing: float, optional
            Logarithmic spacing between the rings. The default is 0.2.
        out_of_domain_hsv: 3-tuple, optional
            3-tuple of HSV values corresponding to the color of complex points with values
            outside of the domain. The default is OUT_OF_DOMAIN_COLOR_HSV (gray).

        """

        self.log_spacing = log_spacing
        self.out_of_domain_hsv = out_of_domain_hsv

    def hsv_tuple(self, z):
        """
        Return a 3-tuple of (H, S, V) arrays corresponding to the input z array of complex values.

        Parameters:
        ----------
        z: numpy.array
            Array of complex values

        """

        S = np.zeros_like(z, dtype=float)
        H = np.ones_like(z, dtype=float) # fill values do not matter
        V = np.zeros_like(z, dtype=float) # this defines a white plane
        r = np.log(np.abs(z)) / self.log_spacing
        r_mod = np.mod(r, 2)
        V = np.less_equal(r_mod, 1)
        return H, S, V
