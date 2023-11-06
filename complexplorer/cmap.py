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


The classes provided are:

- `Cmap`: This class serves as a base class for color maps and defines 
an informal interface for other color map classes. It implements 
the `*.hsv()` and `*.rgb()` methods which are used to convert 
input complex values to HSV and RGB-valued arrays.
"""


OUT_OF_DOMAIN_COLOR_HSV = (0., 0.01, 0.9)

class Cmap():
    "Cmap class defines a function that returns a color map corresponding to input complex matrix"

    def __init__(self, out_of_domain_hsv=OUT_OF_DOMAIN_COLOR_HSV):
        self.out_of_domain_hsv = out_of_domain_hsv

    def hsv_tuple(self, z):
        raise NotImplementedError(".hsv_tuple method not implemented")
    
    def hsv(self, z, mask=None):

        H, S, V = self.hsv_tuple(z)
        # applying domain mask
        if mask is not None:
            H[mask] = self.out_of_domain_hsv[0]
            S[mask] = self.out_of_domain_hsv[1]
            V[mask] = self.out_of_domain_hsv[2]
        HSV = np.dstack((H,S,V))
        return HSV
    
    def rgb(self, z, mask=None):
        HSV = self.hsv(z, mask=mask)
        RGB = colors.hsv_to_rgb(HSV)
        return RGB
    
class Phase(Cmap):
    def __init__(self, n_phi: Optional[int] = None, r_linear_step: float = None,
                 r_log_base: Optional[float] = None, v_base: float = 0.5, out_of_domain_hsv=OUT_OF_DOMAIN_COLOR_HSV):
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
    def __init__(self, spacing: float = 1., center: complex = 0+0.0j, out_of_domain_hsv=OUT_OF_DOMAIN_COLOR_HSV):
        self.center = center
        self.spacing = spacing
        self.out_of_domain_hsv = out_of_domain_hsv

    def hsv_tuple(self, z):
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
    def __init__(self, n_phi: Optional[int] = None, spacing: float = 1, r_log = None, out_of_domain_hsv=OUT_OF_DOMAIN_COLOR_HSV):
        self.phi = np.pi / int(n_phi)
        self.spacing = spacing
        self.r_log = r_log
        self.out_of_domain_hsv = out_of_domain_hsv

    def hsv_tuple(self, z):
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
    def __init__(self, log_spacing: float = 0.2, out_of_domain_hsv=OUT_OF_DOMAIN_COLOR_HSV):
        self.log_spacing = log_spacing
        self.out_of_domain_hsv = out_of_domain_hsv

    def hsv_tuple(self, z):
        S = np.zeros_like(z, dtype=float)
        H = np.ones_like(z, dtype=float) # fill values do not matter
        V = np.zeros_like(z, dtype=float) # this defines a white plane
        r = np.log(np.abs(z)) / self.log_spacing
        r_mod = np.mod(r, 2)
        V = np.less_equal(r_mod, 1)
        return H, S, V
