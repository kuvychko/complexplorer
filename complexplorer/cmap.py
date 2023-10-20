import numpy as np
import matplotlib.colors as colors
from typing import Optional, Tuple
from complexplorer.domain import Domain


def phase(z: complex):
    "Return a phase of complex input mapped to [0, 2*pi) interval"

    phi = np.angle(z)
    # convert phase angles from [pi, -pi) to [0, 2*pi)
    phi[phi<0] = 2*np.pi + phi[phi<0]
    return phi

class Cmap():
    "Cmap class defines a function that returns a color map corresponding to input complex matrix"
    def hsv(self, z):
        raise NotImplementedError(".hsv method not implemented")
    
    def rgb(self, z):
        raise NotImplementedError(".rgb method not implemented")
    
class Phase(Cmap):
    def __init__(self, phi_split: Optional[int] = None, r_log_base: Optional[float] = None):
        self.phi_split = phi_split
        self.r_log_base = r_log_base

    def hsv(self, z, V_base: float = 0.5):
        if V_base < 0 | V_base > 1:
            raise ValueError("V_base must be within [0, 1) interval")
        phi = phase(z)
        S = np.ones_like(z)
        if self.phi_split is not None:
            V_phi = np.ceil(phi / (np.pi/self.phi_split)) - phi / (np.pi/self.phi_split)
        else:
            V_phi = np.ones_like(z)
        
        if self.r_scale is not None:
            log_z = np.log(np.abs(z)) / np.log(self.r_log_base) # converting log from natural to r_log_base
            V_r = np.ceil(log_z) - log_z
        else:
            V_r = np.ones_like(z)
        V_scaler = 1 - V_base
        V = (V_phi + V_r) * V_scaler + V_base # scaling sum of V_phi and V_r to [0, 1] interval
        H = phi/(2*np.pi)
        HSV = np.dstack((H,S,V))
        return HSV
    
    def rgb(self, z, V_base: float = 0.5):
        HSV = self.hsv(z, V_base=V_base)
        RGB = colors.hsv_to_rgb(HSV)
        return RGB

class Chessboard(Cmap):
    def __init__(self, spacing: float = 1., center: complex = 0+0.0j):
        self.center = center
        self.spacing = spacing

    def hsv(self, z):
        S = np.zeros_like(z)
        H = np.ones_like(z) # fill values do not matter
        V = np.zeros_like(z) # this defines a white plane
        z = (z - self.center)/self.spacing # adjusting origin and scaling by spacing
        real = np.real(z)
        real_mod = np.mod(real, 2) <= 0
        real_bool = np.less_equal(real_mod, 1)
        imag = np.imag(z)
        imag_mod = np.mod(imag, 2)
        imag_bool = np.less_equal(imag_mod, 1)
        V = np.logical_and(real_bool, imag_bool)
        HSV = np.dstack((H,S,V))
        return HSV
    
    def rgb(self, z):
        HSV = self.hsv(z)
        RGB = colors.hsv_to_rgb(HSV)
        return RGB

class PolarChessboard(Cmap):
    pass
