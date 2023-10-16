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
    def __init__(self, phi_split: Optional[int] = None, r_scale: Optional[float] = None):
        self.phi_split = phi_split
        self.r_scale = r_scale

class Phase(Cmap):
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
            V_r = np.ceil(np.log(np.abs(z)) * self.r_scale) - np.log(np.abs(z)) * self.r_scale
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
