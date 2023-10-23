import numpy as np
import matplotlib.pyplot as plt
from complexplorer.cmap import Phase, Cmap
from complexplorer.domain import *
from typing import Callable, Optional

def plot_landscape(
        domain: Optional[Domain], z=None, f=None, func: Optional[Callable] = None, n: int = 100, cmap: Cmap = Phase(6), ax = None,
        antialiased=False,
        zaxis_log=False):
    
    if domain is None and z is None:
        msg = 'both domain and z parameters cannot be None'
        raise ValueError(msg)

    if f is None and func is None:
        msg = 'both f and func parameters cannot be None'
        raise ValueError(msg)

    if z is None:
        z = domain.mesh(n)
        mask = domain.outmask(n)
    else:
        mask = None

    if f is None: f = func(z)
    if mask is not None: f[mask] = np.nan

    RGB = cmap.rgb(f, mask=mask)
    z_coord = np.abs(f)
    if zaxis_log:
        with np.errstate(divide='ignore', invalid='ignore'):
            z_coord = np.log(z_coord)

    # Plot the surface
    return_ax = True
    if ax is None:
        return_ax = False
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.plot_surface(np.real(z), np.imag(z), z_coord, rcount=n, ccount=n, facecolors=RGB,
                           linewidth=0, antialiased=antialiased)
    ax.set_xlabel("Re(z)")
    ax.set_ylabel("Im(z)")
    if return_ax: return ax
