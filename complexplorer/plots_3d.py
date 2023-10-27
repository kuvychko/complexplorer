import numpy as np
import matplotlib.pyplot as plt
from complexplorer.cmap import Phase, Cmap
from complexplorer.domain import *
from complexplorer.funcs import stereographic
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

def riemann(func, n=80, cmap=Phase(6, 0.6), project_from_north=False, ax=None):

    tol = 1E-8
    psi_axis = np.linspace(tol, (1-tol), n) * np.pi
    theta_axis = np.linspace(0, 2*np.pi, n)
    theta_grid, psi_grid = np.meshgrid(theta_axis, psi_axis)
        
    # converting from points on Riemann sphere in spherical coordinates to complex values
    r = np.sin(psi_grid) / (1 - np.cos(psi_grid))
    z = r*np.exp(1.0j * theta_grid)
    RGB = cmap.rgb(func(z))

    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.plot_surface(*stereographic(z, project_from_north), facecolors=RGB, rcount=n, ccount=n,
                               linewidth=0, antialiased=False)
        ax.set_axis_off()
        ax.set_proj_type('ortho')
        ax.set_aspect('equal')
        ax.view_init(20, 45)
        # ax.set_title("Phase portrait on Riemann sphere: $f(z) = \dfrac{z^5 - 1}{z^{10} + 0.1}$", fontsize=14)
        fig.tight_layout(rect=[-0.2, -0.2, 1.5, 1.5])

    else:
        ax.plot_surface(*stereographic(z, project_from_north), facecolors=RGB, rcount=n, ccount=n,
                               linewidth=0, antialiased=False)
        ax.set_proj_type('ortho')
        ax.set_aspect('equal')
        return ax
