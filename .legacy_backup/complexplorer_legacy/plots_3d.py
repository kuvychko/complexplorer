import numpy as np
import matplotlib.pyplot as plt
from .cmap import Phase, Cmap
from .domain import *
from .funcs import stereographic
from typing import Callable, Optional


"""
The module contains functions for plotting complex functions on the complex plane as 3D landscapes and phase portraits on the Riemann sphere.

Functions:
----------

- `plot_landscape`: plot a complex function as a 3D landscape on the complex plane.

- `pair_plot_landscape`: - plot analytic landscapes of the domain and the pullback of the co-domain of the function.

- `riemann`: plot a complex function as a phase portrait on the Riemann sphere.

"""


def plot_landscape(
        domain: Optional[Domain], z=None, f=None, func: Optional[Callable] = None, n: int = 100, cmap: Cmap = Phase(6), ax = None,
        antialiased=False,
        zaxis_log=False,
        z_max=None,

        ):
    """
    Plot a complex function as a 3D landscape on the complex plane.

    The Z axis is the absolute value of the function, or its logarithm if zaxis_log is True.
    The color of the surface is defined by the selected color map (default is Phase(6) - 
    phase portrait enhanced along the phase dimension).

    Parameters:
    -----------
    domain: Domain
        Domain object. If None, z must be provided.
    func: Callable
        Complex function. If None, f must be provided.
    z: np.ndarray
        2D array of complex numbers corresponding to the domain. 
        If None, domain must be provided.
    f: np.ndarray
        2D array of complex numbers corresponding to the co-domain. 
        If None, func must be provided. Default is None.
    n: int
        Number of points along the longest side of the domain's mesh.
        Default is 100.
    cmap: Cmap
        Color map to be used. Default is Phase(6).
    ax: matplotlib.axes.Axes
        Axes object where to plot the function. If None, a new figure is created.
        Default is None.
    antialiased: bool
        If True, the plot will be antialiased. Default is False.
    zaxis_log: bool
        If True, the Z axis will be the logarithm of the absolute value of the function.
        Default is False.
    z_max: float
        Maximum value of the Z axis. If None, the maximum value is the maximum absolute value of the function.
        Default is None.

    Returns:
    --------
    ax: matplotlib.axes.Axes
        Axes object where the function is plotted (only returned if input ax is provided).
    """

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
    
    # Ensure f is always an array, even for constant functions
    f = np.asarray(f)
    if f.ndim == 0:  # scalar case
        f = np.full_like(z, f)
    
    if mask is not None: f[mask] = np.nan
    if z_max is not None and z_max <=0: raise ValueError('z_max must be positive or None')

    RGB = cmap.rgb(f, outmask=mask)
    z_coord = np.abs(f)
    if z_max is not None:
        z_coord = np.clip(z_coord, 0, z_max)
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

def pair_plot_landscape(
        domain: Optional[Domain],
        func: Optional[Callable] = None,
        z=None,
        f=None,
        n: int = 400,
        cmap: Cmap = Phase(6, 0.5),
        title=None,
        figsize=(10,5),
        zaxis_log=False,
        z_max=None,
        filename=None,
        ):
    """
    Plot analytic landscapes of the domain and the pullback of the co-domain of the function.

    Surfaces are colored according to the selected color map (default is Phase(6, 0.5)).

    Parameters:
    -----------
    domain: Domain
        Domain object. If None, z must be provided. Default is None.
    func: Callable
        Complex function. If None, f must be provided. Default is None.
    z: np.ndarray
        2D array of complex numbers corresponding to the domain. 
        If None, domain must be provided. Default is None.
    f: np.ndarray
        2D array of complex numbers corresponding to the co-domain. 
        If None, func must be provided. Default is None.
    n: int
        Number of points along the longest side of the domain's mesh.
        Default is 400.
    cmap: Cmap
        Color map to be used. Default is Phase(6, 0.5).
    title: str 
        Title of the plot. Default is None.
    figsize: tuple
        Figure size. Default is (10,5).
    zaxis_log: bool
        If True, the Z axis will be the logarithm of the absolute value of the function.
        Default is False.
    z_max: float
        Maximum value of the Z axis. If None, the maximum value is the maximum absolute value of the function.
        Default is None.
    filename: str
        If provided, the plot will be saved with the given name. Default is None.

    """
        
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=figsize, subplot_kw={"projection": "3d"})
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plot_landscape(domain=domain, func=(lambda x: x), z=z, f=f, n=n, cmap=cmap, ax=ax0, zaxis_log=zaxis_log, z_max=z_max)
    plot_landscape(domain=domain, func=func, z=z, f=f, n=n, cmap=cmap, ax=ax1, zaxis_log=zaxis_log, z_max=z_max)
    fig.suptitle(title)
    if filename: plt.savefig(filename)

def riemann(func, n=80, cmap=Phase(6, 0.6), project_from_north=False, ax=None, title=None, filename=None):
    """
    Plot a complex function as a phase portrait on the Riemann sphere.

    Parameters:
    -----------
    func: Callable
        Complex function.
    n: int
        Number of mesh points.
    cmap: Cmap
        Color map to be used. Default is Phase(6, 0.6).
    project_from_north: bool
        If True, the projection will be from the north pole of the Riemann sphere.
        Default is False. 
    ax: matplotlib.axes.Axes
        Axes object where to plot the function. If None, a new figure is created.
        Default is None.
    title: str
        Title of the plot. Default is None.
    filename: str
        Name of the file to save the figure to. If None, the figure is not saved.
        Default is None.

    Returns:
    --------
    ax: matplotlib.axes.Axes
        Axes object where the function is plotted (only returned if input ax is provided).
    """

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
        if title: ax.set_title(title, fontsize=14)
        if filename: plt.savefig(filename) # order matters here - must be before tight_layout otherwise margins are messed up
        fig.tight_layout(rect=[-0.2, -0.2, 1.5, 1.5])

    else:
        ax.plot_surface(*stereographic(z, project_from_north), facecolors=RGB, rcount=n, ccount=n,
                               linewidth=0, antialiased=False)
        ax.set_proj_type('ortho')
        ax.set_aspect('equal')
        return ax
