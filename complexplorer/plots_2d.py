import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from complexplorer.cmap import Phase, Cmap
from complexplorer.domain import *
from typing import Callable, Optional


"""
This module contains functions for plotting complex functions on the complex plane as 2D color maps.

Functions:
----------

- `plot`: plot complex function as pullback of the color map of the co-domain to the domain.

- `pair_plot`: plot color maps of the domain and the pullback of the co-domain of the function.

- `riemann_chart`: plot the phase portrait of a complex function projected from the Riemann hemisphere.

- `riemann_hemispheres`: plot a pair of phase portraits corresponding to the upper and lower hemispheres of the Riemann sphere.

"""

def plot(
        domain: Optional[Domain],
        func: Optional[Callable] = None,
        z=None,
        f=None,
        n: int = 400,
        cmap: Cmap = Phase(6, 0.5),
        ax = None,
        title=None,
        filename=None,
        ):
    """
    Plot complex function as pullback of the color map of the co-domain to the domain.
    
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
        Default is 400.
    cmap: Cmap
        Color map to be used. Default is Phase(6, 0.5).
    ax: matplotlib.axes.Axes
        Axes object where to plot the function. If None, a new figure is created.
        Default is None.
    title: str
        Title of the plot. Default is None.
    filename: str
        Name of the file to save the figure to. If None, the figure is not saved.
        Default is None.
    
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

    RGB = cmap.rgb(f, outmask=mask)
    extent = [
        np.real(z).min(),
        np.real(z).max(),
        np.imag(z).min(),
        np.imag(z).max(),
    ]
    aspect = (np.real(z).max() - np.real(z).min()) / (np.imag(z).max() - np.imag(z).min())

    if ax is None:
        plt.imshow(RGB, origin="lower", extent=extent, aspect=aspect)
        plt.xlabel("Re(z)")
        plt.ylabel("Im(z)")
        plt.title(title)
        if filename: plt.savefig(filename)
    else:
        ax.imshow(RGB, origin="lower", extent=extent, aspect=aspect)
        ax.set_xlabel("Re(z)")
        ax.set_ylabel("Im(z)")
        ax.set_title(title)
        return ax
    
def pair_plot(
        domain: Optional[Domain],
        func: Optional[Callable] = None,
        z=None,
        f=None,
        n: int = 400,
        cmap: Cmap = Phase(6, 0.5),
        title=None,
        figsize=(10,5),
        filename=None,
        ):
    """
    Plot color maps of the domain and the pullback of the co-domain of the function.

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
    filename: str
        Name of the file to save the figure to. If None, the figure is not saved.
        Default is None.

    """
    
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    plot(domain=domain, func=(lambda x: x), z=z, f=f, n=n, cmap=cmap, title='Domain z', ax=ax0)
    plot(domain=domain, func=func, z=z, f=f, n=n, cmap=cmap, title='Co-domain f(z)', ax=ax1)
    fig.suptitle(title)
    plt.tight_layout()
    if filename: plt.savefig(filename)
    
def riemann_chart(
        func: Callable,
        domain: Optional[Domain] = None,
        n: int = 100,
        show_south_hemisphere=True,
        project_from_north=True,
        cmap: Cmap = Phase(6, 0.5),
        ax = None,
        margin=0.05,
        unit_circle_width: float = 1.,
        ):
    """
    Plot the phase portrait of a complex function projected from the Riemann hemisphere.

    Parameters:
    -----------
    func: Callable
        Complex function.
    domain: Domain
        Domain object.
    n: int
        Number of points along the longest side of the domain's mesh.
        Default is 100.
    show_south_hemisphere: bool
        If True, the lower hemisphere is shown. If False, the upper hemisphere is shown.
        Default is True.
    project_from_north: bool
        If True, stereographic projection is performed from the north pole. 
        This is a standard convention in complex analysis, however it inverts the color map 
        resulting in poles looking like zeros and vice versa. If False, stereographic projection 
        is performed from the south pole. Default is True.
    cmap: Cmap
        Color map to be used. Default is Phase(6, 0.5) - an enhanced phase portrait for both 
        the phase and modulus of the function.
    ax: matplotlib.axes.Axes
        Axes object where to plot the function. If None, a new figure is created.
        Default is None.
    margin: float
        Margin around the unit disk. Default is 0.05.
    unit_circle_width: float
        Width of the black line used to highlight the unit circle. Default is 1.

    """

    if margin < 0: raise ValueError('Margin value cannot be negative')
    if margin > 0.5: raise ValueError('Margin value cannot exceed 0.5')

    margin = 1 + margin
    # a 2-square origin centered Domain is used
    dom = Rectangle(2*margin, 2*margin)
    if domain: dom.mask_list = domain.mask_list
    z = dom.mesh(n)

    if not project_from_north:
        show_south_hemisphere = not show_south_hemisphere

    # evaluating function
    if show_south_hemisphere:
        F = func(z)
        arg_label = 'z'
        tick_labels = [-1, -0.5, 0, 0.5, 1]
    else:
        F = func(1/z)
        arg_label = '1/z'
        tick_labels = [-1, -2, 'inf', 2, 1]

    H, S, V = cmap.hsv_tuple(F)
    # adjusting saturation outside of the unit disk
    S = np.abs(z) < 1
    S = (S + 0.7)/1.7

    # setting V to black to draw unit circle
    unit_circle_width_tol = dom.spacing(n) * unit_circle_width
    UC_mask = np.isclose(np.abs(z), np.ones(z.shape, dtype=float), atol=unit_circle_width_tol)
    V[UC_mask] = 0
    
    HSV = np.dstack((H,S,V))
    RGB = colors.hsv_to_rgb(HSV)
    
    ticks = [-1, -0.5, 0, 0.5, 1]
    
    if ax is None:
        plt.imshow(RGB, origin="lower", aspect=1, extent=[-margin, margin, -margin, margin])
        plt.xlabel(f"Re({arg_label})")
        plt.ylabel(f"Im({arg_label})")
        plt.xticks(ticks=ticks, labels=tick_labels)
        plt.yticks(ticks=ticks, labels=tick_labels)
    else:
        ax.imshow(RGB, origin="lower", aspect=1, extent=[-margin, margin, -margin, margin])
        ax.set_xlabel(f"Re({arg_label})")
        ax.set_ylabel(f"Im({arg_label})")
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(tick_labels)
        ax.set_yticklabels(tick_labels)
    return ax

def riemann_hemispheres(
        func,
        title=None,
        n: int = 400,
        margin=0.05,
        unit_circle_width: float = 1.,
        figsize=(12,4),
        filename=None):
    """
    Plot a pair of phase portraits corresponding to the upper and lower hemispheres of the Riemann sphere.

    Parameters:
    -----------
    func: Callable
        Complex function.
    title: str
        Title of the plot. Default is None.
    n: int
        Number of points along the longest side of the domain's mesh.
        Default is 400.
    margin: float
        Margin around the unit disk. Default is 0.05.
    unit_circle_width: float
        Width of the black line used to highlight the unit circle. Default is 1.
    figsize: tuple
        Figure size. Default is (12,4).
    filename: str
        Name of the file to save the figure to. If None, the figure is not saved.
        Default is None.
    """

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    riemann_chart(func, n=n, show_south_hemisphere=True, ax = axes[0], margin=margin, unit_circle_width=unit_circle_width)
    riemann_chart(func, n=n, show_south_hemisphere=False, ax = axes[1], margin=margin, unit_circle_width=unit_circle_width)

    axes[0].set_title("South (lower) hemisphere")
    axes[1].set_title("North (upper) hemisphere")

    if title: fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, .8, .8])
    fig.subplots_adjust(left=0, right=1, bottom=0, top=.85)
    if filename: plt.savefig(filename)
