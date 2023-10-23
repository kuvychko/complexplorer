import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from complexplorer.cmap import Phase, Cmap
from complexplorer.domain import *
from typing import Callable, Optional

def plot(
        domain: Optional[Domain],
        func: Optional[Callable] = None,
        z=None,
        f=None,
        n: int = 400,
        cmap: Cmap = Phase(6, 0.5),
        ax = None,
        title=None,
        ):
    
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

    RGB = cmap.rgb(f, mask=mask)
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
        plt.show()
    else:
        ax.imshow(RGB, origin="lower", extent=extent, aspect=aspect)
        ax.set_xlabel("Re(z)")
        ax.set_ylabel("Im(z)")
        ax.set_title(title)
        return ax
    
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
        plt.show()
    else:
        ax.imshow(RGB, origin="lower", aspect=1, extent=[-margin, margin, -margin, margin])
        ax.set_xlabel(f"Re({arg_label})")
        ax.set_ylabel(f"Im({arg_label})")
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(tick_labels)
        ax.set_yticklabels(tick_labels)
    return ax
