import matplotlib.pyplot as plt
from complexplorer.cmap import Phase, Cmap
from complexplorer.domain import *
from typing import Callable, Optional

def plot(domain: Optional[Domain], z=None, f=None, func: Optional[Callable] = None, n: int = 100, cmap: Cmap = Phase(6), ax = None):
    
    if domain is None and z is None:
        msg = 'both domain and z parameters cannot be None'
        raise ValueError(msg)

    if f is None and func is None:
        msg = 'both f and func parameters cannot be None'
        raise ValueError(msg)

    if z is None:
        z = domain.domain(n)

    if f is None:
        f = func(z)

    RGB = cmap.rgb(f)
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
        plt.title("$f(z) = z$")
        plt.show()
    else:
        ax.imshow(RGB, origin="lower", extent=extent, aspect=aspect)
        ax.set_xlabel("Re(z)")
        ax.set_ylabel("Im(z)")
        return ax
    