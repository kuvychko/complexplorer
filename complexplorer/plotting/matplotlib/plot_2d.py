"""2D plotting functions using matplotlib.

This module provides functions for visualizing complex functions
as 2D color maps using domain coloring techniques.
"""

from typing import Optional, Callable, Union, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ...core.domain import Domain, Rectangle
from ...core.colormap import Colormap, Phase
from ...utils.validation import (
    validate_domain_or_mesh, validate_function_or_values,
    validate_function, validate_colormap, validate_resolution,
    ValidationError
)
# from ..base import Base2DPlotter, PlotConfig  # TODO: Implement base classes


class Matplotlib2DPlotter:  # (Base2DPlotter):  # TODO: Add base class
    """2D plotter implementation using matplotlib."""
    
    def plot_single(self, 
                   domain: Domain,
                   func: Callable[[np.ndarray], np.ndarray],
                   colormap: Colormap,
                   resolution: int,
                   ax: Optional[Axes] = None,
                   title: Optional[str] = None,
                   config = None) -> Axes:  # config: Optional[PlotConfig] = None
        """Plot a single complex function visualization.
        
        Parameters
        ----------
        domain : Domain
            The domain to plot over.
        func : callable
            Complex function to visualize.
        colormap : Colormap
            Colormap for domain coloring.
        resolution : int
            Number of points along longest edge.
        ax : Axes, optional
            Matplotlib axes to plot on.
        title : str, optional
            Plot title.
        config : PlotConfig, optional
            Additional plot configuration.
            
        Returns
        -------
        Axes
            The matplotlib axes used for plotting.
        """
        # Get mesh and evaluate function
        z = domain.mesh(resolution)
        mask = domain.outmask(resolution)
        f_z = func(z)
        
        # Convert to RGB
        rgb = colormap.rgb(f_z, outmask=mask)
        
        # Calculate extent
        extent = [
            np.real(z).min(),
            np.real(z).max(),
            np.imag(z).min(),
            np.imag(z).max(),
        ]
        
        # Calculate aspect ratio
        aspect = (extent[1] - extent[0]) / (extent[3] - extent[2])
        
        # Create axes if not provided
        if ax is None:
            fig, ax = plt.subplots()
        
        # Plot the image
        ax.imshow(rgb, origin="lower", extent=extent, aspect=aspect)
        ax.set_xlabel("Re(z)")
        ax.set_ylabel("Im(z)")
        
        if title:
            ax.set_title(title)
            
        return ax
    
    def plot_pair(self,
                  domain: Domain,
                  func: Callable[[np.ndarray], np.ndarray],
                  colormap: Colormap,
                  resolution: int,
                  figsize: Tuple[float, float] = (10, 5),
                  title: Optional[str] = None,
                  config = None) -> Figure:  # config: Optional[PlotConfig] = None
        """Plot domain and codomain side by side.
        
        Parameters
        ----------
        domain : Domain
            The domain to plot over.
        func : callable
            Complex function to visualize.
        colormap : Colormap
            Colormap for domain coloring.
        resolution : int
            Number of points along longest edge.
        figsize : tuple, optional
            Figure size (width, height).
        title : str, optional
            Overall figure title.
        config : PlotConfig, optional
            Additional plot configuration.
            
        Returns
        -------
        Figure
            The matplotlib figure.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot domain (identity function)
        self.plot_single(domain, lambda z: z, colormap, resolution, 
                        ax=ax1, title="Domain z")
        
        # Plot codomain
        self.plot_single(domain, func, colormap, resolution,
                        ax=ax2, title="Codomain f(z)")
        
        if title:
            fig.suptitle(title)
        
        plt.tight_layout()
        return fig


def plot(domain: Optional[Domain] = None,
         func: Optional[Callable] = None,
         z: Optional[np.ndarray] = None,
         f: Optional[np.ndarray] = None,
         resolution: int = 400,
         cmap: Optional[Colormap] = None,
         ax: Optional[Axes] = None,
         title: Optional[str] = None,
         filename: Optional[str] = None) -> Optional[Axes]:
    """Plot complex function as domain coloring.
    
    This function provides a convenient interface for plotting complex
    functions using domain coloring. Either provide a domain and function,
    or directly provide the mesh arrays z and f.
    
    Parameters
    ----------
    domain : Domain, optional
        Domain object. If None, z must be provided.
    func : callable, optional
        Complex function. If None, f must be provided.
    z : ndarray, optional
        2D array of complex domain values.
    f : ndarray, optional
        2D array of complex codomain values.
    resolution : int, optional
        Resolution (number of points along longest edge).
    cmap : Colormap, optional
        Colormap to use. Defaults to enhanced phase portrait.
    ax : Axes, optional
        Matplotlib axes to plot on.
    title : str, optional
        Plot title.
    filename : str, optional
        If provided, save figure to this file.
        
    Returns
    -------
    Axes or None
        The axes if ax was provided, otherwise None.
        
    Examples
    --------
    >>> # Using domain and function
    >>> domain = Rectangle(4, 4)
    >>> plot(domain, lambda z: z**2, resolution=200)
    
    >>> # Using pre-computed arrays
    >>> z = domain.mesh(200)
    >>> f = z**2
    >>> plot(z=z, f=f)
    """
    # Validate inputs
    if domain is None and z is None:
        raise ValidationError("Either domain or z must be provided")
    
    if f is None and func is None:
        raise ValidationError("Either f or func must be provided")
    
    # Default colormap
    if cmap is None:
        cmap = Phase(n_phi=6, auto_scale_r=True)
    
    # Get mesh and mask
    if z is None:
        z = domain.mesh(resolution)
        mask = domain.outmask(resolution)
    else:
        mask = None
    
    # Evaluate function
    if f is None:
        f = func(z)
    
    # Ensure f has the same shape as z if it's a scalar
    f = np.asarray(f)
    if f.ndim == 0:
        f = np.full_like(z, f)
    
    # Get RGB values
    rgb = cmap.rgb(f, outmask=mask)
    
    # Calculate extent
    extent = [
        np.real(z).min(),
        np.real(z).max(),
        np.imag(z).min(),
        np.imag(z).max(),
    ]
    
    # Calculate aspect ratio
    aspect = (extent[1] - extent[0]) / (extent[3] - extent[2])
    
    # Plot
    if ax is None:
        plt.imshow(rgb, origin="lower", extent=extent, aspect=aspect)
        plt.xlabel("Re(z)")
        plt.ylabel("Im(z)")
        if title:
            plt.title(title)
        if filename:
            plt.savefig(filename)
        # Return the current axes
        return plt.gca()
    else:
        ax.imshow(rgb, origin="lower", extent=extent, aspect=aspect)
        ax.set_xlabel("Re(z)")
        ax.set_ylabel("Im(z)")
        if title:
            ax.set_title(title)
        return ax


def pair_plot(domain: Optional[Domain] = None,
              func: Optional[Callable] = None,
              z: Optional[np.ndarray] = None,
              f: Optional[np.ndarray] = None,
              resolution: int = 400,
              cmap: Optional[Colormap] = None,
              title: Optional[str] = None,
              figsize: Tuple[float, float] = (10, 5),
              filename: Optional[str] = None) -> Figure:
    """Plot domain and codomain side by side.
    
    Parameters
    ----------
    domain : Domain, optional
        Domain object. If None, z must be provided.
    func : callable, optional
        Complex function. If None, f must be provided.
    z : ndarray, optional
        2D array of complex domain values.
    f : ndarray, optional
        2D array of complex codomain values.
    resolution : int, optional
        Resolution (number of points along longest edge).
    cmap : Colormap, optional
        Colormap to use. Defaults to enhanced phase portrait.
    title : str, optional
        Overall figure title.
    figsize : tuple, optional
        Figure size (width, height).
    filename : str, optional
        If provided, save figure to this file.
        
    Returns
    -------
    Figure
        The matplotlib figure.
    """
    # Default colormap
    if cmap is None:
        cmap = Phase(n_phi=6, auto_scale_r=True)
    
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot domain
    plot(domain=domain, func=lambda x: x, z=z, f=z, resolution=resolution, 
         cmap=cmap, title='Domain z', ax=ax0)
    
    # Plot codomain
    plot(domain=domain, func=func, z=z, f=f, resolution=resolution,
         cmap=cmap, title='Codomain f(z)', ax=ax1)
    
    if title:
        fig.suptitle(title)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename)
        
    return fig


def riemann_chart(func: Callable,
                  domain: Optional[Domain] = None,
                  resolution: int = 100,
                  show_south_hemisphere: bool = True,
                  project_from_north: bool = True,
                  cmap: Optional[Colormap] = None,
                  ax: Optional[Axes] = None,
                  margin: float = 0.05,
                  unit_circle_width: float = 1.0) -> Axes:
    """Plot phase portrait on Riemann hemisphere.
    
    This function visualizes a complex function on either hemisphere
    of the Riemann sphere using stereographic projection.
    
    Parameters
    ----------
    func : callable
        Complex function to visualize.
    domain : Domain, optional
        If provided, its mask will be applied.
    resolution : int, optional
        Resolution for the mesh.
    show_south_hemisphere : bool, optional
        If True, show lower hemisphere; else upper.
    project_from_north : bool, optional
        If True, use north pole projection (standard).
    cmap : Colormap, optional
        Colormap to use.
    ax : Axes, optional
        Matplotlib axes to plot on.
    margin : float, optional
        Margin around unit disk (0 to 0.5).
    unit_circle_width : float, optional
        Width of unit circle highlight.
        
    Returns
    -------
    Axes
        The matplotlib axes used.
    """
    # Validate margin
    if margin < 0:
        raise ValidationError('Margin must be non-negative')
    if margin > 0.5:
        raise ValidationError('Margin cannot exceed 0.5')
    
    # Default colormap
    if cmap is None:
        cmap = Phase(n_phi=6, auto_scale_r=True)
    
    # Create domain for unit disk with margin
    disk_radius = 1 + margin
    dom = Rectangle(2 * disk_radius, 2 * disk_radius)
    
    # Apply mask from provided domain if any
    if domain and hasattr(domain, 'mask_list'):
        dom.mask_list = domain.mask_list
    
    z = dom.mesh(resolution)
    
    # Adjust hemisphere based on projection
    if not project_from_north:
        show_south_hemisphere = not show_south_hemisphere
    
    # Evaluate function
    if show_south_hemisphere:
        F = func(z)
        arg_label = 'z'
        tick_labels = ['-1', '-0.5', '0', '0.5', '1']
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            F = func(1/z)
        arg_label = '1/z'
        tick_labels = ['-1', '-2', '∞', '2', '1']
    
    # Ensure F is array
    F = np.asarray(F)
    if F.ndim == 0:
        F = np.full_like(z, F)
    
    # Get HSV components
    H, S, V = cmap.hsv_tuple(F)
    
    # Reduce saturation outside unit disk
    inside_unit = np.abs(z) < 1
    S = np.where(inside_unit, S, S * 0.7)
    
    # Draw unit circle
    spacing = dom.spacing(resolution)
    unit_circle_tol = spacing * unit_circle_width
    on_unit_circle = np.abs(np.abs(z) - 1) < unit_circle_tol
    V[on_unit_circle] = 0
    
    # Convert to RGB
    HSV = np.dstack((H, S, V))
    RGB = mcolors.hsv_to_rgb(HSV)
    
    # Create axes if needed
    if ax is None:
        fig, ax = plt.subplots()
    
    # Plot
    extent = [-disk_radius, disk_radius, -disk_radius, disk_radius]
    ax.imshow(RGB, origin="lower", aspect=1, extent=extent)
    ax.set_xlabel(f"Re({arg_label})")
    ax.set_ylabel(f"Im({arg_label})")
    
    # Set ticks
    ticks = [-1, -0.5, 0, 0.5, 1]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)
    
    return ax


def riemann_hemispheres(func: Callable,
                       title: Optional[str] = None,
                       resolution: int = 400,
                       margin: float = 0.05,
                       unit_circle_width: float = 1.0,
                       figsize: Tuple[float, float] = (12, 4),
                       filename: Optional[str] = None) -> Figure:
    """Plot both hemispheres of the Riemann sphere.
    
    Parameters
    ----------
    func : callable
        Complex function to visualize.
    title : str, optional
        Overall figure title.
    resolution : int, optional
        Resolution for the mesh.
    margin : float, optional
        Margin around unit disk.
    unit_circle_width : float, optional
        Width of unit circle highlight.
    figsize : tuple, optional
        Figure size (width, height).
    filename : str, optional
        If provided, save figure to this file.
        
    Returns
    -------
    Figure
        The matplotlib figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # South hemisphere
    riemann_chart(func, resolution=resolution, show_south_hemisphere=True, ax=ax1,
                  margin=margin, unit_circle_width=unit_circle_width)
    ax1.set_title("South (lower) hemisphere")
    
    # North hemisphere  
    riemann_chart(func, resolution=resolution, show_south_hemisphere=False, ax=ax2,
                  margin=margin, unit_circle_width=unit_circle_width)
    ax2.set_title("North (upper) hemisphere")
    
    if title:
        fig.suptitle(title, fontsize=14)
    
    fig.tight_layout()
    
    if filename:
        plt.savefig(filename)
        
    return fig