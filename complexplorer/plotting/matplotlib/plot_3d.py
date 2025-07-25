"""3D plotting functions using matplotlib.

This module provides functions for visualizing complex functions
as 3D landscapes and on the Riemann sphere.
"""

from typing import Optional, Callable, Union, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from ...core.domain import Domain
from ...core.colormap import Colormap, Phase
from ...core.functions import stereographic_projection
from ...core.scaling import ModulusScaling
from ...utils.validation import ValidationError
from ...utils.mesh_distortion import get_default_scaling_params


class Matplotlib3DPlotter:  # (Base3DPlotter):  # TODO: Add base class
    """3D plotter implementation using matplotlib."""
    
    def plot_landscape(self,
                      domain: Domain,
                      func: Callable[[np.ndarray], np.ndarray],
                      colormap: Colormap,
                      resolution: int,
                      ax: Optional[Axes3D] = None,
                      zaxis_log: bool = False,
                      z_max: Optional[float] = None,
                      antialiased: bool = False,
                      modulus_mode: str = 'none',
                      modulus_params: Optional[dict] = None,
                      config = None) -> Axes3D:  # config: Optional[PlotConfig] = None
        """Plot 3D landscape of complex function.
        
        Parameters
        ----------
        domain : Domain
            The domain to plot over.
        func : callable
            Complex function to visualize.
        colormap : Colormap
            Colormap for surface coloring.
        resolution : int
            Number of points along longest edge.
        ax : Axes3D, optional
            3D matplotlib axes to plot on.
        zaxis_log : bool, optional
            Use logarithmic scale for z-axis.
        z_max : float, optional
            Maximum z-value to display.
        antialiased : bool, optional
            Enable antialiasing.
        modulus_mode : str, optional
            How to scale the height based on modulus.
        modulus_params : dict, optional
            Parameters for modulus scaling method.
        config : PlotConfig, optional
            Additional plot configuration.
            
        Returns
        -------
        Axes3D
            The 3D axes used for plotting.
        """
        # Get mesh and evaluate function
        z = domain.mesh(resolution)
        mask = domain.outmask(resolution)
        f_z = func(z)
        
        # Ensure f_z is array
        f_z = np.asarray(f_z)
        if f_z.ndim == 0:
            f_z = np.full_like(z, f_z)
        
        # Apply mask
        if mask is not None:
            f_z[mask] = np.nan
        
        # Get RGB colors
        rgb = colormap.rgb(f_z, outmask=mask)
        
        # Calculate z-coordinates (moduli)
        z_coord = np.abs(f_z)
        
        # Apply modulus scaling if requested
        if modulus_mode != 'none':
            if modulus_params is None:
                modulus_params = get_default_scaling_params(modulus_mode)
            
            if modulus_mode == 'custom':
                if 'scaling_func' not in modulus_params:
                    raise ValidationError("Custom mode requires 'scaling_func' in modulus_params")
                z_coord = modulus_params['scaling_func'](z_coord)
            else:
                scaling_method = getattr(ModulusScaling, modulus_mode, None)
                if scaling_method is None:
                    raise ValidationError(f"Unknown scaling mode: {modulus_mode}")
                z_coord = scaling_method(z_coord, **modulus_params)
        
        # Apply z_max clipping after scaling
        if z_max is not None:
            if z_max <= 0:
                raise ValidationError('z_max must be positive')
            z_coord = np.clip(z_coord, 0, z_max)
        
        if zaxis_log:
            with np.errstate(divide='ignore', invalid='ignore'):
                z_coord = np.log(z_coord)
        
        # Create axes if needed
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        
        # Plot surface
        ax.plot_surface(np.real(z), np.imag(z), z_coord, 
                       facecolors=rgb, 
                       rcount=resolution, 
                       ccount=resolution,
                       linewidth=0, 
                       antialiased=antialiased)
        
        ax.set_xlabel("Re(z)")
        ax.set_ylabel("Im(z)")
        
        return ax


def plot_landscape(domain: Optional[Domain] = None,
                  z: Optional[np.ndarray] = None,
                  f: Optional[np.ndarray] = None,
                  func: Optional[Callable] = None,
                  resolution: int = 100,
                  cmap: Optional[Colormap] = None,
                  ax: Optional[Axes3D] = None,
                  antialiased: bool = False,
                  zaxis_log: bool = False,
                  z_max: Optional[float] = None,
                  modulus_mode: str = 'none',
                  modulus_params: Optional[dict] = None) -> Optional[Axes3D]:
    """Plot complex function as 3D landscape.
    
    The height represents |f(z)| and color represents the phase or
    other colormap encoding.
    
    Parameters
    ----------
    domain : Domain, optional
        Domain object. If None, z must be provided.
    z : ndarray, optional
        2D array of complex domain values.
    f : ndarray, optional
        2D array of complex codomain values.
    func : callable, optional
        Complex function. If None, f must be provided.
    resolution : int, optional
        Resolution (number of points along longest edge).
    cmap : Colormap, optional
        Colormap to use. Defaults to enhanced phase portrait.
    ax : Axes3D, optional
        3D matplotlib axes to plot on.
    antialiased : bool, optional
        Enable antialiasing for smoother surface.
    zaxis_log : bool, optional
        Use logarithmic scale for z-axis.
    z_max : float, optional
        Maximum z-value to display.
    modulus_mode : str, optional
        How to scale the height based on modulus:
        - 'none': Direct modulus as height (default)
        - 'constant': Fixed height (shows phase only)
        - 'linear': Linear scaling with parameter
        - 'arctan': Smooth bounded scaling
        - 'logarithmic': Log scaling
        - 'linear_clamp': Linear with clamping
        - 'power': Power scaling
        - 'sigmoid': S-curve scaling
        - 'adaptive': Percentile-based
        - 'hybrid': Linear near zero, log for large
        - 'custom': User-defined function
    modulus_params : dict, optional
        Parameters for modulus scaling method.
        See ModulusScaling class for parameter details.
        
    Returns
    -------
    Axes3D or None
        The 3D axes if ax was provided, otherwise None.
        
    Examples
    --------
    >>> domain = Rectangle(4, 4)
    >>> plot_landscape(domain, func=lambda z: z**2 - 1)
    
    >>> # With logarithmic scale for poles/zeros
    >>> plot_landscape(domain, func=lambda z: 1/z, zaxis_log=True)
    """
    # Validate inputs
    if domain is None and z is None:
        raise ValidationError("Either domain or z must be provided")
    
    if f is None and func is None:
        raise ValidationError("Either f or func must be provided")
    
    # Default colormap
    if cmap is None:
        cmap = Phase(n_phi=6)
    
    # Get mesh and mask
    if z is None:
        z = domain.mesh(resolution)
        mask = domain.outmask(resolution)
    else:
        mask = None
    
    # Evaluate function
    if f is None:
        f = func(z)
    
    # Ensure f is array
    f = np.asarray(f)
    if f.ndim == 0:
        f = np.full_like(z, f)
    
    # Apply mask
    if mask is not None:
        f[mask] = np.nan
    
    # Validate z_max
    if z_max is not None and z_max <= 0:
        raise ValidationError('z_max must be positive or None')
    
    # Get RGB colors
    rgb = cmap.rgb(f, outmask=mask)
    
    # Calculate z-coordinates (moduli)
    z_coord = np.abs(f)
    
    # Apply modulus scaling if requested
    if modulus_mode != 'none':
        if modulus_params is None:
            modulus_params = get_default_scaling_params(modulus_mode)
        
        if modulus_mode == 'custom':
            if 'scaling_func' not in modulus_params:
                raise ValidationError("Custom mode requires 'scaling_func' in modulus_params")
            z_coord = modulus_params['scaling_func'](z_coord)
        else:
            scaling_method = getattr(ModulusScaling, modulus_mode, None)
            if scaling_method is None:
                raise ValidationError(f"Unknown scaling mode: {modulus_mode}")
            z_coord = scaling_method(z_coord, **modulus_params)
    
    # Apply z_max clipping after scaling
    if z_max is not None:
        z_coord = np.clip(z_coord, 0, z_max)
    
    if zaxis_log:
        with np.errstate(divide='ignore', invalid='ignore'):
            z_coord = np.log(z_coord)
    
    # Plot
    return_ax = ax is not None
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    
    ax.plot_surface(np.real(z), np.imag(z), z_coord, 
                   rcount=resolution, ccount=resolution, 
                   facecolors=rgb,
                   linewidth=0, 
                   antialiased=antialiased)
    
    ax.set_xlabel("Re(z)")
    ax.set_ylabel("Im(z)")
    
    if return_ax:
        return ax
    else:
        # Return the current axes even if we created it
        return plt.gca()


def pair_plot_landscape(domain: Optional[Domain] = None,
                       func: Optional[Callable] = None,
                       z: Optional[np.ndarray] = None,
                       f: Optional[np.ndarray] = None,
                       resolution: int = 100,
                       cmap: Optional[Colormap] = None,
                       title: Optional[str] = None,
                       figsize: Tuple[float, float] = (10, 5),
                       zaxis_log: bool = False,
                       z_max: Optional[float] = None,
                       modulus_mode: str = 'none',
                       modulus_params: Optional[dict] = None,
                       filename: Optional[str] = None) -> Figure:
    """Plot domain and codomain landscapes side by side.
    
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
        Colormap to use.
    title : str, optional
        Overall figure title.
    figsize : tuple, optional
        Figure size (width, height).
    zaxis_log : bool, optional
        Use logarithmic scale for z-axis.
    z_max : float, optional
        Maximum z-value to display.
    modulus_mode : str, optional
        How to scale the height based on modulus.
        See plot_landscape for available modes.
    modulus_params : dict, optional
        Parameters for modulus scaling method.
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
    
    fig = plt.figure(figsize=figsize)
    ax0 = fig.add_subplot(121, projection='3d')
    ax1 = fig.add_subplot(122, projection='3d')
    
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    
    # Plot domain (identity)
    plot_landscape(domain=domain, func=lambda x: x, z=z, f=z, resolution=resolution, 
                  cmap=cmap, ax=ax0, zaxis_log=zaxis_log, z_max=z_max,
                  modulus_mode=modulus_mode, modulus_params=modulus_params)
    
    # Plot codomain
    plot_landscape(domain=domain, func=func, z=z, f=f, resolution=resolution,
                  cmap=cmap, ax=ax1, zaxis_log=zaxis_log, z_max=z_max,
                  modulus_mode=modulus_mode, modulus_params=modulus_params)
    
    if title:
        fig.suptitle(title)
    
    if filename:
        plt.savefig(filename)
    
    return fig


def riemann(func: Callable,
            resolution: int = 80,
            cmap: Optional[Colormap] = None,
            project_from_north: bool = False,
            ax: Optional[Axes3D] = None,
            title: Optional[str] = None,
            filename: Optional[str] = None) -> Optional[Axes3D]:
    """Plot complex function on the Riemann sphere.
    
    This function visualizes a complex function on the Riemann sphere
    using stereographic projection. The sphere is colored according to
    the function values.
    
    Parameters
    ----------
    func : callable
        Complex function to visualize.
    resolution : int, optional
        Number of mesh points in each direction.
    cmap : Colormap, optional
        Colormap to use.
    project_from_north : bool, optional
        If True, use north pole projection.
    ax : Axes3D, optional
        3D matplotlib axes to plot on.
    title : str, optional
        Plot title.
    filename : str, optional
        If provided, save figure to this file.
        
    Returns
    -------
    Axes3D or None
        The 3D axes if ax was provided, otherwise None.
        
    Examples
    --------
    >>> # Rational function with poles and zeros
    >>> riemann(lambda z: (z**2 - 1) / (z**2 + 1))
    
    >>> # Essential singularity at origin
    >>> riemann(lambda z: np.exp(1/z))
    """
    # Default colormap
    if cmap is None:
        cmap = Phase(n_phi=6, v_base=0.6)
    
    # Create sphere mesh in spherical coordinates
    tol = 1e-8
    psi_axis = np.linspace(tol, np.pi - tol, resolution)
    theta_axis = np.linspace(0, 2 * np.pi, resolution)
    theta_grid, psi_grid = np.meshgrid(theta_axis, psi_axis)
    
    # Convert to complex plane via stereographic projection
    # Using formula: z = r * e^(i*theta) where r = sin(psi) / (1 - cos(psi))
    # Suppress divide by zero warning as we handle it with tol
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.sin(psi_grid) / (1 - np.cos(psi_grid))
        z = r * np.exp(1.0j * theta_grid)
    
    # Evaluate function and get colors
    f_z = func(z)
    rgb = cmap.rgb(f_z)
    
    # Ensure RGB values are in valid range [0, 1]
    rgb = np.clip(rgb, 0.0, 1.0)
    
    # Convert back to sphere coordinates
    X, Y, Z = stereographic_projection(z, project_from_north=project_from_north).T
    
    # Create figure if needed
    return_ax = ax is not None
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    
    # Plot sphere
    ax.plot_surface(X, Y, Z, facecolors=rgb, 
                   rcount=resolution, ccount=resolution,
                   linewidth=0, antialiased=False)
    
    # Configure axes
    ax.set_axis_off()
    ax.set_proj_type('ortho')
    ax.set_aspect('equal')
    ax.view_init(20, 45)
    
    if title:
        ax.set_title(title, fontsize=14)
    
    if not return_ax and filename:
        plt.savefig(filename)
        plt.tight_layout(rect=[-0.2, -0.2, 1.5, 1.5])
    
    if return_ax:
        return ax
    else:
        # Return the current axes even if we created it
        return plt.gca()