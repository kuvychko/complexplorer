"""Abstract base classes for plotting functionality.

This module provides the foundation for all plotting implementations,
ensuring consistent interfaces and shared functionality.
"""

from abc import ABC, abstractmethod
from typing import Optional, Callable, Dict, Any, Tuple, Union
import numpy as np
from ..utils.validation import (
    validate_domain_or_mesh,
    validate_function_or_values,
    validate_function,
    validate_colormap,
    ValidationError
)


class BasePlotter(ABC):
    """Abstract base class for all plotters.
    
    This class provides common functionality for 2D and 3D plotting,
    including validation, data preparation, and standard interfaces.
    
    Attributes
    ----------
    domain : Domain or None
        Domain object for generating mesh points.
    func : callable or None
        Complex function to visualize.
    cmap : Colormap
        Color mapping object.
    _data : dict or None
        Cached data from preparation step.
    """
    
    def __init__(self, 
                 domain: Optional[Any] = None,
                 func: Optional[Callable] = None,
                 cmap: Optional[Any] = None,
                 z: Optional[np.ndarray] = None,
                 f: Optional[np.ndarray] = None):
        """Initialize base plotter.
        
        Parameters
        ----------
        domain : Domain, optional
            Domain object for generating mesh.
        func : callable, optional
            Complex function to visualize.
        cmap : Colormap, optional
            Color mapping object.
        z : np.ndarray, optional
            Pre-computed mesh points.
        f : np.ndarray, optional
            Pre-computed function values.
        """
        # Validate inputs
        validate_domain_or_mesh(domain, z)
        validate_function_or_values(func, f)
        
        self.domain = domain
        self.z = z
        self.func = validate_function(func) if func is not None else None
        self.f = f
        self.cmap = validate_colormap(cmap)
        self._data = None
    
    @abstractmethod
    def plot(self, **kwargs) -> Any:
        """Main plotting method to be implemented by subclasses.
        
        Parameters
        ----------
        **kwargs
            Additional plotting parameters.
            
        Returns
        -------
        Any
            Plot object or None.
        """
        pass
    
    def _prepare_data(self) -> Dict[str, np.ndarray]:
        """Prepare data for plotting.
        
        This method handles mesh generation, function evaluation,
        and color mapping.
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'z': mesh points
            - 'f': function values
            - 'colors': RGB color values
            - 'x', 'y': real and imaginary parts of z
            - 'u', 'v': real and imaginary parts of f
        """
        if self._data is not None:
            return self._data
        
        # Generate mesh if needed
        if self.z is None:
            if self.domain is None:
                raise ValidationError("No mesh points available")
            self.z = self.domain.mesh()
        
        # Evaluate function if needed
        if self.f is None:
            if self.func is None:
                raise ValidationError("No function values available")
            self.f = self.func(self.z)
        
        # Ensure arrays
        self.z = np.asarray(self.z)
        self.f = np.asarray(self.f)
        
        # Extract components
        x = np.real(self.z)
        y = np.imag(self.z)
        u = np.real(self.f)
        v = np.imag(self.f)
        
        # Generate colors
        # Reshape for colormap if needed
        f_for_color = self.f.ravel() if self.f.ndim > 1 else self.f
        colors = self.cmap.rgb(f_for_color.reshape(-1, 1)).squeeze()
        
        # Cache data
        self._data = {
            'z': self.z,
            'f': self.f,
            'x': x,
            'y': y,
            'u': u,
            'v': v,
            'colors': colors
        }
        
        return self._data
    
    def _get_plot_limits(self, 
                        data: Dict[str, np.ndarray],
                        padding: float = 0.05) -> Dict[str, Tuple[float, float]]:
        """Calculate plot limits from data.
        
        Parameters
        ----------
        data : dict
            Data dictionary from _prepare_data.
        padding : float, optional
            Padding factor for limits.
            
        Returns
        -------
        dict
            Dictionary with 'x', 'y', 'u', 'v' limits.
        """
        limits = {}
        
        for key in ['x', 'y', 'u', 'v']:
            if key in data:
                values = data[key]
                finite_vals = values[np.isfinite(values)]
                
                if len(finite_vals) > 0:
                    vmin, vmax = np.min(finite_vals), np.max(finite_vals)
                    vrange = vmax - vmin
                    pad = vrange * padding
                    limits[key] = (vmin - pad, vmax + pad)
                else:
                    limits[key] = (-1, 1)  # Default if no finite values
        
        return limits
    
    def _handle_infinities(self, 
                          values: np.ndarray,
                          replacement: float = 0.0) -> np.ndarray:
        """Handle infinite and NaN values.
        
        Parameters
        ----------
        values : np.ndarray
            Array possibly containing infinities.
        replacement : float, optional
            Value to use for infinities.
            
        Returns
        -------
        np.ndarray
            Array with infinities replaced.
        """
        result = values.copy()
        mask = ~np.isfinite(values)
        if np.any(mask):
            result[mask] = replacement
        return result


class Base2DPlotter(BasePlotter):
    """Base class for 2D plotting.
    
    Provides additional functionality specific to 2D visualizations.
    """
    
    def _prepare_axes(self, ax=None, figsize=(8, 8)):
        """Prepare matplotlib axes for 2D plotting.
        
        Parameters
        ----------
        ax : matplotlib axes, optional
            Existing axes to use.
        figsize : tuple, optional
            Figure size if creating new figure.
            
        Returns
        -------
        fig, ax
            Figure and axes objects.
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        return fig, ax
    
    def _apply_2d_styling(self, ax, title=None, xlabel='Re', ylabel='Im'):
        """Apply common 2D plot styling.
        
        Parameters
        ----------
        ax : matplotlib axes
            Axes to style.
        title : str, optional
            Plot title.
        xlabel : str, optional
            X-axis label.
        ylabel : str, optional
            Y-axis label.
        """
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)


class Base3DPlotter(BasePlotter):
    """Base class for 3D plotting.
    
    Provides additional functionality specific to 3D visualizations.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize 3D plotter."""
        super().__init__(*args, **kwargs)
        self._mesh_data = None
    
    def get_mesh_data(self) -> Any:
        """Get mesh data for export.
        
        This method should return mesh data suitable for
        STL export or other 3D file formats.
        
        Returns
        -------
        Any
            Mesh data (format depends on implementation).
        """
        if self._mesh_data is None:
            raise ValidationError("No mesh data available. Call plot() first.")
        return self._mesh_data
    
    def _prepare_3d_axes(self, ax=None, figsize=(10, 8), projection='3d'):
        """Prepare matplotlib axes for 3D plotting.
        
        Parameters
        ----------
        ax : matplotlib axes, optional
            Existing axes to use.
        figsize : tuple, optional
            Figure size if creating new figure.
        projection : str, optional
            Projection type.
            
        Returns
        -------
        fig, ax
            Figure and axes objects.
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection=projection)
        else:
            fig = ax.figure
        
        return fig, ax
    
    def _apply_3d_styling(self, ax, title=None, 
                         xlabel='Re', ylabel='Im', zlabel='|f(z)|',
                         elev=30, azim=45):
        """Apply common 3D plot styling.
        
        Parameters
        ----------
        ax : matplotlib 3D axes
            Axes to style.
        title : str, optional
            Plot title.
        xlabel : str, optional
            X-axis label.
        ylabel : str, optional
            Y-axis label.
        zlabel : str, optional
            Z-axis label.
        elev : float, optional
            Elevation angle.
        azim : float, optional
            Azimuth angle.
        """
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if zlabel:
            ax.set_zlabel(zlabel)
        
        ax.view_init(elev=elev, azim=azim)


class PlotConfig:
    """Configuration container for plot parameters.
    
    This class helps manage the many parameters involved in plotting,
    providing defaults and validation.
    """
    
    def __init__(self, **kwargs):
        """Initialize configuration with parameters.
        
        Parameters
        ----------
        **kwargs
            Any configuration parameters.
        """
        # Common parameters
        self.figsize = kwargs.get('figsize', (8, 8))
        self.title = kwargs.get('title', None)
        self.colorbar = kwargs.get('colorbar', True)
        self.grid = kwargs.get('grid', True)
        self.dpi = kwargs.get('dpi', 100)
        
        # File output
        self.filename = kwargs.get('filename', None)
        self.format = kwargs.get('format', 'png')
        
        # 3D specific
        self.projection = kwargs.get('projection', '3d')
        self.elev = kwargs.get('elev', 30)
        self.azim = kwargs.get('azim', 45)
        
        # Interactive
        self.interactive = kwargs.get('interactive', True)
        self.show = kwargs.get('show', True)
        
        # Store any extra parameters
        self._extra = {k: v for k, v in kwargs.items() 
                      if not hasattr(self, k)}
    
    def get(self, key, default=None):
        """Get configuration value.
        
        Parameters
        ----------
        key : str
            Configuration key.
        default : any, optional
            Default value if key not found.
            
        Returns
        -------
        any
            Configuration value.
        """
        if hasattr(self, key):
            return getattr(self, key)
        return self._extra.get(key, default)
    
    def update(self, **kwargs):
        """Update configuration values.
        
        Parameters
        ----------
        **kwargs
            Configuration parameters to update.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self._extra[key] = value