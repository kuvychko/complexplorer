"""Base export functionality for complexplorer.

This module provides abstract base classes and common functionality
for exporting visualizations to various file formats.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, Union
from pathlib import Path
import numpy as np
from ..utils.validation import (
    validate_file_extension,
    ValidationError
)


class BaseExporter(ABC):
    """Abstract base class for export functionality.
    
    All exporters should inherit from this class and implement
    the export method.
    """
    
    def __init__(self, data: Optional[Any] = None):
        """Initialize exporter.
        
        Parameters
        ----------
        data : any, optional
            Data to export (can be set later).
        """
        self.data = data
        self._metadata = {}
    
    @abstractmethod
    def export(self, filename: str, **kwargs) -> Path:
        """Export data to file.
        
        Parameters
        ----------
        filename : str
            Output filename.
        **kwargs
            Additional export parameters.
            
        Returns
        -------
        Path
            Path to exported file.
        """
        pass
    
    def validate_filename(self, 
                         filename: str,
                         valid_extensions: Union[str, list]) -> Path:
        """Validate and process filename.
        
        Parameters
        ----------
        filename : str
            Filename to validate.
        valid_extensions : str or list
            Valid file extensions.
            
        Returns
        -------
        Path
            Validated file path.
        """
        # Validate extension
        ext = validate_file_extension(filename, valid_extensions)
        
        # Convert to Path
        filepath = Path(filename)
        
        # Create parent directory if needed
        if filepath.parent and not filepath.parent.exists():
            filepath.parent.mkdir(parents=True, exist_ok=True)
        
        return filepath
    
    def set_metadata(self, **metadata):
        """Set metadata for export.
        
        Parameters
        ----------
        **metadata
            Metadata key-value pairs.
        """
        self._metadata.update(metadata)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get export metadata.
        
        Returns
        -------
        dict
            Metadata dictionary.
        """
        return self._metadata.copy()


class ImageExporter(BaseExporter):
    """Base class for image export (PNG, JPG, etc.).
    
    Provides common functionality for raster image export.
    """
    
    VALID_EXTENSIONS = ['png', 'jpg', 'jpeg', 'tiff', 'bmp']
    
    def __init__(self, figure=None):
        """Initialize image exporter.
        
        Parameters
        ----------
        figure : matplotlib figure, optional
            Figure to export.
        """
        super().__init__(figure)
        self.figure = figure
    
    def export(self, filename: str, 
              dpi: int = 100,
              bbox_inches: str = 'tight',
              transparent: bool = False,
              **kwargs) -> Path:
        """Export figure to image file.
        
        Parameters
        ----------
        filename : str
            Output filename.
        dpi : int, optional
            Dots per inch for output.
        bbox_inches : str, optional
            Bounding box setting.
        transparent : bool, optional
            Whether to use transparent background.
        **kwargs
            Additional arguments passed to savefig.
            
        Returns
        -------
        Path
            Path to exported file.
        """
        if self.figure is None:
            raise ValidationError("No figure to export")
        
        filepath = self.validate_filename(filename, self.VALID_EXTENSIONS)
        
        # Save figure
        self.figure.savefig(
            filepath,
            dpi=dpi,
            bbox_inches=bbox_inches,
            transparent=transparent,
            **kwargs
        )
        
        return filepath


class VectorExporter(BaseExporter):
    """Base class for vector export (SVG, PDF, EPS).
    
    Provides common functionality for vector graphics export.
    """
    
    VALID_EXTENSIONS = ['svg', 'pdf', 'eps', 'ps']
    
    def __init__(self, figure=None):
        """Initialize vector exporter.
        
        Parameters
        ----------
        figure : matplotlib figure, optional
            Figure to export.
        """
        super().__init__(figure)
        self.figure = figure
    
    def export(self, filename: str,
              bbox_inches: str = 'tight',
              **kwargs) -> Path:
        """Export figure to vector file.
        
        Parameters
        ----------
        filename : str
            Output filename.
        bbox_inches : str, optional
            Bounding box setting.
        **kwargs
            Additional arguments passed to savefig.
            
        Returns
        -------
        Path
            Path to exported file.
        """
        if self.figure is None:
            raise ValidationError("No figure to export")
        
        filepath = self.validate_filename(filename, self.VALID_EXTENSIONS)
        
        # Save figure
        self.figure.savefig(
            filepath,
            bbox_inches=bbox_inches,
            **kwargs
        )
        
        return filepath


class MeshExporter(BaseExporter):
    """Base class for 3D mesh export.
    
    Provides common functionality for exporting 3D meshes
    to various formats (STL, OBJ, PLY, etc.).
    """
    
    def __init__(self, mesh_data=None):
        """Initialize mesh exporter.
        
        Parameters
        ----------
        mesh_data : any, optional
            Mesh data to export.
        """
        super().__init__(mesh_data)
        self.mesh_data = mesh_data
    
    def validate_mesh(self) -> None:
        """Validate mesh data.
        
        Raises
        ------
        ValidationError
            If mesh data is invalid.
        """
        if self.mesh_data is None:
            raise ValidationError("No mesh data to export")
    
    def get_mesh_statistics(self) -> Dict[str, Any]:
        """Get mesh statistics.
        
        Returns
        -------
        dict
            Dictionary with mesh statistics.
        """
        self.validate_mesh()
        
        # This is a placeholder - actual implementation
        # depends on mesh format
        return {
            'vertices': 0,
            'faces': 0,
            'edges': 0
        }


class InteractiveExporter(BaseExporter):
    """Base class for interactive export (HTML).
    
    Provides functionality for exporting interactive visualizations.
    """
    
    VALID_EXTENSIONS = ['html']
    
    def __init__(self, plotter=None):
        """Initialize interactive exporter.
        
        Parameters
        ----------
        plotter : any, optional
            Plotter object (e.g., PyVista plotter).
        """
        super().__init__(plotter)
        self.plotter = plotter
    
    def validate_dependencies(self) -> None:
        """Validate that required dependencies are available.
        
        Raises
        ------
        ValidationError
            If required dependencies are missing.
        """
        # Override in subclasses to check specific dependencies
        pass


class ExportConfig:
    """Configuration for export operations.
    
    Centralizes export parameters and provides defaults.
    """
    
    def __init__(self, **kwargs):
        """Initialize export configuration.
        
        Parameters
        ----------
        **kwargs
            Export configuration parameters.
        """
        # Common parameters
        self.overwrite = kwargs.get('overwrite', True)
        self.create_dirs = kwargs.get('create_dirs', True)
        self.verbose = kwargs.get('verbose', False)
        
        # Image parameters
        self.dpi = kwargs.get('dpi', 100)
        self.transparent = kwargs.get('transparent', False)
        self.bbox_inches = kwargs.get('bbox_inches', 'tight')
        
        # 3D parameters
        self.binary = kwargs.get('binary', True)
        self.precision = kwargs.get('precision', 6)
        
        # Metadata
        self.include_metadata = kwargs.get('include_metadata', True)
        self.author = kwargs.get('author', 'complexplorer')
        self.description = kwargs.get('description', '')
        
        # Store extra parameters
        self._extra = {k: v for k, v in kwargs.items()
                      if not hasattr(self, k)}
    
    def get(self, key, default=None):
        """Get configuration value."""
        if hasattr(self, key):
            return getattr(self, key)
        return self._extra.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for key in dir(self):
            if not key.startswith('_') and not callable(getattr(self, key)):
                result[key] = getattr(self, key)
        result.update(self._extra)
        return result