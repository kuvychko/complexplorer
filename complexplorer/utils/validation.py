"""Centralized validation utilities for complexplorer.

This module provides common validation functions used throughout the library,
reducing code duplication and standardizing error messages.
"""

from typing import Optional, Callable, Union, Any
import numpy as np
import warnings


class ValidationError(ValueError):
    """Custom exception for validation errors in complexplorer."""
    pass


def validate_domain_or_mesh(domain: Optional[Any], 
                          mesh: Optional[np.ndarray],
                          param_names: tuple = ('domain', 'z')) -> None:
    """Validate that either domain or mesh is provided.
    
    Parameters
    ----------
    domain : Domain or None
        Domain object for generating mesh.
    mesh : np.ndarray or None
        Pre-computed mesh points.
    param_names : tuple, optional
        Names of parameters for error message.
        
    Raises
    ------
    ValidationError
        If both domain and mesh are None.
    """
    if domain is None and mesh is None:
        raise ValidationError(
            f"Both {param_names[0]} and {param_names[1]} parameters cannot be None. "
            f"Provide either a {param_names[0]} object or a {param_names[1]} array."
        )


def validate_function_or_values(func: Optional[Callable],
                              values: Optional[np.ndarray],
                              param_names: tuple = ('func', 'f')) -> None:
    """Validate that either function or values are provided.
    
    Parameters
    ----------
    func : callable or None
        Complex function to evaluate.
    values : np.ndarray or None
        Pre-computed function values.
    param_names : tuple, optional
        Names of parameters for error message.
        
    Raises
    ------
    ValidationError
        If both func and values are None.
    """
    if func is None and values is None:
        raise ValidationError(
            f"Both {param_names[0]} and {param_names[1]} parameters cannot be None. "
            f"Provide either a {param_names[0]} callable or a {param_names[1]} array."
        )


def validate_function(func: Any) -> Callable:
    """Validate and wrap function for safe evaluation.
    
    Parameters
    ----------
    func : any
        Object that should be callable.
        
    Returns
    -------
    callable
        Validated function wrapped for safe evaluation.
        
    Raises
    ------
    ValidationError
        If func is not callable.
    """
    if not callable(func):
        raise ValidationError(
            f"Function parameter must be callable, got {type(func).__name__}"
        )
    
    def safe_func(z):
        """Wrapped function with error handling."""
        try:
            result = func(z)
            # Ensure result is array-like
            return np.asarray(result)
        except Exception as e:
            raise ValidationError(
                f"Function evaluation failed: {e}\n"
                f"Input shape: {np.shape(z)}, dtype: {np.asarray(z).dtype}"
            ) from e
    
    return safe_func


def validate_colormap(cmap: Optional[Any], default_class: Any = None) -> Any:
    """Validate colormap with sensible defaults.
    
    Parameters
    ----------
    cmap : Cmap or None
        Color map object.
    default_class : class, optional
        Default colormap class to use if cmap is None.
        If not provided, will try to import Phase from legacy.
        
    Returns
    -------
    Cmap
        Valid colormap object.
    """
    if cmap is None:
        if default_class is None:
            # Import default here to avoid circular imports
            try:
                from complexplorer.legacy.cmap import Phase
                default_class = Phase
            except ImportError:
                raise ValidationError(
                    "No colormap provided and default Phase colormap not available"
                )
        
        # Return default colormap with standard parameters
        return default_class(n_phi=6, auto_scale_r=True)
    
    # Validate that cmap has required methods
    required_methods = ['hsv', 'rgb']
    for method in required_methods:
        if not hasattr(cmap, method):
            raise ValidationError(
                f"Colormap must have '{method}' method, got {type(cmap).__name__}"
            )
    
    return cmap


def validate_stl_parameters(size_mm: float, 
                          wall_thickness: float,
                          min_size: float = 1.0,
                          max_size: float = 500.0) -> None:
    """Validate STL export parameters.
    
    Parameters
    ----------
    size_mm : float
        Size of the ornament in millimeters.
    wall_thickness : float
        Wall thickness for hollow ornaments.
    min_size : float, optional
        Minimum allowed size.
    max_size : float, optional
        Maximum allowed size.
        
    Raises
    ------
    ValidationError
        If parameters are invalid.
    """
    if not isinstance(size_mm, (int, float)):
        raise ValidationError(f"Size must be numeric, got {type(size_mm).__name__}")
    
    if size_mm <= 0:
        raise ValidationError(f"Size must be positive, got {size_mm}")
    
    if size_mm < min_size or size_mm > max_size:
        raise ValidationError(
            f"Size must be between {min_size} and {max_size} mm, got {size_mm}"
        )
    
    if not isinstance(wall_thickness, (int, float)):
        raise ValidationError(
            f"Wall thickness must be numeric, got {type(wall_thickness).__name__}"
        )
    
    if wall_thickness <= 0:
        raise ValidationError(f"Wall thickness must be positive, got {wall_thickness}")
    
    if wall_thickness >= size_mm / 2:
        raise ValidationError(
            f"Wall thickness ({wall_thickness}) must be less than half the size ({size_mm/2})"
        )


def validate_resolution(resolution: Any, 
                       param_name: str = 'resolution',
                       min_val: int = 10,
                       max_val: int = 1000) -> int:
    """Validate resolution parameter.
    
    Parameters
    ----------
    resolution : any
        Resolution value to validate.
    param_name : str, optional
        Parameter name for error messages.
    min_val : int, optional
        Minimum allowed resolution.
    max_val : int, optional
        Maximum allowed resolution.
        
    Returns
    -------
    int
        Validated resolution value.
        
    Raises
    ------
    ValidationError
        If resolution is invalid.
    """
    try:
        res_int = int(resolution)
    except (TypeError, ValueError):
        raise ValidationError(
            f"{param_name} must be an integer, got {type(resolution).__name__}"
        )
    
    if res_int < min_val or res_int > max_val:
        raise ValidationError(
            f"{param_name} must be between {min_val} and {max_val}, got {res_int}"
        )
    
    return res_int


def validate_array_shape(array: np.ndarray,
                        expected_shape: Optional[tuple] = None,
                        expected_ndim: Optional[int] = None,
                        param_name: str = 'array') -> None:
    """Validate array shape and dimensions.
    
    Parameters
    ----------
    array : np.ndarray
        Array to validate.
    expected_shape : tuple, optional
        Expected shape.
    expected_ndim : int, optional
        Expected number of dimensions.
    param_name : str, optional
        Parameter name for error messages.
        
    Raises
    ------
    ValidationError
        If array shape is invalid.
    """
    if not isinstance(array, np.ndarray):
        raise ValidationError(
            f"{param_name} must be a numpy array, got {type(array).__name__}"
        )
    
    if expected_ndim is not None and array.ndim != expected_ndim:
        raise ValidationError(
            f"{param_name} must have {expected_ndim} dimensions, got {array.ndim}"
        )
    
    if expected_shape is not None and array.shape != expected_shape:
        raise ValidationError(
            f"{param_name} must have shape {expected_shape}, got {array.shape}"
        )


def validate_file_extension(filename: str, 
                          valid_extensions: Union[str, list],
                          param_name: str = 'filename') -> str:
    """Validate file extension.
    
    Parameters
    ----------
    filename : str
        Filename to validate.
    valid_extensions : str or list
        Valid extension(s).
    param_name : str, optional
        Parameter name for error messages.
        
    Returns
    -------
    str
        The file extension (lowercase).
        
    Raises
    ------
    ValidationError
        If file extension is invalid.
    """
    if not isinstance(filename, str):
        raise ValidationError(
            f"{param_name} must be a string, got {type(filename).__name__}"
        )
    
    if not filename:
        raise ValidationError(f"{param_name} cannot be empty")
    
    # Extract extension
    parts = filename.rsplit('.', 1)
    if len(parts) != 2:
        raise ValidationError(f"{param_name} must have a file extension")
    
    ext = parts[1].lower()
    
    # Normalize valid_extensions to list
    if isinstance(valid_extensions, str):
        valid_extensions = [valid_extensions]
    valid_extensions = [e.lower().lstrip('.') for e in valid_extensions]
    
    if ext not in valid_extensions:
        raise ValidationError(
            f"Invalid file extension '.{ext}'. "
            f"Valid extensions: {', '.join(f'.{e}' for e in valid_extensions)}"
        )
    
    return ext


def warn_deprecated(old_name: str, 
                   new_name: str,
                   version: str = "next version") -> None:
    """Issue a deprecation warning.
    
    Parameters
    ----------
    old_name : str
        Name of deprecated feature.
    new_name : str
        Name of replacement feature.
    version : str, optional
        Version when removal will occur.
    """
    warnings.warn(
        f"{old_name} is deprecated and will be removed in {version}. "
        f"Use {new_name} instead.",
        DeprecationWarning,
        stacklevel=3
    )