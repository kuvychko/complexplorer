"""Custom exceptions for complexplorer.

This module defines the exception hierarchy used throughout the library,
providing clear, specific error types for different failure modes.

Exception Hierarchy
-------------------
ComplexplorerError (base)
├── ValidationError
│   ├── DomainError
│   ├── ColormapError
│   └── ResolutionError
├── ComputationError
│   ├── FunctionEvaluationError
│   └── MeshGenerationError
├── ExportError
│   ├── STLExportError
│   └── ImageExportError
└── DependencyError
    ├── PyVistaNotAvailableError
    └── OptionalDependencyError
"""


class ComplexplorerError(Exception):
    """Base exception for all complexplorer errors.

    All custom exceptions in complexplorer inherit from this base class,
    making it easy to catch any library-specific error.

    Examples
    --------
    >>> try:
    ...     # complexplorer code
    ...     pass
    ... except ComplexplorerError as e:
    ...     print(f"Complexplorer error: {e}")
    """
    pass


# ============================================================================
# Validation Errors
# ============================================================================

class ValidationError(ComplexplorerError, ValueError):
    """Invalid input parameters or arguments.

    Raised when user-provided parameters fail validation checks.
    Inherits from both ComplexplorerError and ValueError for backward
    compatibility with existing code that catches ValueError.

    Examples
    --------
    >>> raise ValidationError("Resolution must be positive")
    """
    pass


class DomainError(ValidationError):
    """Invalid domain specification or parameters.

    Raised when domain parameters are invalid, such as negative dimensions,
    equal bounds, or incompatible domain operations.

    Examples
    --------
    >>> raise DomainError("Rectangle dimensions must be positive")
    >>> raise DomainError("Inner radius must be less than outer radius")
    """
    pass


class ColormapError(ValidationError):
    """Invalid colormap parameters or configuration.

    Raised when colormap parameters are out of valid range or incompatible
    with each other.

    Examples
    --------
    >>> raise ColormapError("n_phi must be specified when auto_scale_r=True")
    >>> raise ColormapError("v_base must be in [0, 1)")
    """
    pass


class ResolutionError(ValidationError):
    """Invalid resolution parameter.

    Raised when resolution is out of acceptable range or incompatible
    with computational resources.

    Examples
    --------
    >>> raise ResolutionError("Resolution must be between 10 and 1000")
    """
    pass


# ============================================================================
# Computation Errors
# ============================================================================

class ComputationError(ComplexplorerError, RuntimeError):
    """Error during mathematical computation.

    Raised when mathematical operations fail during execution.
    Inherits from RuntimeError for compatibility.

    Examples
    --------
    >>> raise ComputationError("Function evaluation produced invalid values")
    """
    pass


class FunctionEvaluationError(ComputationError):
    """Error evaluating complex function.

    Raised when a user-provided function fails to evaluate or produces
    invalid results.

    Examples
    --------
    >>> raise FunctionEvaluationError(
    ...     "Function must return complex values, got string"
    ... )
    """
    pass


class MeshGenerationError(ComputationError):
    """Error generating domain mesh.

    Raised when mesh generation fails due to numerical issues or
    invalid configurations.

    Examples
    --------
    >>> raise MeshGenerationError("Failed to generate valid Riemann sphere mesh")
    """
    pass


# ============================================================================
# Export Errors
# ============================================================================

class ExportError(ComplexplorerError, IOError):
    """Error exporting visualization or data.

    Raised when file export operations fail.
    Inherits from IOError for compatibility.

    Examples
    --------
    >>> raise ExportError("Failed to write file: permission denied")
    """
    pass


class STLExportError(ExportError):
    """Error exporting STL file for 3D printing.

    Raised when STL mesh generation or file writing fails.

    Examples
    --------
    >>> raise STLExportError("Mesh is not watertight: 5 open edges found")
    """
    pass


class ImageExportError(ExportError):
    """Error exporting image file.

    Raised when image rendering or file writing fails.

    Examples
    --------
    >>> raise ImageExportError("Unsupported image format: .xyz")
    """
    pass


# ============================================================================
# Dependency Errors
# ============================================================================

class DependencyError(ComplexplorerError, ImportError):
    """Missing required or optional dependency.

    Raised when a required Python package is not installed.
    Inherits from ImportError for compatibility.

    Examples
    --------
    >>> raise DependencyError("NumPy is required but not installed")
    """
    pass


class PyVistaNotAvailableError(DependencyError):
    """PyVista is not installed.

    Raised when PyVista functionality is requested but the package
    is not available.

    Examples
    --------
    >>> raise PyVistaNotAvailableError(
    ...     "PyVista is required for this function. "
    ...     "Install with: pip install pyvista"
    ... )
    """
    pass


class OptionalDependencyError(DependencyError):
    """Optional dependency not available.

    Raised when an optional feature requires a package that is not installed.

    Examples
    --------
    >>> raise OptionalDependencyError(
    ...     "PyQt6 is required for interactive matplotlib plots. "
    ...     "Install with: pip install complexplorer[qt]"
    ... )
    """
    pass


# ============================================================================
# Convenience function for migration
# ============================================================================

def _get_legacy_exception_mapping():
    """Map old exception types to new hierarchy.

    This helps with gradual migration of existing code.

    Returns
    -------
    dict
        Mapping from exception context to recommended new type.
    """
    return {
        'domain': DomainError,
        'colormap': ColormapError,
        'resolution': ResolutionError,
        'function': FunctionEvaluationError,
        'mesh': MeshGenerationError,
        'stl': STLExportError,
        'export': ExportError,
        'pyvista': PyVistaNotAvailableError,
        'dependency': DependencyError,
    }
