"""Centralized validation utilities for complexplorer.

Only the resolution validator remains here; the broader validation helper set was removed in
3.0 as unused. ``ValidationError`` is re-exported from its canonical home
(:mod:`complexplorer.exceptions`) for backward-compatible imports.
"""

from typing import Any

# Historical import location — the class is defined in complexplorer.exceptions.
from complexplorer.exceptions import ValidationError

__all__ = ["ValidationError", "validate_resolution"]


def validate_resolution(
    resolution: Any, param_name: str = "resolution", min_val: int = 10, max_val: int = 1000
) -> int:
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
    except (TypeError, ValueError) as err:
        raise ValidationError(
            f"{param_name} must be an integer, got {type(resolution).__name__}"
        ) from err

    if res_int < min_val or res_int > max_val:
        raise ValidationError(
            f"{param_name} must be between {min_val} and {max_val}, got {res_int}"
        )

    return res_int
