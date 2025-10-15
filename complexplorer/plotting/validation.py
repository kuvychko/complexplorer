"""Validation utilities specific to plotting functions.

This module provides specialized validation for plotting parameters,
consolidating common validation patterns used across matplotlib and PyVista
plotting functions.
"""

from typing import Optional, Dict, Any, Tuple
from complexplorer.exceptions import ValidationError
from complexplorer.core.scaling import ModulusScaling


def validate_modulus_mode(modulus_mode: str,
                         modulus_params: Optional[Dict[str, Any]] = None) -> None:
    """Validate modulus scaling mode and parameters.

    Parameters
    ----------
    modulus_mode : str
        Modulus scaling mode ('linear', 'log', 'arctan', 'sqrt', 'custom', etc.).
    modulus_params : dict, optional
        Parameters for the scaling function.

    Raises
    ------
    ValidationError
        If mode is invalid or required parameters are missing.
    """
    if modulus_params is None:
        modulus_params = {}

    if modulus_mode == 'custom':
        if 'scaling_func' not in modulus_params:
            raise ValidationError(
                "Custom modulus mode requires 'scaling_func' in modulus_params. "
                "Provide a callable that takes a numpy array and returns scaled values."
            )
        if not callable(modulus_params['scaling_func']):
            raise ValidationError(
                f"modulus_params['scaling_func'] must be callable, "
                f"got {type(modulus_params['scaling_func']).__name__}"
            )
    else:
        # Validate that the mode exists as a method on ModulusScaling
        scaling_method = getattr(ModulusScaling, modulus_mode, None)
        if scaling_method is None:
            # Get available modes
            available_modes = [
                name for name in dir(ModulusScaling)
                if not name.startswith('_') and callable(getattr(ModulusScaling, name))
            ]
            raise ValidationError(
                f"Unknown modulus scaling mode: '{modulus_mode}'. "
                f"Available modes: {', '.join(available_modes)}, 'custom'"
            )


def validate_z_max(z_max: Optional[float]) -> None:
    """Validate z_max parameter for plot height clipping.

    Parameters
    ----------
    z_max : float or None
        Maximum height for 3D plots.

    Raises
    ------
    ValidationError
        If z_max is not None and not positive.
    """
    if z_max is not None:
        if not isinstance(z_max, (int, float)):
            raise ValidationError(
                f"z_max must be numeric or None, got {type(z_max).__name__}"
            )
        if z_max <= 0:
            raise ValidationError(
                f"z_max must be positive or None, got {z_max}"
            )


def validate_margin(margin: float,
                   min_margin: float = 0.0,
                   max_margin: float = 0.5) -> None:
    """Validate margin parameter for pair plots.

    Parameters
    ----------
    margin : float
        Margin between subplots as fraction of figure size.
    min_margin : float, optional
        Minimum allowed margin.
    max_margin : float, optional
        Maximum allowed margin.

    Raises
    ------
    ValidationError
        If margin is out of valid range.
    """
    if not isinstance(margin, (int, float)):
        raise ValidationError(
            f"Margin must be numeric, got {type(margin).__name__}"
        )

    if margin < min_margin:
        raise ValidationError(
            f"Margin must be at least {min_margin}, got {margin}"
        )

    if margin > max_margin:
        raise ValidationError(
            f"Margin cannot exceed {max_margin}, got {margin}"
        )


def validate_axes_limits(xlim: Optional[Tuple[float, float]],
                        ylim: Optional[Tuple[float, float]]) -> None:
    """Validate axes limit tuples.

    Parameters
    ----------
    xlim : tuple or None
        X-axis limits (min, max).
    ylim : tuple or None
        Y-axis limits (min, max).

    Raises
    ------
    ValidationError
        If limits are invalid.
    """
    for limit, name in [(xlim, 'xlim'), (ylim, 'ylim')]:
        if limit is not None:
            if not isinstance(limit, (tuple, list)):
                raise ValidationError(
                    f"{name} must be a tuple or list, got {type(limit).__name__}"
                )
            if len(limit) != 2:
                raise ValidationError(
                    f"{name} must have exactly 2 elements (min, max), got {len(limit)}"
                )
            if limit[0] >= limit[1]:
                raise ValidationError(
                    f"{name} min ({limit[0]}) must be less than max ({limit[1]})"
                )


def validate_elevation_azimuth(elevation: float, azimuth: float) -> None:
    """Validate 3D view angles.

    Parameters
    ----------
    elevation : float
        Elevation angle in degrees.
    azimuth : float
        Azimuth angle in degrees.

    Raises
    ------
    ValidationError
        If angles are not numeric.
    """
    if not isinstance(elevation, (int, float)):
        raise ValidationError(
            f"Elevation must be numeric, got {type(elevation).__name__}"
        )
    if not isinstance(azimuth, (int, float)):
        raise ValidationError(
            f"Azimuth must be numeric, got {type(azimuth).__name__}"
        )


def validate_colorbar_params(show_colorbar: bool,
                            colorbar_label: Optional[str]) -> None:
    """Validate colorbar parameters.

    Parameters
    ----------
    show_colorbar : bool
        Whether to show colorbar.
    colorbar_label : str or None
        Label for colorbar.

    Raises
    ------
    ValidationError
        If parameters are invalid.
    """
    if not isinstance(show_colorbar, bool):
        raise ValidationError(
            f"show_colorbar must be boolean, got {type(show_colorbar).__name__}"
        )

    if colorbar_label is not None and not isinstance(colorbar_label, str):
        raise ValidationError(
            f"colorbar_label must be a string or None, "
            f"got {type(colorbar_label).__name__}"
        )


def validate_title(title: Optional[str]) -> None:
    """Validate title parameter.

    Parameters
    ----------
    title : str or None
        Plot title.

    Raises
    ------
    ValidationError
        If title is not a string or None.
    """
    if title is not None and not isinstance(title, str):
        raise ValidationError(
            f"Title must be a string or None, got {type(title).__name__}"
        )


def validate_figure_size(figsize: Optional[Tuple[float, float]]) -> None:
    """Validate figure size parameter.

    Parameters
    ----------
    figsize : tuple or None
        Figure size (width, height) in inches.

    Raises
    ------
    ValidationError
        If figsize is invalid.
    """
    if figsize is not None:
        if not isinstance(figsize, (tuple, list)):
            raise ValidationError(
                f"figsize must be a tuple or list, got {type(figsize).__name__}"
            )
        if len(figsize) != 2:
            raise ValidationError(
                f"figsize must have exactly 2 elements (width, height), "
                f"got {len(figsize)}"
            )
        if not all(isinstance(x, (int, float)) and x > 0 for x in figsize):
            raise ValidationError(
                "figsize dimensions must be positive numbers, got {figsize}"
            )
