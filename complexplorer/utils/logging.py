"""Logging utilities for complexplorer.

This module provides a centralized logging configuration for the library,
allowing users to control verbosity and debug output across all components.
"""

import logging
import sys
from typing import Optional
from pathlib import Path


# Package logger name
LOGGER_NAME = "complexplorer"

# Default format for log messages
DEFAULT_FORMAT = "%(levelname)s [%(name)s.%(funcName)s] %(message)s"
DEBUG_FORMAT = "%(asctime)s %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s"


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger for complexplorer.

    Parameters
    ----------
    name : str, optional
        Submodule name. If provided, creates logger named 'complexplorer.name'.
        If None, returns the root complexplorer logger.

    Returns
    -------
    logging.Logger
        Configured logger instance.

    Examples
    --------
    >>> # In a module
    >>> logger = get_logger('core.colormap')
    >>> logger.info("Computing colormap...")

    >>> # Get root logger
    >>> logger = get_logger()
    """
    if name:
        return logging.getLogger(f"{LOGGER_NAME}.{name}")
    return logging.getLogger(LOGGER_NAME)


def setup_logging(
    level: int = logging.WARNING,
    format_string: Optional[str] = None,
    handler: Optional[logging.Handler] = None,
    propagate: bool = True
) -> None:
    """Configure logging for complexplorer.

    This function sets up the root logger for the complexplorer package.
    Call this once at the start of your script to configure logging behavior.

    Parameters
    ----------
    level : int, optional
        Logging level (logging.DEBUG, logging.INFO, logging.WARNING, etc.).
        Default is WARNING.
    format_string : str, optional
        Custom format string for log messages.
        If None, uses DEFAULT_FORMAT (or DEBUG_FORMAT if level is DEBUG).
    handler : logging.Handler, optional
        Custom handler. If None, creates a StreamHandler to stderr.
    propagate : bool, optional
        Whether to propagate to parent loggers. Default is True.

    Examples
    --------
    >>> # Enable info-level logging
    >>> setup_logging(level=logging.INFO)

    >>> # Enable debug logging with timestamps
    >>> setup_logging(level=logging.DEBUG)

    >>> # Log to file
    >>> import logging
    >>> file_handler = logging.FileHandler('complexplorer.log')
    >>> setup_logging(level=logging.DEBUG, handler=file_handler)
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = propagate

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create handler
    if handler is None:
        handler = logging.StreamHandler(sys.stderr)

    # Set format
    if format_string is None:
        format_string = DEBUG_FORMAT if level == logging.DEBUG else DEFAULT_FORMAT

    formatter = logging.Formatter(format_string)
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def enable_debug_logging() -> None:
    """Enable debug-level logging with detailed format.

    Convenience function to quickly enable debug output.

    Examples
    --------
    >>> enable_debug_logging()
    >>> # Now all debug messages will be shown
    """
    setup_logging(level=logging.DEBUG, format_string=DEBUG_FORMAT)


def disable_logging() -> None:
    """Disable all logging output from complexplorer.

    Sets logging level to CRITICAL+1 to suppress all messages.

    Examples
    --------
    >>> disable_logging()
    >>> # Now no log messages will be shown
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.CRITICAL + 1)


def log_function_call(logger: logging.Logger, func_name: str, **kwargs) -> None:
    """Log a function call with its parameters.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance to use.
    func_name : str
        Name of the function being called.
    **kwargs
        Function parameters to log.

    Examples
    --------
    >>> logger = get_logger('core.colormap')
    >>> log_function_call(logger, 'oklch_to_srgb', L=0.5, C=0.15, H=45)
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return

    param_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
    logger.debug(f"Called {func_name}({param_str})")


def log_array_stats(logger: logging.Logger, name: str, array) -> None:
    """Log statistics about a numpy array.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance to use.
    name : str
        Name of the array for logging.
    array : np.ndarray
        Array to log statistics for.

    Examples
    --------
    >>> import numpy as np
    >>> logger = get_logger('core.domain')
    >>> z = np.linspace(-2, 2, 100)
    >>> log_array_stats(logger, 'domain_mesh', z)
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return

    import numpy as np
    array = np.asarray(array)

    logger.debug(
        f"{name}: shape={array.shape}, dtype={array.dtype}, "
        f"min={np.min(array):.3g}, max={np.max(array):.3g}, "
        f"mean={np.mean(array):.3g}"
    )


def log_performance(logger: logging.Logger, operation: str, elapsed_time: float) -> None:
    """Log performance timing information.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance to use.
    operation : str
        Description of the operation.
    elapsed_time : float
        Time elapsed in seconds.

    Examples
    --------
    >>> import time
    >>> logger = get_logger('plotting')
    >>> start = time.time()
    >>> # ... do work ...
    >>> log_performance(logger, 'mesh_generation', time.time() - start)
    """
    if not logger.isEnabledFor(logging.INFO):
        return

    if elapsed_time < 0.001:
        time_str = f"{elapsed_time * 1000000:.1f} µs"
    elif elapsed_time < 1.0:
        time_str = f"{elapsed_time * 1000:.1f} ms"
    else:
        time_str = f"{elapsed_time:.2f} s"

    logger.info(f"{operation} completed in {time_str}")


# Initialize with default configuration (WARNING level, no output unless explicitly configured)
# Users can call setup_logging() to configure as needed
_default_logger = logging.getLogger(LOGGER_NAME)
_default_logger.addHandler(logging.NullHandler())
