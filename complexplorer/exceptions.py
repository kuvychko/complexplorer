"""Exception hierarchy for complexplorer.

All errors the library raises deliberately derive from :class:`ComplexplorerError`,
so callers can catch everything library-specific with a single handler::

    try:
        cp.quick_plot(f, mode="bogus")
    except cp.ComplexplorerError:
        ...
"""


class ComplexplorerError(Exception):
    """Base class for all complexplorer-domain errors."""


class ValidationError(ComplexplorerError, ValueError):
    """Invalid argument, state, or input data.

    Also subclasses :class:`ValueError`, so pre-3.0 ``except ValueError`` handlers
    continue to catch it.
    """


__all__ = ["ComplexplorerError", "ValidationError"]


class ColormapError(ValidationError):
    """Raised for invalid colormap configuration.

    A ``ValidationError`` (and therefore a ``ComplexplorerError`` and a ``ValueError``), so
    handlers written against either contract keep working. Restored in 3.0 for parity with
    2.0.0, which raised it for invalid colormap parameters.
    """
