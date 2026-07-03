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
