"""The public typing contract.

The package ships ``py.typed``, so its annotations are a promise to every downstream type
checker. This module holds the pieces of that promise which are not obvious from a signature:
what a "complex function" is as far as this library is concerned, and the shapes of the
dictionaries the catalog serializes.

Nothing here is enforced at runtime. A ``Protocol`` is a description a type checker checks
structurally; the library does no isinstance testing against it.
"""

from __future__ import annotations

from typing import Any, Protocol, TypedDict, runtime_checkable

import numpy as np

__all__ = [
    "ComplexFunction",
    "DomainSpec",
    "CmapSpec",
    "ScalingSpec",
    "SingularityRecord",
]


@runtime_checkable
class ComplexFunction(Protocol):
    """A function of a complex variable, as this library actually calls it.

    Every renderer samples its domain into a numpy array and calls the function **once** with
    that whole array, not once per point. A function therefore has to be vectorized -- which
    is automatic for anything written with arithmetic and numpy ufuncs::

        lambda z: (z**2 - 1) / (z**2 + 1)      # satisfies this protocol
        np.sin                                  # so does a bare ufunc
        cp.ee.TransferFunction([1], [1, 1])     # and so does a callable object

    The protocol is structural, so none of those had to be declared as anything: a callable
    that accepts an array of complex values and returns an array of the same shape satisfies
    it. Scalar input is accepted too, because the natural way to write such a function also
    works one point at a time.

    This replaces ``Callable[[complex], complex]``, which described a scalar contract the
    library never used, and which rejected correctly written vectorized functions.
    """

    def __call__(self, z: np.ndarray, /) -> np.ndarray:
        """Map an array of complex values to an array of complex values."""
        ...


class DomainSpec(TypedDict, total=False):
    """A serialized :class:`~complexplorer.Domain`, as stored in a preset and the manifest.

    ``type`` selects the class; the remaining keys mirror that class's constructor arguments.
    Complex values are ``[re, im]`` pairs, because JSON has no complex type.
    """

    type: str
    re_length: float
    im_length: float
    radius: float
    inner_radius: float
    outer_radius: float
    center: list[float]
    square: bool


class CmapSpec(TypedDict, total=False):
    """A serialized :class:`~complexplorer.Colormap`.

    ``type`` is the class name (``"Phase"``, ``"CubehelixPhase"``, ...) and the rest mirror its
    constructor. The sector-count key is ``phase_sectors``; it was ``n_phi`` before 2.0.
    """

    type: str
    phase_sectors: int
    auto_scale_r: bool
    r_linear_step: float
    r_log_base: float
    v_base: float
    scale_radius: float
    emphasize_unit_circle: bool
    spacing: float
    log_spacing: float


class ScalingSpec(TypedDict, total=False):
    """A serialized modulus scaling: a mode and its parameters.

    A preset may also carry a plain string naming a preset scaling, such as ``"balanced"``.
    """

    method: str
    params: dict[str, Any]


class SingularityRecord(TypedDict, total=False):
    """One entry of a preset's answer key.

    ``at`` is an ``[re, im]`` pair. ``order`` is the multiplicity for a zero or pole, the
    branching order for a branch point, and ``None`` for an essential singularity. ``label`` is
    an optional display name such as ``"pi/2"``.
    """

    type: str
    at: list[float]
    order: int | None
    label: str
