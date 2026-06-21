"""Backend-agnostic sampled complex field.

A ``ComplexField`` is the result of evaluating a complex function over a sampling of the
plane (planar grid) or of the Riemann sphere. It is the shared input to the 3D mesh
builders (``complexplorer.mesh``) and is deliberately **PyVista-free** so that the core /
2D paths stay importable without the 3D backend.

Sampling records where the function is non-finite (in-domain poles / essential
singularities) but does NOT clamp it — the inf→geometry mapping is topology-specific and
belongs to the mesh builders (relief bounds radius; landscape clips/blanks height).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field as _dc_field
from typing import TYPE_CHECKING

import numpy as np

from ..utils.validation import ValidationError
from .functions import inverse_stereographic

if TYPE_CHECKING:
    from .domain import Domain


@dataclass
class ComplexField:
    """A complex function sampled over the plane or the sphere.

    Attributes
    ----------
    kind : str
        ``"planar"`` (sampled on a complex grid) or ``"sphere"`` (sampled on the unit
        Riemann sphere via the canonical stereographic projection).
    w : np.ndarray
        Function values ``f(z)`` (complex), shape matching the sampling grid.
    modulus, phase : np.ndarray
        ``|w|`` and ``arg(w)`` (``arg`` in ``[-pi, pi]`` via ``np.angle``).
    mask : np.ndarray or None
        Boolean array, ``True`` where the sample is *outside* the domain. ``None`` if no
        domain restriction was applied.
    z : np.ndarray or None
        Planar sampling: the complex grid of sample points. ``None`` for spheres.
    sphere_xyz : np.ndarray or None
        Sphere sampling: ``(..., 3)`` unit-sphere Cartesian coordinates of the sample
        points (the base geometry the relief builder distorts radially). ``None`` for
        planar fields.
    metadata : dict
        Free-form provenance (e.g. ``resolution``).
    """

    kind: str
    w: np.ndarray
    modulus: np.ndarray
    phase: np.ndarray
    mask: np.ndarray | None = None
    z: np.ndarray | None = None
    sphere_xyz: np.ndarray | None = None
    metadata: dict = _dc_field(default_factory=dict)

    @property
    def nonfinite(self) -> np.ndarray:
        """Boolean array marking non-finite ``w`` (in-domain singularities)."""
        return ~np.isfinite(self.w)


def _evaluate(func: Callable, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate ``func`` on ``z`` and derive modulus/phase, suppressing math warnings."""
    with np.errstate(all="ignore"):
        w = np.asarray(func(z))
        if w.ndim == 0:  # scalar-valued callable
            w = np.full_like(z, w)
        modulus = np.abs(w)
        phase = np.angle(w)
    return w, modulus, phase


def sample(
    func: Callable,
    domain: Domain | None = None,
    *,
    z: np.ndarray | None = None,
    resolution: int = 100,
    metadata: dict | None = None,
) -> ComplexField:
    """Sample ``func`` over a planar domain (or an explicit ``z`` grid).

    Reuses ``Domain.mesh`` / ``Domain.outmask`` for the grid and out-of-domain mask.
    """
    if z is None:
        if domain is None:
            raise ValidationError("sample() requires either a domain or an explicit z grid")
        z = domain.mesh(resolution)
        mask = domain.outmask(resolution)
    else:
        z = np.asarray(z)
        mask = None

    w, modulus, phase = _evaluate(func, z)
    meta = {"resolution": resolution, **(metadata or {})}
    return ComplexField("planar", w, modulus, phase, mask=mask, z=z, metadata=meta)


def sphere_coordinates(
    resolution: int, *, avoid_poles: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure-numpy latitude/longitude unit-sphere Cartesian coordinates.

    Matches ``RectangularSphereGenerator``'s grid (radius 1) but without constructing any
    PyVista object, so it can be used from the PyVista-free core layer.
    """
    if avoid_poles:
        theta = np.linspace(0.01, np.pi - 0.01, resolution)
    else:
        theta = np.linspace(0.0, np.pi, resolution)
    phi = np.linspace(0.0, 2.0 * np.pi, resolution)
    THETA, PHI = np.meshgrid(theta, phi)
    x = np.sin(THETA) * np.cos(PHI)
    y = np.sin(THETA) * np.sin(PHI)
    z = np.cos(THETA)
    return x, y, z


def sample_sphere(
    func: Callable,
    *,
    resolution: int = 100,
    domain: Domain | None = None,
    avoid_poles: bool = True,
    metadata: dict | None = None,
) -> ComplexField:
    """Sample ``func`` on the unit Riemann sphere using the canonical projection.

    The canonical convention maps ``z = 0`` to the **south** pole (matching
    ``core.functions.stereographic_projection``'s default, the matplotlib ``riemann``
    renderer, and the STL ornament). It is realized here by
    ``inverse_stereographic(..., project_from_north=True)`` (denominator ``1 - z``).
    """
    x, y, zc = sphere_coordinates(resolution, avoid_poles=avoid_poles)
    sphere_xyz = np.stack([x, y, zc], axis=-1)

    with np.errstate(all="ignore"):
        w_plane = inverse_stereographic(x, y, zc, project_from_north=True)
    w, modulus, phase = _evaluate(func, w_plane)

    mask = None
    if domain is not None:
        with np.errstate(all="ignore"):
            mask = ~np.asarray(domain.contains(w_plane))

    meta = {"resolution": resolution, "avoid_poles": avoid_poles, **(metadata or {})}
    return ComplexField(
        "sphere", w, modulus, phase, mask=mask, sphere_xyz=sphere_xyz, metadata=meta
    )


__all__ = ["ComplexField", "sample", "sample_sphere", "sphere_coordinates"]
