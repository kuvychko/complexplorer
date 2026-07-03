"""Riemann *surface* meshes for multivalued functions (power roots, logarithm).

A Riemann surface is the multi-sheeted cover on which a multivalued function becomes
single-valued — distinct from the Riemann *sphere* (which compactifies the plane for a
single-valued function, see ``plotting.pyvista.riemann``).

The key idea is to **parametrize the cover**, not the ``z``-plane: a Riemann surface is
single-valued over a different base, so it reduces to an ordinary rectangular parameter grid
that maps onto the shared :class:`~complexplorer.mesh.surface.SurfaceMesh` kernel unchanged.

- ``power`` (``w = z**(1/n)``): invert to ``z = w**n`` and sample ``w`` over a disk. Height is
  ``Re(w)``, which places the self-intersection on the negative real axis (the conventional
  principal-branch cut). The surface is one continuous, self-intersecting mesh with ``n``
  sheets.
- ``log`` (``w = log z``): sample ``(r, theta)`` with ``theta`` over ``[0, 2*pi*turns]``. Height
  is ``theta = Im(w)`` — the helicoid.
- ``algebraic`` (``w**2 = P(z)``): the two sheets are the graphs of ``+/-Re(sqrt(P(z)))`` over a
  disk in the ``z``-plane. ``Re(sqrt(zeta)) = sqrt((|zeta| + Re(zeta)) / 2)`` is continuous, so
  each sheet is a continuous graph; the sheets intersect exactly along ``{z : P(z) <= 0}`` — the
  cut curves joining the branch points (roots of ``P``). The phase jump across a cut on each
  sheet is the monodromy: the surface continues onto the other sheet there.

The surface is colored by the phase of the value ``w`` (domain coloring on the surface). This
module is PyVista-backed (3D).
"""

from __future__ import annotations

import numpy as np

from ..core.colormap import Colormap, Phase
from ..utils.validation import ValidationError
from .surface import SurfaceMesh

try:
    import pyvista as pv
except ImportError:  # pragma: no cover - exercised only without the 3D backend
    pv = None

RIEMANN_FAMILIES = ("power", "log", "algebraic")


def _power_grid(n: int, r_max: float, resolution: int):
    """Parameter grid for ``z = w**n``: sample ``w`` over a disk; height ``Re(w)``."""
    if n < 2:
        raise ValidationError(f"power family requires n >= 2, got {n}")
    radius = r_max ** (1.0 / n)
    rho = np.linspace(0.0, radius, resolution)
    # Inclusive endpoint so the phi=0 and phi=2*pi columns coincide -> the seam closes.
    phi = np.linspace(0.0, 2.0 * np.pi, 4 * resolution)
    rho_g, phi_g = np.meshgrid(rho, phi, indexing="ij")
    w = rho_g * np.exp(1j * phi_g)
    z = w**n
    return z.real, z.imag, w.real, w  # X, Y, height=Re(w), value


def _log_grid(turns: int, r_max: float, resolution: int):
    """Parameter grid for ``w = log z``: the helicoid; height ``theta = Im(w)``."""
    if turns < 1:
        raise ValidationError(f"log family requires turns >= 1, got {turns}")
    r = np.linspace(r_max / resolution, r_max, resolution)
    theta = np.linspace(0.0, 2.0 * np.pi * turns, 2 * resolution * turns)
    r_g, theta_g = np.meshgrid(r, theta, indexing="ij")
    w = np.log(r_g) + 1j * theta_g
    x = r_g * np.cos(theta_g)
    y = r_g * np.sin(theta_g)
    return x, y, theta_g, w  # X, Y, height=Im(w)=theta, value


def _algebraic_grid(p, r_max: float, resolution: int):
    """Parameter grid for ``w**2 = P(z)``: sample ``z`` over a disk; sheets ``+/-Re(sqrt(P))``.

    Returns ``(X, Y, Z, w)`` for the principal sheet; the second sheet is its mirror
    ``(X, Y, -Z, -w)``.
    """
    rho = np.linspace(0.0, r_max, resolution)
    # Inclusive endpoint so the phi=0 and phi=2*pi columns coincide -> the seam closes.
    phi = np.linspace(0.0, 2.0 * np.pi, 4 * resolution)
    rho_g, phi_g = np.meshgrid(rho, phi, indexing="ij")
    z = rho_g * np.exp(1j * phi_g)
    w = np.sqrt(np.polyval(p, z))
    return z.real, z.imag, w.real, w  # X, Y, height=Re(w), value


def _validate_polynomial(p) -> np.ndarray:
    """Validate algebraic-family coefficients (numpy.polyval order, highest degree first)."""
    if p is None:
        raise ValidationError(
            "algebraic family requires polynomial coefficients p (highest degree first), "
            "e.g. p=[1, 0, -1, 0] for w**2 = z**3 - z"
        )
    coeffs = np.atleast_1d(np.asarray(p, dtype=complex))
    if coeffs.ndim != 1 or coeffs.size < 2:
        raise ValidationError(
            f"algebraic family requires at least two coefficients (degree >= 1), got {p!r}"
        )
    if coeffs[0] == 0:
        raise ValidationError("algebraic family requires a nonzero leading coefficient in p")
    return coeffs


def build_riemann_surface(
    family: str,
    *,
    n: int = 2,
    turns: int = 3,
    p=None,
    r_max: float = 1.5,
    resolution: int = 60,
    cmap: Colormap | None = None,
) -> SurfaceMesh:
    """Build the Riemann surface of a multivalued family as a :class:`SurfaceMesh`.

    Parameters
    ----------
    family : str
        ``"power"`` (``z**(1/n)``), ``"log"``, or ``"algebraic"`` (``w**2 = P(z)``).
    n : int, default=2
        Sheet count for the power family (sqrt=2, cbrt=3, ...).
    turns : int, default=3
        Number of 2*pi turns for the log helicoid.
    p : sequence of numbers, optional
        Polynomial coefficients of ``P`` for the algebraic family, in ``numpy.polyval``
        order (highest degree first); e.g. ``[1, 0, -1, 0]`` is the elliptic curve
        ``w**2 = z**3 - z``. Required when ``family="algebraic"``.
    r_max : float, default=1.5
        Radius in the ``z``-plane that the surface spans. For the algebraic family choose
        it to enclose the interesting branch points (the roots of ``P``).
    resolution : int, default=60
        Radial sample count (angular samples are derived to keep cells well-shaped).
    cmap : Colormap, optional
        Colormap for the phase of the value. Defaults to ``Phase(n_phi=6, v_base=0.6)``.

    Returns
    -------
    SurfaceMesh
        The embedded surface, colored by the phase of the value. For the algebraic family
        the metadata records the roots of ``P`` under ``branch_points``.
    """
    if pv is None:
        raise ImportError("PyVista is required for 3D mesh building.")
    if family not in RIEMANN_FAMILIES:
        raise ValidationError(f"Unknown family {family!r}; supported: {RIEMANN_FAMILIES}")
    if cmap is None:
        cmap = Phase(n_phi=6, v_base=0.6)

    if family == "algebraic":
        return _build_algebraic_surface(p, r_max, resolution, cmap)

    if family == "power":
        X, Y, Z, w = _power_grid(n, r_max, resolution)
        topology = f"riemann_surface:power:n={n}"
    else:
        X, Y, Z, w = _log_grid(turns, r_max, resolution)
        topology = f"riemann_surface:log:turns={turns}"

    X = X.astype(float).copy()
    Y = Y.astype(float).copy()
    Z = Z.astype(float).copy()

    # Extract the surface FIRST so its point order is the C-order ravel of the grid, matching
    # the value array's ravel order (the same alignment fix used by build_relief).
    surf = pv.StructuredGrid(X, Y, Z).extract_surface(algorithm="dataset_surface")
    sm = SurfaceMesh(surf, metadata={"topology": topology})
    sm.attach_colors(cmap, w)
    return sm


def _build_algebraic_surface(p, r_max: float, resolution: int, cmap: Colormap) -> SurfaceMesh:
    """Assemble the two-sheeted cover of ``w**2 = P(z)``.

    Colors are attached per sheet *before* merging: ``attach_colors`` needs the value
    array's ravel order to match the mesh point order, which no longer holds after merge.
    ``merge_points=False`` keeps the sheets' intersection an emergent crossing rather than
    welding the meshes together.
    """
    coeffs = _validate_polynomial(p)
    X, Y, Z, w = _algebraic_grid(coeffs, r_max, resolution)
    X = X.astype(float).copy()
    Y = Y.astype(float).copy()
    Z = Z.astype(float).copy()

    degree = coeffs.size - 1
    topology = f"riemann_surface:algebraic:deg={degree}"
    metadata = {"topology": topology, "branch_points": np.roots(coeffs).tolist()}

    sheets = []
    for sign in (1.0, -1.0):
        surf = pv.StructuredGrid(X, Y, sign * Z).extract_surface(algorithm="dataset_surface")
        SurfaceMesh(surf).attach_colors(cmap, sign * w)
        sheets.append(surf)

    merged = sheets[0].merge(sheets[1], merge_points=False)
    return SurfaceMesh(merged, metadata=metadata)
