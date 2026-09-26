"""Utility functions for STL export.

This module provides validation and helper functions for 3D printing.
"""

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from complexplorer.exceptions import ValidationError

if TYPE_CHECKING:
    import pyvista as pv

# Every edge class ``extract_feature_edges`` can extract. All four default to True, which is the
# trap ``count_edges`` exists to close.
_EDGE_CLASSES = ("boundary_edges", "feature_edges", "manifold_edges", "non_manifold_edges")


def count_edges(mesh: "pv.PolyData", **which: bool) -> int:
    """Count one class of edge, with every other class explicitly switched off.

    ``extract_feature_edges`` enables ``boundary_edges``, ``feature_edges``, ``manifold_edges`` and
    ``non_manifold_edges`` by default, so asking it for boundary edges alone returns substantially
    every edge in the mesh: a closed cube reports hundreds of "boundary" edges because each of its
    creases is a feature edge. Passing all four flags explicitly is the only way to count one class,
    so every call site goes through here rather than repeating the flags and getting them wrong.

    Parameters
    ----------
    mesh : pv.PolyData
        Mesh to measure.
    **which : bool
        The edge classes to enable, named as ``extract_feature_edges`` names them. Any class not
        named is disabled.

    Returns
    -------
    int
        Number of edges (cells) of the requested class.

    Raises
    ------
    ValidationError
        If an unknown edge class is named.
    """
    unknown = sorted(set(which) - set(_EDGE_CLASSES))
    if unknown:
        raise ValidationError(
            f"Unknown edge class(es): {', '.join(unknown)}. Available: {', '.join(_EDGE_CLASSES)}"
        )
    flags = dict.fromkeys(_EDGE_CLASSES, False)
    flags.update(which)
    return int(mesh.extract_feature_edges(**flags).n_cells)


def validate_printability(
    mesh: "pv.PolyData", size_mm: float | None = None, verbose: bool = True
) -> dict[str, Any]:
    """Validate mesh for 3D printing requirements.

    Parameters
    ----------
    mesh : pv.PolyData
        Mesh to validate. Call this **before** centring: the relief is a star-shaped solid about
        the origin, so the radii reported below are only meaningful while the star centre still
        sits there.
    size_mm : float, optional
        Target print size in millimeters. Reported alongside the radial extent, so the numbers can
        be read against the size the piece will be printed at.
    verbose : bool, optional
        Print validation results.

    Returns
    -------
    dict
        Validation results with the following keys:
        - is_watertight: bool, whether mesh is closed
        - is_manifold: bool, whether mesh is manifold
        - n_boundary_edges: int, number of boundary edges
        - n_non_manifold_edges: int, number of non-manifold edges
        - bounds: tuple, mesh bounding box
        - dimensions: tuple, mesh dimensions (x, y, z)
        - volume: float, mesh volume
        - surface_area: float, mesh surface area
        - min_radius_mm: float, smallest distance from the origin to the surface
        - max_radius_mm: float, largest distance from the origin to the surface

        The radii carry the ``_mm`` suffix because this runs on the scaled mesh in ``save_stl``;
        on an unscaled mesh they are in that mesh's own units.

    Notes
    -----
    No wall thickness is reported. A relief has no walls — it is a solid star-shaped body about the
    origin, at least ``2 * min_radius`` thick through the centre — and the quantity that does fail
    on a fused-deposition printer is the ridge width between two adjacent pits, which this does not
    measure. ``min_radius_mm`` is the honest proxy for it.
    """
    results = {}

    # Both the boolean and its count come from one measure, and each measure asks for exactly one
    # edge class. See count_edges: the defaults make the obvious call report a closed mesh as open.
    results["n_boundary_edges"] = count_edges(mesh, boundary_edges=True)
    results["is_watertight"] = results["n_boundary_edges"] == 0

    results["n_non_manifold_edges"] = count_edges(mesh, non_manifold_edges=True)
    results["is_manifold"] = results["n_non_manifold_edges"] == 0

    # Radial extent from the origin, which is where the relief's star centre is until the mesh is
    # centred. Centring moves the box centre to the origin instead, which understates the range
    # badly on a lopsided piece whose star centre and box centre are far apart.
    radii = np.linalg.norm(np.asarray(mesh.points, dtype=float), axis=1)
    results["min_radius_mm"] = float(radii.min()) if radii.size else 0.0
    results["max_radius_mm"] = float(radii.max()) if radii.size else 0.0

    # Get mesh properties
    results["bounds"] = mesh.bounds
    results["dimensions"] = (
        mesh.bounds[1] - mesh.bounds[0],  # x
        mesh.bounds[3] - mesh.bounds[2],  # y
        mesh.bounds[5] - mesh.bounds[4],  # z
    )

    # Calculate volume and surface area
    try:
        results["volume"] = mesh.volume
        results["surface_area"] = mesh.area
    except Exception:
        results["volume"] = None
        results["surface_area"] = None
        warnings.warn(
            "Could not compute volume/area. Mesh may not be watertight.",
            stacklevel=2,
        )

    # Print results if verbose
    if verbose:
        print("=== Mesh Validation Results ===")
        print(
            f"Watertight: {results['is_watertight']} ({results['n_boundary_edges']} boundary edges)"
        )
        print(
            f"Manifold: {results['is_manifold']} ({results['n_non_manifold_edges']} non-manifold edges)"
        )
        print(
            f"Dimensions: {results['dimensions'][0]:.3f} x {results['dimensions'][1]:.3f} x {results['dimensions'][2]:.3f}"
        )

        if results["volume"] is not None:
            print(f"Volume: {results['volume']:.3f}")
            print(f"Surface area: {results['surface_area']:.3f}")

        at_size = f" (at {size_mm}mm)" if size_mm is not None else ""
        print(
            f"Radius from origin{at_size}: "
            f"{results['min_radius_mm']:.3f} to {results['max_radius_mm']:.3f}, "
            "measured before centring"
        )
        print("Ridge width between adjacent features is not measured; min radius is the proxy.")

        # Overall assessment
        print("\n=== Overall Assessment ===")
        if results["is_watertight"] and results["is_manifold"]:
            print("[ok] Mesh is ready for 3D printing!")
        else:
            # Not softened by a size threshold any more. Before repair a sphere relief is genuinely
            # open -- the duplicated seam meridian and a missing cap at each pole -- and after
            # repair it should report zero, so a non-zero count here is a real defect either way.
            print(
                f"[fail] Mesh is not closed: {results['n_boundary_edges']} boundary edges, "
                f"{results['n_non_manifold_edges']} non-manifold edges"
            )
            print("  Repair it with repair_mesh_simple(), which welds the seam and fills the caps.")

    return results


def max_extent(mesh: "pv.PolyData", n_directions: int = 3000) -> float:
    """The object's true tip-to-tip width: the diameter of its convex hull.

    For a convex body the diameter equals the maximum *support width* over directions,
    ``max(p.u) - min(p.u)``, so sampling directions gives it without ever building a hull. That
    matters at this scale: an ornament carries 60k-160k points, and a brute-force pairwise distance
    over even the hull vertices wants tens of gigabytes.

    Fibonacci directions alone run about 0.8% low on a spiky piece, because the widest direction is a
    spike axis and the sampling misses it by up to ~0.06 rad. So the directions of the
    highest-radius points -- which *are* the spike axes on a relief -- are added as candidates, which
    makes the result exact to rounding for these shapes.

    This is the honest measure of how big a piece is. The axis-aligned bounding box is not: it
    depends on how the object happens to sit in the coordinate frame. A relief with spikes on the
    cube diagonals ``(+-1, +-1, +-1)/sqrt(3)`` projects each spike onto a coordinate axis at only
    0.577 of its length, so its box understates it by a factor of ``sqrt(3)``.

    Parameters
    ----------
    mesh : pv.PolyData
        Mesh to measure.
    n_directions : int, default=3000
        How many Fibonacci directions to sample, before the spike axes are added.

    Returns
    -------
    float
        The maximum width, in the mesh's own units.
    """
    points = np.asarray(mesh.points, dtype=np.float64)
    if points.size == 0:
        return 0.0

    i = np.arange(n_directions) + 0.5
    polar = np.arccos(1.0 - 2.0 * i / n_directions)
    azimuth = np.pi * (1.0 + 5.0**0.5) * i
    fibonacci = np.stack(
        [np.cos(azimuth) * np.sin(polar), np.sin(azimuth) * np.sin(polar), np.cos(polar)],
        axis=1,
    )

    # The spike axes, as the directions of the points furthest from the origin.
    radii = np.linalg.norm(points, axis=1)
    tips = points[np.argsort(radii)[-256:]]
    tip_norms = np.linalg.norm(tips, axis=1, keepdims=True)
    tips = tips[tip_norms[:, 0] > 0] / tip_norms[tip_norms[:, 0] > 0]

    directions = np.vstack([fibonacci, tips]) if tips.size else fibonacci
    widest = 0.0
    for start in range(0, len(directions), 256):
        projected = points @ directions[start : start + 256].T
        widest = max(widest, float((projected.max(axis=0) - projected.min(axis=0)).max()))
    return widest


def scale_to_size(mesh: "pv.PolyData", target_size_mm: float, axis: str = "max") -> "pv.PolyData":
    """Scale mesh to target size in millimeters.

    Parameters
    ----------
    mesh : pv.PolyData
        Mesh to scale.
    target_size_mm : float
        Target size in millimeters.
    axis : str, optional
        Which measure to scale to:
        - 'extent': Scale so the true tip-to-tip width equals target_size_mm. This is what the
          ornament path uses, because it is a property of the object rather than of its orientation.
          See :func:`max_extent`.
        - 'max': Scale so the largest axis-aligned bounding-box dimension equals target_size_mm
        - 'x', 'y', 'z': Scale specific axis to target_size_mm

    Returns
    -------
    pv.PolyData
        Scaled mesh.
    """
    bounds = mesh.bounds
    dimensions = [
        bounds[1] - bounds[0],  # x
        bounds[3] - bounds[2],  # y
        bounds[5] - bounds[4],  # z
    ]

    if axis == "extent":
        current_size = max_extent(mesh)
    elif axis == "max":
        current_size = max(dimensions)
    elif axis == "x":
        current_size = dimensions[0]
    elif axis == "y":
        current_size = dimensions[1]
    elif axis == "z":
        current_size = dimensions[2]
    else:
        raise ValidationError(f"Invalid axis: {axis}. Available: extent, max, x, y, z")

    if current_size <= 0:
        raise ValidationError(
            f"Cannot scale a mesh whose {axis} measure is {current_size}; it has no extent to scale"
        )

    scale_factor = target_size_mm / current_size

    # Create scaled copy
    scaled = mesh.copy()
    scaled.points *= scale_factor

    return scaled


def center_mesh(mesh: "pv.PolyData") -> "pv.PolyData":
    """Center mesh at origin.

    Parameters
    ----------
    mesh : pv.PolyData
        Mesh to center.

    Returns
    -------
    pv.PolyData
        Centered mesh.
    """
    centered = mesh.copy()
    center = centered.center
    centered.points -= center

    return centered
