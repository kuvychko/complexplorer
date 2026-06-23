"""Topology builders: turn a ``ComplexField`` into a ``SurfaceMesh``.

- ``build_landscape`` — planar height map (``height = scale(|f|)``), reproducing the
  current ``plotting.pyvista.plot_3d.create_complex_surface``.
- ``build_relief`` — radially-distorted sphere (``radius = scale(|f|)``), reproducing the
  current ``OrnamentGenerator`` / ``riemann_pv`` mesh, but on the canonical projection the
  field already applies (so ``riemann_pv`` flips to ``z=0`` at the south pole).

PyVista enters only here; ``ComplexField`` stays PyVista-free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..core.colormap import Colormap, Phase
from ..core.scaling import ModulusScaling
from ..utils.mesh_distortion import apply_modulus_distortion, get_default_scaling_params
from ..utils.validation import ValidationError
from .surface import SurfaceMesh

if TYPE_CHECKING:
    from ..core.field import ComplexField

try:
    import pyvista as pv
except ImportError:  # pragma: no cover - exercised only without the 3D backend
    pv = None


def _default_cmap() -> Colormap:
    return Phase(n_phi=6, v_base=0.6)


def build_landscape(
    field: ComplexField,
    *,
    cmap: Colormap | None = None,
    z_scale: float = 1.0,
    log_z: bool = False,
    z_max: float | None = None,
    modulus_mode: str = "none",
    modulus_params: dict | None = None,
) -> SurfaceMesh:
    """Build a planar landscape mesh from a planar ``ComplexField``."""
    if pv is None:
        raise ImportError("PyVista is required for 3D mesh building.")
    if field.kind != "planar" or field.z is None:
        raise ValidationError("build_landscape requires a planar ComplexField")
    if cmap is None:
        cmap = _default_cmap()

    z = field.z
    mask = field.mask
    w = np.asarray(field.w).copy()
    if mask is not None:
        w[mask] = np.nan

    magnitude = np.abs(w)
    if modulus_mode != "none":
        if modulus_params is None:
            modulus_params = get_default_scaling_params(modulus_mode)
        if modulus_mode == "custom":
            if "scaling_func" not in modulus_params:
                raise ValidationError("Custom mode requires 'scaling_func' in modulus_params")
            magnitude = modulus_params["scaling_func"](magnitude)
        else:
            scaling_method = getattr(ModulusScaling, modulus_mode, None)
            if scaling_method is None:
                raise ValidationError(f"Unknown scaling mode: {modulus_mode}")
            magnitude = scaling_method(magnitude, **modulus_params)

    if z_max is not None:
        magnitude = np.clip(magnitude, 0, z_max)

    if log_z:
        with np.errstate(divide="ignore", invalid="ignore"):
            height = np.log1p(magnitude) * z_scale
    else:
        height = magnitude * z_scale

    X = np.real(z)
    Y = np.imag(z)
    Z = np.asarray(height, dtype=float).copy()
    if mask is not None:
        Z[mask] = np.nan

    grid = pv.StructuredGrid(X, Y, Z)
    sm = SurfaceMesh(grid, field, metadata={"topology": "landscape"})
    sm.attach_colors(cmap, w, outmask=mask)
    sm.attach_scalar("magnitude", magnitude)
    return sm


def build_relief(
    field: ComplexField,
    *,
    cmap: Colormap | None = None,
    scaling: str = "arctan",
    scaling_params: dict | None = None,
    for_stl: bool = False,
) -> SurfaceMesh:
    """Build a radially-distorted sphere (relief) mesh from a sphere ``ComplexField``."""
    if pv is None:
        raise ImportError("PyVista is required for 3D mesh building.")
    if field.kind != "sphere" or field.sphere_xyz is None:
        raise ValidationError("build_relief requires a sphere ComplexField")
    if cmap is None:
        cmap = _default_cmap()
    if scaling_params is None:
        scaling_params = get_default_scaling_params(scaling, for_stl=for_stl)

    xyz = np.asarray(field.sphere_xyz)
    X = xyz[..., 0].astype(float).copy()
    Y = xyz[..., 1].astype(float).copy()
    Z = xyz[..., 2].astype(float).copy()
    mask = field.mask

    # Radial distortion on the full (ravel-ordered) point set.
    points_flat = xyz.reshape(-1, 3)
    moduli_flat = np.asarray(field.modulus).ravel()
    scaled_flat, radii = apply_modulus_distortion(points_flat, moduli_flat, scaling, scaling_params)

    # Blank out-of-domain points so their cells can be removed (matches the generator).
    if mask is not None:
        X[mask] = np.nan
        Y[mask] = np.nan
        Z[mask] = np.nan

    # Extract the surface FIRST: its point order is the C-order ravel of the grid, which
    # matches the field's (meshgrid) order — unlike the StructuredGrid's native VTK order.
    surf = pv.StructuredGrid(X, Y, Z).extract_surface(algorithm="dataset_surface")

    sm = SurfaceMesh(surf, field, metadata={"topology": "relief"})
    sm.attach_colors(cmap, np.asarray(field.w), outmask=None)
    sm.attach_scalar("magnitude", field.modulus)
    sm.attach_scalar("radius", radii)
    # Carry the scaled coordinates as point arrays so they survive cell removal.
    surf["__sx"] = scaled_flat[:, 0]
    surf["__sy"] = scaled_flat[:, 1]
    surf["__sz"] = scaled_flat[:, 2]

    if mask is not None and not np.all(np.isfinite(surf.points)):
        cells_to_remove = [
            i for i in range(surf.n_cells) if np.any(np.isnan(surf.get_cell(i).points))
        ]
        if cells_to_remove:
            surf = surf.remove_cells(cells_to_remove)

    surf.points = np.column_stack([surf["__sx"], surf["__sy"], surf["__sz"]])
    for tmp in ("__sx", "__sy", "__sz"):
        surf.point_data.remove(tmp)

    sm.mesh = surf
    return sm
