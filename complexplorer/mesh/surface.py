"""``SurfaceMesh`` — a PyVista mesh wrapper with one decorate + export path.

Wraps a ``pyvista`` dataset together with attached color/scalar fields. The colormap
decoration is faithful: non-finite ``f`` produces non-finite RGB, which is left as-is
(PyVista tolerates NaN scalars — unlike matplotlib's 3D facecolors). STL export reuses the
existing ``export.stl`` post-processing in place (no relocation).
"""

from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from ..core.colormap import Colormap
from ..export.stl.mesh_repair import repair_mesh_simple
from ..export.stl.utils import center_mesh, scale_to_size, validate_printability

if TYPE_CHECKING:
    import pyvista as pv

    from ..core.field import ComplexField


def _flatten_rgb(rgb: np.ndarray, n_points: int) -> np.ndarray:
    """Reshape colormap RGB (``(H, W, 3)`` or ``(N, 3)``) to ``(n_points, 3)``."""
    rgb = np.asarray(rgb)
    flat = rgb.reshape(-1, 3)
    if flat.shape[0] != n_points:
        raise ValueError(f"RGB has {flat.shape[0]} rows but mesh has {n_points} points")
    return flat


class SurfaceMesh:
    """A complex-function surface: a PyVista dataset plus decoration, scalars, and export.

    Parameters
    ----------
    mesh : pyvista.DataSet
        The underlying geometry (a ``StructuredGrid`` or ``PolyData``).
    field : ComplexField, optional
        The field this mesh was built from (provenance; not required for export).
    metadata : dict, optional
        Free-form provenance.
    """

    def __init__(
        self,
        mesh: pv.DataSet,
        field: ComplexField | None = None,
        metadata: dict | None = None,
    ):
        self.mesh = mesh
        self.field = field
        self.metadata = metadata or {}

    # -- decoration -------------------------------------------------------------------

    def attach_colors(
        self, cmap: Colormap, w: np.ndarray, *, outmask: np.ndarray | None = None
    ) -> SurfaceMesh:
        """Attach ``RGB`` and ``phase`` from a colormap (the single shared decorate path).

        ``w`` is the (possibly mask-blanked) complex values; ``outmask`` is passed straight
        to ``cmap.rgb`` so out-of-domain points get the colormap's out-of-domain color. The
        RGB is left faithful (no NaN sanitization).
        """
        rgb = cmap.rgb(w, outmask=outmask)
        self.mesh["RGB"] = _flatten_rgb(rgb, self.mesh.n_points)
        self.mesh["phase"] = np.angle(w).ravel()
        return self

    def attach_scalar(self, name: str, values: np.ndarray) -> SurfaceMesh:
        """Attach a named per-point scalar (e.g. ``magnitude``, ``radius``)."""
        self.mesh[name] = np.asarray(values).ravel()
        return self

    # -- access / export --------------------------------------------------------------

    def to_pyvista(self) -> pv.DataSet:
        """Return the underlying PyVista dataset for custom work."""
        return self.mesh

    def _triangulated(self) -> pv.PolyData:
        """A triangulated surface suitable for STL export."""
        surf = self.mesh
        if not hasattr(surf, "faces"):  # e.g. StructuredGrid -> surface
            surf = surf.extract_surface(algorithm="dataset_surface")
        return surf.triangulate()

    def validate_printability(self, size_mm: float = 50, verbose: bool = False) -> dict:
        """Run the shared 3D-printing validation on the (triangulated) mesh."""
        return validate_printability(self._triangulated(), size_mm, verbose)

    def save_stl(
        self,
        filename: str,
        *,
        size_mm: float = 50,
        center: bool = True,
        repair: bool = True,
        binary: bool = True,
        validate: bool = True,
        verbose: bool = False,
    ) -> str:
        """Triangulate and export to STL, reusing the existing ``export.stl`` post-proc."""
        mesh = self._triangulated()

        # Drop non-finite vertices so a landscape (or masked) surface stays printable.
        if not np.all(np.isfinite(mesh.points)):
            finite = np.all(np.isfinite(mesh.points), axis=1)
            mesh = (
                mesh.extract_points(finite, adjacent_cells=False)
                .extract_surface(algorithm="dataset_surface")
                .triangulate()
            )

        if repair:
            mesh = repair_mesh_simple(mesh, fill_holes=True, verbose=verbose)
        if center:
            mesh = center_mesh(mesh)
        mesh = scale_to_size(mesh, size_mm, axis="max")

        if validate:
            results = validate_printability(mesh, size_mm, verbose=verbose)
            if not results["is_watertight"] and not repair:
                warnings.warn(
                    "Mesh is not watertight. Consider enabling repair=True.", stacklevel=2
                )

        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        mesh.save(filename, binary=binary)
        return filename

    def screenshot(
        self, filename: str, *, window_size: tuple[int, int] = (1024, 768), **kwargs: Any
    ) -> str:
        """Render the mesh off-screen and save an image (headless-friendly)."""
        import pyvista as pv

        plotter = pv.Plotter(off_screen=True, window_size=window_size)
        plotter.add_mesh(
            self.mesh,
            scalars="RGB" if "RGB" in self.mesh.array_names else None,
            rgb="RGB" in self.mesh.array_names,
            **kwargs,
        )
        plotter.screenshot(filename)
        plotter.close()
        return filename
