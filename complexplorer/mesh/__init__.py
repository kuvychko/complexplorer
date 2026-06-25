"""PyVista surface-mesh kernel.

A small, additive layer over the existing mesh/STL utilities (which stay where they are):

- ``SurfaceMesh`` wraps a PyVista dataset with a single decorate + export path.
- ``build_landscape`` / ``build_relief`` turn a :class:`complexplorer.core.field.ComplexField`
  into a ``SurfaceMesh`` (planar height map / radially-distorted sphere).

It imports — and does not relocate — ``utils.mesh``, ``utils.mesh_distortion``, and the
``export.stl`` post-processing.
"""

from .builders import build_landscape, build_relief
from .riemann_surface import build_riemann_surface
from .surface import SurfaceMesh

__all__ = ["SurfaceMesh", "build_landscape", "build_relief", "build_riemann_surface"]
