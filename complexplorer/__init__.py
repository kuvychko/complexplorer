"""Temporary compatibility layer during migration."""
import warnings

# Import from legacy to maintain compatibility
from complexplorer.legacy.domain import *
from complexplorer.legacy.cmap import *
from complexplorer.legacy.plots_2d import *
from complexplorer.legacy.plots_3d import *
from complexplorer.legacy.funcs import *
from complexplorer.legacy.utils import setup_matplotlib_backend, ensure_interactive_plots
from complexplorer.legacy.stl_export import *

# Export internal classes needed by tests
from complexplorer.legacy.mesh_utils import (
    RectangularSphereGenerator,
    stereographic_projection,
    inverse_stereographic,
    ModulusScaling
)

# Optional PyVista imports (only if PyVista is installed)
try:
    import pyvista
    from complexplorer.legacy.plots_3d_pyvista import *
    # Export internal functions needed by tests
    from complexplorer.legacy.plots_3d_pyvista import (
        _create_complex_surface,
        _handle_export,
        _ensure_pyvista_setup,
        _add_complex_mesh_to_plotter,
        _add_axes_widget
    )
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

warnings.warn(
    "Complexplorer is undergoing a major refactoring. "
    "The API will change in the next version. "
    "Please see the migration guide for details.",
    DeprecationWarning,
    stacklevel=2
)
