from complexplorer.domain import *
from complexplorer.cmap import *
from complexplorer.plots_2d import *
from complexplorer.plots_3d import *
from complexplorer.funcs import *
from complexplorer.utils import setup_matplotlib_backend, ensure_interactive_plots

# Optional PyVista imports (only if PyVista is installed)
try:
    import pyvista
    from complexplorer.plots_3d_pyvista import *
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
