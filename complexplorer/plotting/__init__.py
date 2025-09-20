"""Plotting modules for complexplorer."""

# Import matplotlib plotting functions
from .matplotlib.plot_2d import plot, pair_plot
from .matplotlib.plot_3d import plot_landscape, pair_plot_landscape, riemann

# Try to import PyVista plotting functions  
try:
    from .pyvista.plot_3d import plot_landscape_pv, pair_plot_landscape_pv
    from .pyvista.riemann import riemann_pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

__all__ = [
    'plot',
    'pair_plot',
    'plot_landscape',
    'pair_plot_landscape', 
    'riemann',
]

if HAS_PYVISTA:
    __all__.extend([
        'plot_landscape_pv',
        'pair_plot_landscape_pv',
        'riemann_pv'
    ])