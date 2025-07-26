"""
Complexplorer: A Python library for visualizing complex functions.

This library provides tools to create beautiful visualizations of complex-valued
functions using various color mapping techniques and plot types.
"""

from complexplorer._version import __version__

# Core functionality
from complexplorer.core.domain import (
    Domain,
    Rectangle,
    Disk,
    Annulus,
    CompositeDomain
)

from complexplorer.core.colormap import (
    Colormap,
    Phase,
    Chessboard,
    PolarChessboard,
    LogRings
)

from complexplorer.core.functions import (
    phase,
    sawtooth,
    stereographic_projection,
    inverse_stereographic
)

from complexplorer.core.scaling import (
    ModulusScaling,
    get_scaling_preset
)

# Plotting functions (matplotlib)
from complexplorer.plotting.matplotlib.plot_2d import (
    plot,
    pair_plot,
    riemann_chart,
    riemann_hemispheres
)

from complexplorer.plotting.matplotlib.plot_3d import (
    plot_landscape,
    pair_plot_landscape,
    riemann
)

# Utility functions
from complexplorer.utils.backend import (
    setup_matplotlib_backend,
    ensure_interactive_plots
)

# High-level API
from complexplorer.api import (
    quick_plot,
    analyze_function,
    visualize,
    explore,
    Presets
)

# STL export (requires PyVista)
try:
    from complexplorer.export.stl import (
        OrnamentGenerator,
        create_ornament
    )
    HAS_STL_EXPORT = True
except ImportError:
    HAS_STL_EXPORT = False

# PyVista plotting (optional, high-performance)
try:
    import pyvista
    from complexplorer.plotting.pyvista.plot_3d import (
        plot_landscape_pv,
        pair_plot_landscape_pv
    )
    from complexplorer.plotting.pyvista.riemann import riemann_pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False


__all__ = [
    # Version
    '__version__',
    
    # Core classes
    'Domain', 'Rectangle', 'Disk', 'Annulus', 'CompositeDomain',
    'Colormap', 'Phase', 'Chessboard', 'PolarChessboard', 'LogRings',
    'ModulusScaling', 'get_scaling_preset',
    
    # Core functions
    'phase', 'sawtooth', 'stereographic_projection', 'inverse_stereographic',
    
    # Plotting functions
    'plot', 'pair_plot', 'riemann_chart', 'riemann_hemispheres',
    'plot_landscape', 'pair_plot_landscape', 'riemann',
    
    # Utilities
    'setup_matplotlib_backend', 'ensure_interactive_plots',
    
    # High-level API
    'quick_plot', 'analyze_function', 'visualize', 'explore', 'Presets',
    
    # Flags
    'HAS_PYVISTA', 'HAS_STL_EXPORT'
]

# Add optional exports
if HAS_STL_EXPORT:
    __all__.extend(['OrnamentGenerator', 'create_ornament'])

if HAS_PYVISTA:
    __all__.extend(['plot_landscape_pv', 'pair_plot_landscape_pv', 'riemann_pv'])