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

# Legacy compatibility layer (deprecated)
import warnings

def _setup_legacy_compatibility():
    """Set up legacy imports for backward compatibility."""
    # Only import if someone tries to use legacy names
    import sys
    module = sys.modules[__name__]
    
    # Import legacy domain classes
    from complexplorer.legacy.domain import (
        Domain as LegacyDomain,
        Rectangle as LegacyRectangle, 
        Disk as LegacyDisk,
        Annulus as LegacyAnnulus
    )
    
    # Legacy domain classes
    module.Domain = LegacyDomain
    module.Rectangle = LegacyRectangle
    module.Disk = LegacyDisk
    module.Annulus = LegacyAnnulus
    
    # Legacy domain names
    module.rectangle = LegacyRectangle
    module.disk = LegacyDisk
    module.annulus = LegacyAnnulus
    
    # Import legacy colormap classes (non-abstract Cmap and concrete implementations)
    from complexplorer.legacy.cmap import (
        Cmap,
        Phase as LegacyPhase,
        Chessboard as LegacyChessboard,
        PolarChessboard as LegacyPolarChessboard,
        LogRings as LegacyLogRings
    )
    
    # Legacy colormap names - use legacy implementations
    module.Cmap = Cmap  # Legacy non-abstract base class
    
    # Override new colormap classes with legacy ones for full compatibility
    module.Phase = LegacyPhase
    module.Chessboard = LegacyChessboard
    module.PolarChessboard = LegacyPolarChessboard
    module.LogRings = LegacyLogRings
    
    # Import legacy plotting functions 
    from complexplorer.legacy.plots_2d import (
        plot as legacy_plot,
        pair_plot as legacy_pair_plot,
        riemann_chart as legacy_riemann_chart,
        riemann_hemispheres as legacy_riemann_hemispheres
    )
    from complexplorer.legacy.plots_3d import (
        plot_landscape as legacy_plot_landscape,
        pair_plot_landscape as legacy_pair_plot_landscape,
        riemann as legacy_riemann
    )
    
    # Legacy function names
    module.plot_domain_coloring = legacy_plot
    module.pair_plot_domain_coloring = legacy_pair_plot
    module.plot = legacy_plot
    module.pair_plot = legacy_pair_plot
    module.plot_landscape = legacy_plot_landscape
    module.pair_plot_landscape = legacy_pair_plot_landscape
    module.riemann = legacy_riemann
    module.riemann_chart = legacy_riemann_chart
    module.riemann_hemispheres = legacy_riemann_hemispheres
    
    # Legacy function exports
    from complexplorer.legacy.funcs import (
        phase as legacy_phase,
        stereographic as legacy_stereographic,
        sawtooth as legacy_sawtooth
    )
    module.phase = legacy_phase  # Use legacy version for compatibility
    module.stereographic = legacy_stereographic  # Use legacy version that returns (x,y,z)
    module.sawtooth = legacy_sawtooth  # Use legacy version for compatibility
    
    # Import all legacy functions for full compatibility
    from complexplorer.legacy import domain as legacy_domain
    from complexplorer.legacy import cmap as legacy_cmap
    from complexplorer.legacy import plots_2d as legacy_2d
    from complexplorer.legacy import plots_3d as legacy_3d
    from complexplorer.legacy import funcs as legacy_funcs
    from complexplorer.legacy import stl_export as legacy_stl
    
    # Add legacy mesh utils for tests
    from complexplorer.legacy.mesh_utils import (
        RectangularSphereGenerator,
        stereographic_projection as mesh_stereo_proj,
        inverse_stereographic as mesh_inverse_stereo,
        ModulusScaling as LegacyModulusScaling
    )
    module.RectangularSphereGenerator = RectangularSphereGenerator
    module.stereographic_projection = mesh_stereo_proj  # This is the 3D version
    module.inverse_stereographic = mesh_inverse_stereo
    # Note: ModulusScaling is imported from new code at top level
    
    # Legacy STL exports
    for name in ['StlExporter', 'spherical_shell_healing', 'optimize_phase_coloring']:
        if hasattr(legacy_stl, name):
            setattr(module, name, getattr(legacy_stl, name))
    
    # Legacy PyVista functions
    if HAS_PYVISTA:
        try:
            from complexplorer.legacy import plots_3d_pyvista as legacy_pv
            # Export all PyVista functions
            module.plot_landscape_pv = legacy_pv.plot_landscape_pv
            module.pair_plot_landscape_pv = legacy_pv.pair_plot_landscape_pv
            module.riemann_pv = legacy_pv.riemann_pv
            
            # Export internal functions needed by tests
            for name in ['_create_complex_surface', '_handle_export', 
                        '_ensure_pyvista_setup', '_add_complex_mesh_to_plotter',
                        '_add_axes_widget']:
                if hasattr(legacy_pv, name):
                    setattr(module, name, getattr(legacy_pv, name))
        except ImportError:
            pass
    
    warnings.warn(
        "You are using legacy Complexplorer APIs that will be removed in v2.0. "
        "Please update your code to use the new API. "
        "See https://github.com/mtorpey/complexplorer for migration guide.",
        DeprecationWarning,
        stacklevel=3
    )

# Always set up legacy compatibility for now during migration
_setup_legacy_compatibility()

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