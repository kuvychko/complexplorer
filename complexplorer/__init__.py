"""
Complexplorer: A Python library for visualizing complex functions.

This library provides tools to create beautiful visualizations of complex-valued
functions using various color mapping techniques and plot types.
"""

# Engineering mode (namespaced: cp.ee.TransferFunction, cp.ee.bode_plot, ...)
from complexplorer import ee
from complexplorer._version import __version__

# High-level API
from complexplorer.api import PlotPresets, quick_plot
from complexplorer.core.colormap import (
    AnalogousWedge,
    Chessboard,
    Colormap,
    CubehelixPhase,
    DivergingWarmCool,
    EarthTopographic,
    FourQuadrant,
    InkPaper,
    Isoluminant,
    LogRings,
    OklabPhase,
    PerceptualPastel,
    Phase,
    PolarChessboard,
)

# Core functionality
from complexplorer.core.domain import Annulus, CompositeDomain, Disk, Domain, Rectangle
from complexplorer.core.functions import (
    inverse_stereographic,
    phase,
    sawtooth,
    stereographic_projection,
)
from complexplorer.core.presets import FunctionPreset, catalog
from complexplorer.core.scaling import ModulusScaling, get_scaling_preset
from complexplorer.exceptions import ColormapError, ComplexplorerError, ValidationError

# PyVista is a required core dependency as of 3.0 (the sole 3D backend; also powers STL
# export), so these imports are unconditional.
from complexplorer.export.stl import OrnamentGenerator, create_ornament
from complexplorer.gallery import generate_gallery

# Plotting functions (matplotlib)
from complexplorer.plotting.matplotlib.plot_2d import (
    pair_plot,
    plot,
    riemann_chart,
    riemann_hemispheres,
)
from complexplorer.plotting.pyvista.plot_3d import pair_plot_landscape_pv, plot_landscape_pv
from complexplorer.plotting.pyvista.riemann import riemann_pv
from complexplorer.plotting.pyvista.riemann_surface import riemann_surface_pv

# Utility functions

__all__ = [
    # Version
    "__version__",
    # Exceptions
    "ComplexplorerError",
    "ValidationError",
    "ColormapError",
    # Core classes
    "Domain",
    "Rectangle",
    "Disk",
    "Annulus",
    "CompositeDomain",
    "Colormap",
    "Phase",
    "Chessboard",
    "PolarChessboard",
    "LogRings",
    "OklabPhase",
    "PerceptualPastel",
    "AnalogousWedge",
    "DivergingWarmCool",
    "Isoluminant",
    "CubehelixPhase",
    "InkPaper",
    "EarthTopographic",
    "FourQuadrant",
    "ModulusScaling",
    "get_scaling_preset",
    # Function preset registry: catalog supplies functions, PlotPresets configures renders
    "catalog",
    "FunctionPreset",
    # Gallery generator
    "generate_gallery",
    # Core functions
    "phase",
    "sawtooth",
    "stereographic_projection",
    "inverse_stereographic",
    # Plotting functions
    "plot",
    "pair_plot",
    "riemann_chart",
    "riemann_hemispheres",
    # Utilities
    # High-level API
    "quick_plot",
    "PlotPresets",
    # Engineering mode (namespaced subpackage)
    "ee",
    # STL export (PyVista-backed)
    "OrnamentGenerator",
    "create_ornament",
    # PyVista 3D plotting
    "plot_landscape_pv",
    "pair_plot_landscape_pv",
    "riemann_pv",
    "riemann_surface_pv",
]
