"""Utility functions for complexplorer."""

import sys
import warnings


def setup_matplotlib_backend():
    """
    Set up the appropriate matplotlib backend for the current environment.
    
    - In Jupyter notebooks: Use default (inline)
    - In CLI scripts: Try to use Qt backend for interactive plots
    - Falls back gracefully if Qt is not available
    
    Returns
    -------
    str
        The backend that was set
    """
    # Check if we're in Jupyter/IPython
    try:
        __IPYTHON__  # This is defined in IPython environments
        return "inline"  # Jupyter handles its own backend
    except NameError:
        pass
    
    # We're in a regular Python script
    import matplotlib
    
    # If backend is already set (e.g., by user), don't change it
    current_backend = matplotlib.get_backend()
    if current_backend != 'agg':
        return current_backend
    
    # Try to use Qt backend for interactive plots
    try:
        import PyQt6  # noqa: F401
        # Check if we can actually use the Qt backend (not in headless environment)
        import os
        if os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'):
            matplotlib.use('Qt5Agg')
            return 'Qt5Agg'
        else:
            # Headless environment, stick with Agg
            return 'agg'
    except ImportError:
        # Qt not available, try TkAgg
        try:
            import tkinter  # noqa: F401
            matplotlib.use('TkAgg')
            return 'TkAgg'
        except ImportError:
            # No interactive backend available
            warnings.warn(
                "No interactive matplotlib backend available. "
                "Install PyQt6 with: pip install 'complexplorer[qt]' "
                "for interactive plots in CLI scripts.",
                UserWarning
            )
            return 'agg'


def ensure_interactive_plots():
    """
    Ensure matplotlib is set up for interactive plots if possible.
    
    This is a convenience function that can be called at the start
    of example scripts to ensure plots are interactive.
    """
    backend = setup_matplotlib_backend()
    
    if backend == 'agg':
        print("Warning: Using non-interactive backend. Plots will be static.")
        print("Install PyQt6 for interactive plots: pip install 'complexplorer[qt]'")
    
    return backend