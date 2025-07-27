"""Backend setup utilities for matplotlib and PyQt6."""

import sys
import os
import warnings


def setup_matplotlib_backend(force_qt: bool = False):
    """Set up matplotlib backend for interactive plots.
    
    Parameters
    ----------
    force_qt : bool
        Force PyQt6 backend even in non-interactive environments
        
    Returns
    -------
    str
        The backend that was set up
    """
    import matplotlib
    
    # Check if we're in a notebook environment
    if 'ipykernel' in sys.modules or 'IPython' in sys.modules:
        return matplotlib.get_backend()
    
    # Check if we're in a headless environment
    if not force_qt and (os.environ.get('DISPLAY') is None and sys.platform != 'win32'):
        matplotlib.use('Agg')
        return 'Agg'
    
    # Try to use Qt backend if available
    try:
        import PyQt6
        # Use qtagg which supports both Qt5 and Qt6
        matplotlib.use('qtagg')
        return 'qtagg'
    except ImportError:
        try:
            import PyQt5
            matplotlib.use('qt5agg')
            return 'qt5agg'
        except ImportError:
            # Fall back to default
            return matplotlib.get_backend()


def ensure_interactive_plots():
    """Ensure plots are shown in an interactive environment.
    
    This function sets up the backend and ensures plt.ion() is called
    for interactive plots when running scripts.
    """
    import matplotlib.pyplot as plt
    
    backend = setup_matplotlib_backend()
    
    # Enable interactive mode for supported backends
    if backend not in ['Agg', 'svg', 'pdf', 'ps']:
        plt.ion()
    
    return backend


__all__ = ['setup_matplotlib_backend', 'ensure_interactive_plots']