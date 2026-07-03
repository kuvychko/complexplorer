"""Basic integration tests for core workflows."""

import os
import tempfile

import numpy as np

from complexplorer import plot_landscape_pv, riemann_pv
from complexplorer.api import quick_plot
from complexplorer.core.colormap import Phase

# Import from new API
from complexplorer.core.domain import Annulus, Disk, Rectangle
from complexplorer.plotting.matplotlib.plot_2d import plot


class TestCoreWorkflows:
    """Test essential workflows."""

    def test_basic_2d_plot(self):
        """Test basic 2D plotting workflow."""
        # Standard function
        func = lambda z: z**2
        domain = Rectangle(2, 2)

        # Basic plot
        ax = plot(domain, func, resolution=30)
        assert ax is not None

        import matplotlib.pyplot as plt

        plt.close("all")

    def test_enhanced_phase_portrait(self):
        """Test enhanced phase portrait."""
        func = lambda z: (z - 1) * (z + 1)
        domain = Rectangle(3, 3)
        cmap = Phase(n_phi=6, auto_scale_r=True)

        ax = plot(domain, func, cmap=cmap, resolution=40)
        assert ax is not None

        import matplotlib.pyplot as plt

        plt.close("all")

    def test_3d_landscape(self):
        """Test 3D landscape plot (PyVista)."""
        func = lambda z: z**3 - z
        domain = Rectangle(2, 2)

        plotter = plot_landscape_pv(
            domain, func, resolution=30, interactive=False, return_plotter=True
        )
        assert plotter is not None
        plotter.close()

    def test_riemann_sphere(self):
        """Test Riemann sphere visualization (PyVista)."""
        func = lambda z: (z**2 - 1) / (z**2 + 1)

        plotter = riemann_pv(func, resolution=25, interactive=False, return_plotter=True)
        assert plotter is not None
        plotter.close()

    def test_high_level_api(self):
        """Test high-level API functions."""
        func = lambda z: np.exp(z) / (z + 1)

        # Quick plot
        ax = quick_plot(func, mode="2d", resolution=30)
        assert ax is not None

        import matplotlib.pyplot as plt

        plt.close("all")


class TestDomainTypes:
    """Test different domain types."""

    def test_rectangle_domain(self):
        """Test with rectangular domain."""
        func = lambda z: z**2 - 1
        domain = Rectangle(3, 2)  # 3 wide, 2 tall

        ax = plot(domain, func, resolution=40)
        assert ax is not None

        import matplotlib.pyplot as plt

        plt.close("all")

    def test_disk_domain(self):
        """Test with disk domain."""
        func = lambda z: z**3
        domain = Disk(1.5)

        ax = plot(domain, func, resolution=40)
        assert ax is not None

        import matplotlib.pyplot as plt

        plt.close("all")

    def test_annulus_domain(self):
        """Test with annular domain."""
        func = lambda z: (z - 0.5) / (z + 0.5)
        domain = Annulus(0.3, 1.5)  # Avoid poles

        ax = plot(domain, func, resolution=40)
        assert ax is not None

        import matplotlib.pyplot as plt

        plt.close("all")


class TestFunctionTypes:
    """Test various function types."""

    def test_polynomial(self):
        """Test polynomial function."""
        func = lambda z: z**4 - 2 * z**2 + 1
        domain = Rectangle(2, 2)

        ax = plot(domain, func, resolution=40)
        assert ax is not None

        import matplotlib.pyplot as plt

        plt.close("all")

    def test_rational_with_pole(self):
        """Test rational function with pole."""
        func = lambda z: 1 / (z - 0.5)
        # Use domain that avoids the pole
        domain = Annulus(1, 2)

        ax = plot(domain, func, resolution=40)
        assert ax is not None

        import matplotlib.pyplot as plt

        plt.close("all")

    def test_transcendental(self):
        """Test transcendental function."""
        func = lambda z: np.sin(z) * np.exp(-np.abs(z) / 3)
        domain = Rectangle(4, 4)

        ax = plot(domain, func, resolution=50)
        assert ax is not None

        import matplotlib.pyplot as plt

        plt.close("all")


# STL export (PyVista-backed) is always available as of 3.0 (PyVista is a core dependency).
from complexplorer.export.stl import create_ornament


class TestSTLExport:
    """Test STL export functionality."""

    def test_stl_generation(self):
        """Test basic STL generation."""
        func = lambda z: z**3 - 1

        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
            filename = tmp.name

        try:
            saved = create_ornament(func, filename, size_mm=50, resolution=25, verbose=False)

            assert os.path.exists(saved)
            assert os.path.getsize(saved) > 1000  # Should have content

        finally:
            if os.path.exists(filename):
                os.unlink(filename)
