#!/usr/bin/env python3
"""Showcase different texture modes with various colormaps.

This script demonstrates how physical textures derived from colormaps
can create tactile features for 3D printing.
"""

import numpy as np
import complexplorer as cp

# Define test function
def f(z):
    """Rational function with interesting features."""
    return (z**2 - 1) / (z**2 + 1)

print("Complexplorer Texture Showcase")
print("=" * 40)

# Example 1: Binary texture with Chessboard
print("\n1. Chessboard with Binary Height Texture")
print("   White squares raised, black squares lowered")
cmap1 = cp.Chessboard(spacing=0.5)
cp.riemann_pv(
    f,
    resolution=80,
    cmap=cmap1,
    texture_height=0.5,
    texture_mode='binary',
    texture_preview_scale=10.0,
    title="Chessboard Binary Texture",
    interactive=True,
    show_grid=True
)

# Example 2: Ridges with Enhanced Phase Portrait
print("\n2. Enhanced Phase Portrait with Ridge Texture")
print("   Ridges appear at phase sector boundaries")
cmap2 = cp.Phase(n_phi=12, r_linear_step=1.0, v_base=0.4)
cp.riemann_pv(
    f,
    resolution=100,
    cmap=cmap2,
    texture_height=0.8,
    texture_mode='ridges',
    texture_sharpness=0.9,
    texture_preview_scale=8.0,
    title="Phase Portrait with Ridges",
    interactive=True
)

# Example 3: Grooves with Polar Chessboard
print("\n3. Polar Chessboard with Groove Texture")
print("   Grooves at polar grid boundaries")
cmap3 = cp.PolarChessboard(n_phi=8, spacing=0.3)
cp.riemann_pv(
    f,
    resolution=80,
    cmap=cmap3,
    texture_height=0.6,
    texture_mode='grooves',
    texture_preview_scale=10.0,
    title="Polar Chessboard with Grooves",
    interactive=True
)

# Example 4: Ridges with Logarithmic Rings
print("\n4. Logarithmic Rings with Ridge Texture")
print("   Circular ridges at logarithmic intervals")
cmap4 = cp.LogRings(log_spacing=0.2)
cp.riemann_pv(
    lambda z: z**3 - 1,  # Different function to show poles
    resolution=100,
    cmap=cmap4,
    texture_height=0.7,
    texture_mode='ridges',
    texture_preview_scale=8.0,
    title="Log Rings with Ridges",
    interactive=True
)

# Example 5: Combined with Modulus Scaling
print("\n5. Texture with Arctan Modulus Scaling")
print("   Shows texture remains uniform after scaling")
cmap5 = cp.Phase(n_phi=6, auto_scale_r=True)
cp.riemann_pv(
    lambda z: (z - 1) / (z**2 + z + 1),
    resolution=100,
    cmap=cmap5,
    modulus_mode='arctan',
    modulus_params={'scale': 2.0},
    texture_height=1.0,
    texture_mode='ridges',
    texture_sharpness=0.95,
    texture_preview_scale=5.0,
    title="Texture + Modulus Scaling",
    interactive=True
)

# Example 6: Side-by-side comparison
print("\n6. Side-by-side: Smooth vs Textured")
print("   Compare visual colormap with physical texture")

# Create a more complex function
def g(z):
    """Function with essential singularity."""
    return np.exp(1/z) if np.isscalar(z) else np.where(z != 0, np.exp(1/z), 0)

# Use PyVista's multi-subplot feature
import pyvista as pv
plotter = pv.Plotter(shape=(1, 2), window_size=(1200, 600))

# Left: Smooth surface
plotter.subplot(0, 0)
cp.riemann_pv(
    g,
    resolution=80,
    cmap=cmap2,
    domain=cp.Disk(2),
    texture_height=0.0,  # No texture
    title="Smooth (Visual Only)",
    interactive=False,
    return_plotter=False
)

# Right: With texture
plotter.subplot(0, 1)
cp.riemann_pv(
    g,
    resolution=80,
    cmap=cmap2,
    domain=cp.Disk(2),
    texture_height=0.8,
    texture_mode='ridges',
    texture_preview_scale=10.0,
    title="With Ridge Texture",
    interactive=False,
    return_plotter=False
)

plotter.link_views()
plotter.show()

print("\n" + "=" * 40)
print("Texture Preview Notes:")
print("- Preview scale is exaggerated for visibility")
print("- Actual STL export will use the specified height")
print("- Texture follows colormap boundaries exactly")
print("- Height remains uniform regardless of modulus scaling")