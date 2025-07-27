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
    resolution=500,
    cmap=cmap1,
    texture_height=0.005,  # 0.5% of radius
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
    resolution=500,
    cmap=cmap2,
    texture_height=0.008,  # 0.8% of radius
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
    resolution=500,
    cmap=cmap3,
    texture_height=0.006,  # 0.6% of radius
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
    resolution=500,
    cmap=cmap4,
    texture_height=0.007,  # 0.7% of radius
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
    resolution=500,
    cmap=cmap5,
    modulus_mode='arctan',
    modulus_params={'r_min': 0.3, 'r_max': 2.0},  # Correct parameters for arctan
    texture_height=0.01,  # 1% of radius
    texture_mode='ridges',
    texture_sharpness=0.95,
    texture_preview_scale=5.0,
    title="Texture + Modulus Scaling",
    interactive=True
)

# Example 6: Save comparison images
print("\n6. Generating comparison images...")
print("   Saving smooth vs textured versions for comparison")

# Create a simple function for comparison
func_compare = lambda z: (z**2 - 1) / (z**2 + 1)
cmap_compare = cp.Chessboard(spacing=0.5)

# Generate smooth version
cp.riemann_pv(
    func_compare,
    resolution=150,
    cmap=cmap_compare,
    texture_height=0.0,  # No texture
    title="Smooth (Visual Only)",
    interactive=False,
    filename="showcase_smooth.png"
)

# Generate textured version
cp.riemann_pv(
    func_compare,
    resolution=150,
    cmap=cmap_compare,
    texture_height=0.005,  # 0.5% of radius
    texture_mode='binary',
    texture_preview_scale=10.0,
    title="With Binary Texture",
    interactive=False,
    filename="showcase_textured.png"
)

print("   ✓ Saved showcase_smooth.png and showcase_textured.png")
print("   (Run texture_comparison.py for interactive side-by-side view)")

print("\n" + "=" * 40)
print("Texture Preview Notes:")
print("- Preview scale is exaggerated for visibility")
print("- Actual STL export will use the specified height")
print("- Texture follows colormap boundaries exactly")
print("- Height remains uniform regardless of modulus scaling")