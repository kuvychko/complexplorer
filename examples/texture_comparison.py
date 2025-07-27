#!/usr/bin/env python3
"""Side-by-side comparison of smooth vs textured Riemann spheres."""

import numpy as np
import complexplorer as cp

# Define function and colormap for comparison
func = lambda z: (z**2 - 1) / (z**2 + 1)
cmap = cp.Chessboard(spacing=0.5)

print("Texture Comparison Demo")
print("=" * 40)
print("\nThis demo shows the same function with and without texture.")
print("Two windows will open sequentially for comparison.\n")

# First window: Smooth surface
print("1. Opening smooth surface (visual only)...")
cp.riemann_pv(
    func,
    resolution=200,
    cmap=cmap,
    texture_height=0.0,  # No texture
    title="Smooth Surface - Chessboard Pattern",
    interactive=True,
    window_size=(800, 800),
    camera_position=(2.5, 2.5, 2.5)
)

# Second window: With texture
print("\n2. Opening textured surface...")
print("   Notice the raised white squares and lowered black squares.")
cp.riemann_pv(
    func,
    resolution=200,
    cmap=cmap,
    texture_height=0.005,  # 0.5% of radius
    texture_mode='binary',
    texture_preview_scale=10.0,
    title="With Binary Texture - Chessboard Pattern",
    interactive=True,
    window_size=(800, 800),
    camera_position=(2.5, 2.5, 2.5)
)

print("\n" + "=" * 40)
print("Comparison Notes:")
print("- Both show the same function and colormap")
print("- Texture adds physical height to the color pattern")
print("- White squares are raised, black squares are lowered")
print("- Preview scale exaggerates height for visibility")