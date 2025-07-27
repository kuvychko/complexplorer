#!/usr/bin/env python3
"""Demonstrate workflow from texture preview to STL export.

This script shows how to:
1. Preview textures in riemann_pv
2. Use the same settings for STL export
"""

import numpy as np
import complexplorer as cp
from complexplorer.export.stl import create_ornament

# Define our function
def f(z):
    """Rational function with interesting phase portrait."""
    return (z**2 - 1) / (z**2 + 1)

print("Texture Preview to STL Workflow")
print("=" * 40)

# Step 1: Define colormap and texture settings
cmap = cp.Phase(n_phi=12, r_linear_step=1.0, v_base=0.4)
texture_height = 0.008  # 0.8% of radius
texture_mode = 'ridges'
texture_sharpness = 0.9

print("\nStep 1: Preview texture in interactive viewer")
print(f"Colormap: Enhanced phase portrait with {cmap.n_phi} sectors")
print(f"Texture: {texture_mode} mode, height={texture_height}")

# Preview with exaggerated texture for visibility
cp.riemann_pv(
    f,
    resolution=200,
    cmap=cmap,
    texture_height=texture_height,
    texture_mode=texture_mode,
    texture_sharpness=texture_sharpness,
    texture_preview_scale=10.0,  # 10x exaggeration for preview
    title="Texture Preview (10× scale)",
    interactive=True
)

# Step 2: Export to STL with the same settings
print("\nStep 2: Export to STL with actual texture scale")
print("Creating STL file with the same texture settings...")

# Use EXACTLY the same parameters for STL
create_ornament(
    f,
    filename="ornament_from_preview.stl",
    size_mm=50,
    resolution=200,  # Same resolution as preview
    cmap=cmap,  # Same colormap
    texture_height=texture_height,  # Same height (no preview scale)
    texture_mode=texture_mode,  # Same mode
    texture_sharpness=texture_sharpness,  # Same sharpness
    verbose=True
)

print("\n" + "=" * 40)
print("Workflow Complete!")
print("\nKey points:")
print("- Preview uses texture_preview_scale for visibility")
print("- STL export uses actual texture_height without scaling")
print("- All other parameters remain the same")
print("\nThe STL file now contains the physical texture that was")
print("previewed in the interactive viewer, ready for 3D printing!")