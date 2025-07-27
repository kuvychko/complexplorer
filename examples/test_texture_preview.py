#!/usr/bin/env python3
"""Test script for texture preview functionality."""

import numpy as np
import complexplorer as cp

# Test 1: Chessboard with binary texture
print("Test 1: Chessboard pattern with binary texture")
func = lambda z: z
cmap = cp.Chessboard(spacing=0.5)

try:
    cp.riemann_pv(
        func,
        resolution=50,
        cmap=cmap,
        texture_height=0.005,  # 0.5% of radius
        texture_mode='binary',
        texture_preview_scale=5.0,
        interactive=False,
        filename='test_texture_chessboard.png'
    )
    print("✓ Chessboard texture test passed")
except Exception as e:
    print(f"✗ Chessboard texture test failed: {e}")

# Test 2: Phase portrait with ridges
print("\nTest 2: Enhanced phase portrait with ridges")
func = lambda z: (z**2 - 1) / (z**2 + 1)
cmap = cp.Phase(n_phi=12, r_linear_step=1.0, v_base=0.4)

try:
    cp.riemann_pv(
        func,
        resolution=50,
        cmap=cmap,
        texture_height=0.008,  # 0.8% of radius
        texture_mode='ridges',
        texture_sharpness=0.9,
        texture_preview_scale=10.0,
        interactive=False,
        filename='test_texture_phase.png'
    )
    print("✓ Phase ridges test passed")
except Exception as e:
    print(f"✗ Phase ridges test failed: {e}")

# Test 3: With modulus scaling
print("\nTest 3: Texture with arctan modulus scaling")
func = lambda z: z**3 - 1
cmap = cp.PolarChessboard(n_phi=6, spacing=0.5)

try:
    cp.riemann_pv(
        func,
        resolution=50,
        cmap=cmap,
        modulus_mode='arctan',
        texture_height=0.006,  # 0.6% of radius
        texture_mode='grooves',
        texture_preview_scale=8.0,
        interactive=False,
        filename='test_texture_modulus.png'
    )
    print("✓ Modulus + texture test passed")
except Exception as e:
    print(f"✗ Modulus + texture test failed: {e}")

print("\nAll tests completed.")