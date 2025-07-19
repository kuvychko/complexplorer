#!/usr/bin/env python
"""
Generate gallery images for complexplorer documentation.

This script generates all the example images used in the documentation gallery.
Images are saved to the current directory when the script is run.

Usage:
    python generate_gallery_images.py
"""

import numpy as np
import complexplorer as cx
import matplotlib.pyplot as plt
import re
import os.path


def make_pathname(path, name, suffix):
    """Helper function to prepare a nice filename."""
    name = re.sub(r"[()-]", "", name)
    name = "_".join(name.split())  # remove any multiple spaces, then join using underscore
    name = name + suffix
    return os.path.join(path, name)


def generate_2d_cmaps(domain, test_func, cmaps_dict, path='.'):
    """Generate 2D colormap examples."""
    for name, cmap in cmaps_dict.items():
        pathname = make_pathname(path, name, '_2d.png')
        print(f"Generating {pathname}...")
        cx.pair_plot(domain, test_func, figsize=(6, 3), title=name, cmap=cmap, filename=pathname)


def generate_3d_cmaps(domain, test_func, cmaps_dict, path='.'):
    """Generate 3D landscape colormap examples."""
    for name, cmap in cmaps_dict.items():
        pathname = make_pathname(path, name, '_3d.png')
        print(f"Generating {pathname}...")
        cx.pair_plot_landscape(domain, test_func, figsize=(6, 3), title=name, cmap=cmap, z_max=10, filename=pathname)


def generate_riemann_charts(test_func, path='.'):
    """Generate Riemann chart examples."""
    # 2D Riemann hemispheres
    pathname = os.path.join(path, 'riemann_chart_2d.png')
    print(f"Generating {pathname}...")
    cx.riemann_hemispheres(test_func, filename=pathname)
    
    # 3D Riemann sphere
    pathname = os.path.join(path, 'riemann_sphere_3d.png')
    print(f"Generating {pathname}...")
    cx.riemann(
        test_func, n=600, project_from_north=True,
        cmap=cx.Phase(6, 1, r_log_base=np.e),
        filename=pathname
    )


def generate_all_gallery_images(output_path='.'):
    """Generate all gallery images."""
    print("Generating complexplorer gallery images...")
    
    # Set default figure size
    plt.rcParams["figure.figsize"] = (3, 3)
    
    # Define domain and test function
    domain = cx.Rectangle(3, 3)
    test_func = lambda z: (z - 1) / (z**2 + z + 1)
    
    # Define all color maps to showcase
    cmaps_dict = {
        'Phase portrait': cx.Phase(),
        'Phase portrait (phase enhanced)': cx.Phase(6),
        'Phase portrait (modulus enhanced)': cx.Phase(None, 0.6),
        'Enhanced phase portrait (phase and modulus enhanced)': cx.Phase(6, 0.6),
        'Cartesian chessboard': cx.Chessboard(spacing=0.25),
        'Polar chessboard (linear modulus spacing)': cx.PolarChessboard(6, 0.25),
        'Polar chessboard (log modulus spacing)': cx.PolarChessboard(6, r_log=np.e),
        'Logarithmic rings': cx.LogRings(log_spacing=0.2),
    }
    
    # Generate 2D examples
    print("\n2D colormap examples:")
    generate_2d_cmaps(domain, test_func, cmaps_dict, output_path)
    
    # Generate 3D examples
    print("\n3D landscape examples:")
    generate_3d_cmaps(domain, test_func, cmaps_dict, output_path)
    
    # Generate Riemann examples
    print("\nRiemann sphere examples:")
    generate_riemann_charts(test_func, output_path)
    
    print("\nAll gallery images generated successfully!")


if __name__ == "__main__":
    # When run as a script, generate images in the current directory
    generate_all_gallery_images()