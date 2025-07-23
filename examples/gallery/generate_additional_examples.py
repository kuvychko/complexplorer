#!/usr/bin/env python
"""
Generate additional example images for complexplorer documentation.

This script generates supplementary example images that showcase specific features
not covered in the main gallery.

Usage:
    python generate_additional_examples.py
"""

import numpy as np
import complexplorer as cx
import matplotlib.pyplot as plt


def generate_basic_examples():
    """Generate basic usage examples."""
    # Basic plot with default settings
    domain = cx.Rectangle(3, 3)
    test_func = lambda z: (z - 1) / (z**2 + z + 1)
    
    plt.figure(figsize=(3, 3))
    cx.plot(domain, test_func)
    plt.savefig('basic_plot_default.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Basic plot with lower resolution
    plt.figure(figsize=(3, 3))
    cx.plot(domain, test_func, n=50)
    plt.savefig('basic_plot_low_res.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Generated basic plot examples")


def generate_domain_examples():
    """Generate examples with different domains."""
    test_func = lambda z: (z - 1) / (z**2 + z + 1)
    
    # Annulus domain example
    annulus = cx.Annulus(0.25, 3, center=-1+1j)
    plt.figure(figsize=(3, 3))
    cx.plot(annulus, test_func, n=400, 
            cmap=cx.Phase(6, 0.6, out_of_domain_hsv=(0.0, 0.1, 0.5)))
    plt.savefig('annulus_domain_example.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Generated domain examples")


def generate_comparison_plot():
    """Generate the analytic vs non-analytic comparison plot."""
    domain = cx.Rectangle(3, 3)
    test_func = lambda z: (z - 1) / (z**2 + z + 1)
    test_func2 = lambda z: (z - 1) / (np.conjugate(z)**2 + z + 1)
    
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 5))
    
    cx.plot(domain, lambda x: x, n=400,
            cmap=cx.PolarChessboard(6, r_log=0.7), title=r'$z$', ax=ax0)
    cx.plot(domain, test_func, n=400,
            cmap=cx.PolarChessboard(6, r_log=0.7), 
            title=r'$f(z) = \frac{z - 1}{z^2 + z + 1}$ - analytic (angle-preserving)', ax=ax1)
    cx.plot(domain, test_func2, n=400, 
            cmap=cx.PolarChessboard(6, r_log=0.7), 
            title=r'$f(z) = \frac{z - 1}{(z*)^2 + z + 1}$ - non-analytic', ax=ax2)
    
    fig.suptitle('Visual comparison of analytic and non-analytic maps using polar chessboard coloring scheme', 
                 fontsize=14)
    fig.savefig('analytic_vs_non_analytic_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Generated comparison plot")


def generate_3d_landscape_examples():
    """Generate additional 3D landscape examples."""
    domain = cx.Rectangle(3, 3)
    test_func = lambda z: (z - 1) / (z**2 + z + 1)
    
    # Basic 3D landscape
    fig, _ = cx.pair_plot_landscape(domain, test_func, figsize=(6, 3))
    fig.savefig('landscape_basic.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3D landscape with logarithmic z-axis
    fig, _ = cx.pair_plot_landscape(domain, test_func, figsize=(6, 3), 
                                    zaxis_log=True, z_max=100)
    fig.savefig('landscape_log_z_axis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Generated 3D landscape examples")


def generate_all_additional_examples(output_path='.'):
    """Generate all additional example images."""
    import os
    original_dir = os.getcwd()
    try:
        os.chdir(output_path)
        
        print("Generating additional complexplorer examples...")
        
        generate_basic_examples()
        generate_domain_examples()
        generate_comparison_plot()
        generate_3d_landscape_examples()
        
        print("\nAll additional examples generated successfully!")
        
    finally:
        os.chdir(original_dir)


if __name__ == "__main__":
    # When run as a script, generate images in the current directory
    generate_all_additional_examples()