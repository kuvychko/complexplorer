#!/usr/bin/env python
"""
Interactive showcase of all new colormaps in Complexplorer v2.0.

This script demonstrates the 8 new perceptually-optimized colormap families,
allowing interactive comparison and parameter exploration.

Usage:
    python new_colormaps_showcase.py [--save]
    
    --save: Save images to gallery folder
"""

import numpy as np
import matplotlib.pyplot as plt
import complexplorer as cp
from pathlib import Path
import sys

# Check for PyVista
try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    print("Note: Install PyVista for 3D visualizations: pip install 'complexplorer[pyvista]'")


def get_test_functions():
    """Collection of interesting test functions."""
    return {
        'Rational': lambda z: (z - 1) / (z**2 + z + 1),
        'Polynomial': lambda z: z**4 - 1,
        'Essential Singularity': lambda z: np.exp(1/z) if np.abs(z) > 0.01 else 0,
        'Trigonometric': lambda z: np.sin(z) * np.cos(z),
        'Logarithmic': lambda z: np.log(z + 1),
        'Gamma-like': lambda z: z / (z**2 + 1) * np.exp(-np.abs(z)/3),
    }


def showcase_perceptual_colormaps():
    """Demonstrate perceptually uniform colormaps."""
    print("\n" + "="*60)
    print("PERCEPTUALLY UNIFORM COLORMAPS")
    print("="*60)
    
    funcs = get_test_functions()
    f = funcs['Rational']
    domain = cp.Rectangle(3, 3)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Perceptually Uniform Colormaps (v2.0)", fontsize=16)
    
    # PerceptualPastel
    cmap1 = cp.PerceptualPastel(n_phi=6, auto_scale_r=True)
    cp.plot(domain, f, cmap=cmap1, ax=axes[0,0], resolution=500)
    axes[0,0].set_title("PerceptualPastel\n(OkLCh, uniform brightness)")
    
    # Isoluminant  
    cmap2 = cp.Isoluminant(L=0.6, C=0.15, show_contours=True)
    cp.plot(domain, f, cmap=cmap2, ax=axes[0,1], resolution=500)
    axes[0,1].set_title("Isoluminant\n(Constant brightness)")
    
    # CubehelixPhase
    cmap3 = cp.CubehelixPhase(n_phi=6, auto_scale_r=True)
    cp.plot(domain, f, cmap=cmap3, ax=axes[0,2], resolution=500)
    axes[0,2].set_title("CubehelixPhase\n(Grayscale-optimal)")
    
    # Compare with traditional Phase
    cmap4 = cp.Phase(n_phi=6, auto_scale_r=True)
    cp.plot(domain, f, cmap=cmap4, ax=axes[1,0], resolution=500)
    axes[1,0].set_title("Traditional Phase\n(For comparison)")
    
    # PerceptualPastel with different parameters
    cmap5 = cp.PerceptualPastel(L_center=0.7, L_range=0.2, C=0.08)
    cp.plot(domain, f, cmap=cmap5, ax=axes[1,1], resolution=500)
    axes[1,1].set_title("PerceptualPastel\n(Lighter variant)")
    
    # CubehelixPhase with custom rotation
    cmap6 = cp.CubehelixPhase(start=0.3, rotations=2.0)
    cp.plot(domain, f, cmap=cmap6, ax=axes[1,2], resolution=500)
    axes[1,2].set_title("CubehelixPhase\n(2 rotations)")
    
    plt.tight_layout()
    return fig


def showcase_artistic_colormaps():
    """Demonstrate artistic and thematic colormaps."""
    print("\n" + "="*60)
    print("ARTISTIC & THEMATIC COLORMAPS")
    print("="*60)
    
    funcs = get_test_functions()
    f = funcs['Rational']
    domain = cp.Rectangle(3, 3)
    
    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Artistic & Thematic Colormaps (v2.0)", fontsize=16)
    
    # AnalogousWedge - Ocean
    cmap1 = cp.AnalogousWedge(H_center=0.55, H_wedge=0.2, n_phi=6, auto_scale_r=True)
    cp.plot(domain, f, cmap=cmap1, ax=axes[0,0], resolution=500)
    axes[0,0].set_title("AnalogousWedge\n(Ocean theme)")
    
    # AnalogousWedge - Sunset
    cmap2 = cp.AnalogousWedge(H_center=0.08, H_wedge=0.25, n_phi=6, auto_scale_r=True)
    cp.plot(domain, f, cmap=cmap2, ax=axes[0,1], resolution=500)
    axes[0,1].set_title("AnalogousWedge\n(Sunset theme)")
    
    # DivergingWarmCool
    cmap3 = cp.DivergingWarmCool(n_phi=6, auto_scale_r=True)
    cp.plot(domain, f, cmap=cmap3, ax=axes[0,2], resolution=500)
    axes[0,2].set_title("DivergingWarmCool\n(Cartographic)")
    
    # InkPaper
    cmap4 = cp.InkPaper(phase_strength=0.05, n_phi=8)
    cp.plot(domain, f, cmap=cmap4, ax=axes[0,3], resolution=500)
    axes[0,3].set_title("InkPaper\n(Minimalist)")
    
    # EarthTopographic
    cmap5 = cp.EarthTopographic(n_phi=6, auto_scale_r=True)
    cp.plot(domain, f, cmap=cmap5, ax=axes[1,0], resolution=500)
    axes[1,0].set_title("EarthTopographic\n(Terrain-inspired)")
    
    # FourQuadrant
    cmap6 = cp.FourQuadrant(n_phi=6, auto_scale_r=True)
    cp.plot(domain, f, cmap=cmap6, ax=axes[1,1], resolution=500)
    axes[1,1].set_title("FourQuadrant\n(Geometric)")
    
    # AnalogousWedge - Violet
    cmap7 = cp.AnalogousWedge(H_center=0.75, H_wedge=0.2, S=0.4)
    cp.plot(domain, f, cmap=cmap7, ax=axes[1,2], resolution=500)
    axes[1,2].set_title("AnalogousWedge\n(Violet theme)")
    
    # InkPaper with stronger phase
    cmap8 = cp.InkPaper(phase_strength=0.1, contour_strength=0.15)
    cp.plot(domain, f, cmap=cmap8, ax=axes[1,3], resolution=500)
    axes[1,3].set_title("InkPaper\n(Enhanced contrast)")
    
    plt.tight_layout()
    return fig


def compare_all_colormaps():
    """Create a comprehensive comparison of all 13 colormaps."""
    print("\n" + "="*60)
    print("ALL COLORMAPS COMPARISON")
    print("="*60)
    
    f = lambda z: (z - 1) / (z**2 + z + 1)
    domain = cp.Rectangle(2.5, 2.5)
    
    # All 13 colormaps
    colormaps = [
        ('Phase', cp.Phase(n_phi=6, auto_scale_r=True)),
        ('Chessboard', cp.Chessboard(spacing=0.3)),
        ('PolarChessboard', cp.PolarChessboard(n_phi=8, spacing=0.3)),
        ('LogRings', cp.LogRings(log_spacing=0.3)),
        ('PerceptualPastel', cp.PerceptualPastel(n_phi=6, auto_scale_r=True)),
        ('AnalogousWedge', cp.AnalogousWedge(H_center=0.55, H_wedge=0.2)),
        ('DivergingWarmCool', cp.DivergingWarmCool(n_phi=6, auto_scale_r=True)),
        ('Isoluminant', cp.Isoluminant(L=0.6, C=0.15)),
        ('CubehelixPhase', cp.CubehelixPhase(n_phi=6, auto_scale_r=True)),
        ('InkPaper', cp.InkPaper(phase_strength=0.05)),
        ('EarthTopographic', cp.EarthTopographic(n_phi=6, auto_scale_r=True)),
        ('FourQuadrant', cp.FourQuadrant(n_phi=6, auto_scale_r=True)),
        ('Phase (basic)', cp.Phase()),  # Basic for comparison
    ]
    
    # Create figure
    n_cols = 4
    n_rows = (len(colormaps) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 12))
    fig.suptitle("Complete Colormap Collection - Complexplorer v2.0", fontsize=16)
    
    # Flatten axes for easier iteration
    axes_flat = axes.flatten()
    
    for idx, (name, cmap) in enumerate(colormaps):
        if idx < len(axes_flat):
            cp.plot(domain, f, cmap=cmap, ax=axes_flat[idx], resolution=400)
            axes_flat[idx].set_title(name, fontsize=10)
    
    # Hide extra axes
    for idx in range(len(colormaps), len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    plt.tight_layout()
    return fig


def showcase_3d_with_new_colormaps():
    """Show 3D landscapes with new colormaps (if PyVista available)."""
    if not HAS_PYVISTA:
        print("\nSkipping 3D showcase (PyVista not installed)")
        return None
        
    print("\n" + "="*60)
    print("3D LANDSCAPES WITH NEW COLORMAPS")
    print("="*60)
    print("Opening interactive 3D windows...")
    
    f = lambda z: (z - 1) / (z**2 + z + 1)
    domain = cp.Disk(2)
    
    # Show a few interesting colormaps in 3D
    colormaps = [
        cp.PerceptualPastel(n_phi=6, auto_scale_r=True),
        cp.EarthTopographic(n_phi=6, auto_scale_r=True),
        cp.DivergingWarmCool(n_phi=6, auto_scale_r=True),
    ]
    
    for i, cmap in enumerate(colormaps):
        print(f"  Opening {cmap.__class__.__name__} in 3D...")
        cp.plot_landscape_pv(
            domain, f, cmap=cmap,
            resolution=600,
            z_max=5,
            notebook=False,
            show=True,
            window_size=(800, 600),
            title=f"{cmap.__class__.__name__} - 3D Landscape"
        )


def interactive_parameter_exploration():
    """Interactive exploration of colormap parameters."""
    print("\n" + "="*60)
    print("PARAMETER EXPLORATION")
    print("="*60)
    
    f = lambda z: z**3 - z
    domain = cp.Rectangle(2, 2)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Parameter Effects on PerceptualPastel", fontsize=16)
    
    # Vary L_center
    for i, L in enumerate([0.4, 0.55, 0.7]):
        cmap = cp.PerceptualPastel(L_center=L, n_phi=6, auto_scale_r=True)
        cp.plot(domain, f, cmap=cmap, ax=axes[0,i], resolution=400)
        axes[0,i].set_title(f"L_center = {L}")
    
    # Vary Chroma
    for i, C in enumerate([0.05, 0.10, 0.15]):
        cmap = cp.PerceptualPastel(C=C, n_phi=6, auto_scale_r=True)
        cp.plot(domain, f, cmap=cmap, ax=axes[1,i], resolution=400)
        axes[1,i].set_title(f"Chroma = {C}")
    
    plt.tight_layout()
    return fig


def save_gallery_images(save_path='examples/gallery/v2_colormaps'):
    """Save example images for documentation."""
    print(f"\nSaving gallery images to {save_path}/...")
    
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # Test function
    f = lambda z: (z - 1) / (z**2 + z + 1)
    domain = cp.Rectangle(3, 3)
    
    # List of colormaps to save
    colormaps = [
        ('perceptual_pastel', cp.PerceptualPastel(n_phi=6, auto_scale_r=True)),
        ('analogous_wedge_ocean', cp.AnalogousWedge(H_center=0.55, H_wedge=0.2)),
        ('analogous_wedge_sunset', cp.AnalogousWedge(H_center=0.08, H_wedge=0.25)),
        ('diverging_warm_cool', cp.DivergingWarmCool(n_phi=6, auto_scale_r=True)),
        ('isoluminant', cp.Isoluminant(L=0.6, C=0.15)),
        ('cubehelix_phase', cp.CubehelixPhase(n_phi=6, auto_scale_r=True)),
        ('ink_paper', cp.InkPaper(phase_strength=0.05)),
        ('earth_topographic', cp.EarthTopographic(n_phi=6, auto_scale_r=True)),
        ('four_quadrant', cp.FourQuadrant(n_phi=6, auto_scale_r=True)),
    ]
    
    for name, cmap in colormaps:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        # Domain
        cp.plot(domain, lambda z: z, cmap=cmap, ax=ax1, resolution=500)
        ax1.set_title("Domain")
        
        # Codomain
        cp.plot(domain, f, cmap=cmap, ax=ax2, resolution=500)
        ax2.set_title("f(z) = (z-1)/(z²+z+1)")
        
        fig.suptitle(f"{cmap.__class__.__name__}", fontsize=14)
        plt.tight_layout()
        
        filename = Path(save_path) / f"{name}_2d.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  Saved: {filename}")
        plt.close()


def main():
    """Main showcase execution."""
    print("="*60)
    print("COMPLEXPLORER v2.0 - NEW COLORMAPS SHOWCASE")
    print("="*60)
    
    # Check command line arguments
    save_images = '--save' in sys.argv
    
    # Run showcases
    fig1 = showcase_perceptual_colormaps()
    fig2 = showcase_artistic_colormaps()
    fig3 = compare_all_colormaps()
    fig4 = interactive_parameter_exploration()
    
    # 3D showcase if available
    if HAS_PYVISTA and '--no3d' not in sys.argv:
        showcase_3d_with_new_colormaps()
    
    # Save images if requested
    if save_images:
        save_gallery_images()
    
    # Show all figures
    plt.show()
    
    print("\n" + "="*60)
    print("Showcase complete!")
    print("\nTips:")
    print("- Use 'auto_scale_r=True' for automatic square cells")
    print("- PerceptualPastel is best for print materials")
    print("- CubehelixPhase ensures grayscale readability")
    print("- AnalogousWedge creates sophisticated themed palettes")
    print("- See docs/colormap_guide.md for detailed documentation")
    print("="*60)


if __name__ == "__main__":
    main()