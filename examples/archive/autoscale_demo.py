#!/usr/bin/env python3
"""
Demonstration of auto-scaled Enhanced Phase portraits.

Shows how auto_scale_r=True creates visually square cells automatically.
"""

import numpy as np
import complexplorer as cp
cp.ensure_interactive_plots()  # Set up interactive backend if available
import matplotlib.pyplot as plt


def demo_autoscaling(save_plots=True):
    """Compare manual vs auto-scaled enhanced phase portraits."""
    
    # Create domain and function
    domain = cp.Rectangle(4, 4)
    func = lambda z: (z**2 - 1) / (z**2 + 1)
    
    # Create figure with 3x2 subplots
    fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    fig.suptitle("Manual vs Auto-scaled Enhanced Phase Portraits", fontsize=16)
    
    # Test different n_phi values
    n_phi_values = [6, 12, 24]
    
    for i, n_phi in enumerate(n_phi_values):
        # Manual calculation (old way)
        # For square cells: r_linear_step = 2π/n_phi
        manual_r_step = 2 * np.pi / n_phi
        cmap_manual = cp.Phase(n_phi=n_phi, r_linear_step=manual_r_step, v_base=0.4)
        
        # Auto-scaled (new way)
        cmap_auto = cp.Phase(n_phi=n_phi, auto_scale_r=True, v_base=0.4)
        
        # Plot manual
        ax_manual = axes[i, 0]
        cp.plot(domain, func, cmap=cmap_manual, ax=ax_manual)
        ax_manual.set_title(f"Manual: n_phi={n_phi}, r_step={manual_r_step:.3f}")
        
        # Plot auto-scaled
        ax_auto = axes[i, 1]
        cp.plot(domain, func, cmap=cmap_auto, ax=ax_auto)
        ax_auto.set_title(f"Auto-scaled: n_phi={n_phi}")
        
        # Verify they're identical
        print(f"n_phi={n_phi}: manual r_step={manual_r_step:.6f}, "
              f"auto r_step={cmap_auto.r_linear_step:.6f}, "
              f"equal={np.isclose(manual_r_step, cmap_auto.r_linear_step)}")
    
    plt.tight_layout()
    if save_plots:
        plt.savefig('autoscale_manual_vs_auto.png', dpi=150, bbox_inches='tight')
        print("Saved: autoscale_manual_vs_auto.png")
    try:
        plt.show()
    except:
        pass  # Ignore display errors in headless environments


def demo_scale_radius(save_plots=True):
    """Show effect of different scale_radius values."""
    
    domain = cp.Rectangle(4, 4)
    func = lambda z: z**3 - 1
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle("Effect of scale_radius Parameter", fontsize=16)
    axes = axes.flatten()
    
    # Different scale_radius values
    scale_values = [0.5, 1.0, 1.5, 2.0]
    
    for i, scale_r in enumerate(scale_values):
        cmap = cp.Phase(n_phi=12, auto_scale_r=True, scale_radius=scale_r, v_base=0.4)
        cp.plot(domain, func, cmap=cmap, ax=axes[i])
        axes[i].set_title(f"scale_radius={scale_r} (r_step={cmap.r_linear_step:.3f})")
    
    plt.tight_layout()
    if save_plots:
        plt.savefig('autoscale_scale_radius_effect.png', dpi=150, bbox_inches='tight')
        print("Saved: autoscale_scale_radius_effect.png")
    try:
        plt.show()
    except:
        pass


def demo_comparison(save_plots=True):
    """Compare different enhancement methods."""
    
    domain = cp.Disk(2)
    func = lambda z: (z - 0.5) * (z + 0.5) * (z - 0.5j) * (z + 0.5j)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Comparison of Phase Portrait Types", fontsize=16)
    
    # 1. Basic phase
    cmap1 = cp.Phase()
    cp.plot(domain, func, cmap=cmap1, ax=axes[0, 0])
    axes[0, 0].set_title("Basic Phase")
    
    # 2. Phase with sectors (no modulus)
    cmap2 = cp.Phase(n_phi=12)
    cp.plot(domain, func, cmap=cmap2, ax=axes[0, 1])
    axes[0, 1].set_title("Phase with 12 sectors")
    
    # 3. Manual enhanced (rectangular cells)
    cmap3 = cp.Phase(n_phi=12, r_linear_step=0.3, v_base=0.4)
    cp.plot(domain, func, cmap=cmap3, ax=axes[0, 2])
    axes[0, 2].set_title("Manual Enhanced (rectangular)")
    
    # 4. Auto-scaled enhanced (square cells)
    cmap4 = cp.Phase(n_phi=12, auto_scale_r=True, v_base=0.4)
    cp.plot(domain, func, cmap=cmap4, ax=axes[1, 0])
    axes[1, 0].set_title("Auto-scaled Enhanced (square)")
    
    # 5. Auto-scaled with larger cells
    cmap5 = cp.Phase(n_phi=12, auto_scale_r=True, scale_radius=1.5, v_base=0.4)
    cp.plot(domain, func, cmap=cmap5, ax=axes[1, 1])
    axes[1, 1].set_title("Auto-scaled (larger cells)")
    
    # 6. Auto-scaled with finer resolution
    cmap6 = cp.Phase(n_phi=24, auto_scale_r=True, v_base=0.4)
    cp.plot(domain, func, cmap=cmap6, ax=axes[1, 2])
    axes[1, 2].set_title("Auto-scaled (24 sectors)")
    
    plt.tight_layout()
    if save_plots:
        plt.savefig('autoscale_comparison_types.png', dpi=150, bbox_inches='tight')
        print("Saved: autoscale_comparison_types.png")
    try:
        plt.show()
    except:
        pass


if __name__ == "__main__":
    print("Auto-scaling Demo for Enhanced Phase Portraits")
    print("=" * 50)
    
    # In headless environments, just save plots
    save = True
    
    print("\n1. Showing manual vs auto-scaled (should be identical)")
    demo_autoscaling(save)
    
    print("\n2. Showing effect of scale_radius parameter")
    demo_scale_radius(save)
    
    print("\n3. Comparing different phase portrait types")
    demo_comparison(save)
    
    print("\nDone! Check the generated PNG files.")