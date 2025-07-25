#!/usr/bin/env python3
"""
Showcase of modulus scaling modes for 3D landscape plots.

This example demonstrates how different modulus scaling modes affect
the height representation in 3D visualizations of complex functions.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import complexplorer as cp

# Function with interesting features: poles, zeros, and essential singularity
def f(z):
    """Complex function with diverse behavior."""
    return (z**3 - 1) / (z**2 + 0.5)

# Create domain
domain = cp.Rectangle(4, 4)
cmap = cp.Phase(n_phi=6, auto_scale_r=True)

# Different modulus scaling modes to showcase
modes = [
    ('none', 'Direct modulus (default)'),
    ('constant', 'Constant height (phase only)'),
    ('arctan', 'Smooth bounded scaling'),
    ('logarithmic', 'Logarithmic scaling'),
    ('linear_clamp', 'Linear with clamping'),
    ('adaptive', 'Percentile-based adaptive')
]

# Create figure with subplots
fig = plt.figure(figsize=(15, 10))

for i, (mode, description) in enumerate(modes):
    ax = fig.add_subplot(2, 3, i + 1, projection='3d')
    
    # Plot with specific modulus mode
    cp.plot_landscape(
        domain=domain,
        func=f,
        n=80,
        ax=ax,
        cmap=cmap,
        modulus_mode=mode,
        antialiased=True
    )
    
    ax.set_title(f'{description}\n(mode="{mode}")', fontsize=10)
    ax.view_init(elev=25, azim=45)
    
    # Adjust subplot spacing
    ax.dist = 11

plt.suptitle('Modulus Scaling Modes for 3D Landscape Plots', fontsize=14)
plt.tight_layout()
plt.savefig('modulus_scaling_modes.png', dpi=150, bbox_inches='tight')
print("Saved: modulus_scaling_modes.png")

# Example with custom scaling function
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), 
                                subplot_kw={'projection': '3d'})

# Standard arctan scaling
cp.plot_landscape(
    domain=domain,
    func=f,
    n=80,
    ax=ax1,
    cmap=cmap,
    modulus_mode='arctan'
)
ax1.set_title('Standard arctan scaling')
ax1.view_init(elev=25, azim=45)

# Custom scaling that emphasizes values near 1
def custom_scale(moduli):
    """Custom scaling that highlights unit circle."""
    # Emphasize values near |f(z)| = 1
    return 1 + 0.3 * np.tanh(5 * (moduli - 1))

cp.plot_landscape(
    domain=domain,
    func=f,
    n=80,
    ax=ax2,
    cmap=cmap,
    modulus_mode='custom',
    modulus_params={'scaling_func': custom_scale}
)
ax2.set_title('Custom scaling (emphasize |f|=1)')
ax2.view_init(elev=25, azim=45)

plt.suptitle('Custom Modulus Scaling Example', fontsize=14)
plt.tight_layout()
plt.savefig('custom_modulus_scaling.png', dpi=150, bbox_inches='tight')
print("Saved: custom_modulus_scaling.png")

# Example with parameter variations
fig3, axes = plt.subplots(2, 2, figsize=(10, 10), 
                         subplot_kw={'projection': '3d'})
axes = axes.flatten()

# Different parameter settings for arctan mode
params = [
    {'r_min': 0.5, 'r_max': 1.5},  # Default
    {'r_min': 0.2, 'r_max': 1.0},  # Compressed
    {'r_min': 0.8, 'r_max': 2.0},  # Expanded
    {'r_min': 0.5, 'r_max': 3.0},  # Wide range
]

for i, param in enumerate(params):
    ax = axes[i]
    cp.plot_landscape(
        domain=domain,
        func=f,
        n=80,
        ax=ax,
        cmap=cmap,
        modulus_mode='arctan',
        modulus_params=param
    )
    
    ax.set_title(f"r_min={param['r_min']}, r_max={param['r_max']}", 
                 fontsize=10)
    ax.view_init(elev=25, azim=45)

plt.suptitle('Arctan Scaling with Different Parameters', fontsize=14)
plt.tight_layout()
plt.savefig('arctan_parameters.png', dpi=150, bbox_inches='tight')
print("Saved: arctan_parameters.png")

plt.close('all')
print("\nModulus scaling showcase complete!")
print("Generated 3 demonstration images showing various scaling modes.")