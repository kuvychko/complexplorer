import complexplorer as cp

# Rational function with poles
f = lambda z: (z**2 - 1) / (z**2 + 1)

# Create Riemann sphere visualization
plotter = cp.riemann_pv(
    f,
    cmap=cp.Phase(n_phi=16),
    modulus_mode='constant',
    resolution=150,
    notebook=False,
    interactive=True  # For interactive viewing
)