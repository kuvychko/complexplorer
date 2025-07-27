import complexplorer as cp

# Function with interesting modulus behavior
f = lambda z: (z - 1) * (z + 1) / (z**2 + 0.5)

# Riemann sphere with arctan modulus scaling
plotter = cp.riemann_pv(
    f,
    cmap=cp.Phase(n_phi=12, auto_scale_r=True),
    modulus_mode='arctan',
    scaling_params={'r_min': 0.3, 'r_max': 1.0},
    resolution=200,
    notebook=False,
    interactive=True  # For interactive viewing
)