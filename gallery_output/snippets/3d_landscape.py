import complexplorer as cp

# Define function and domain
f = lambda z: (z - 1) / (z**2 + z + 1)
domain = cp.Rectangle(4, 4)

# Create 3D landscape with PyVista
# Use notebook=False for high-quality window
plotter = cp.plot_landscape_pv(
    domain, f,
    cmap=cp.Phase(n_phi=12, auto_scale_r=True),
    resolution=200,
    z_scale=0.4,
    notebook=False,
    interactive=True  # For interactive viewing
)