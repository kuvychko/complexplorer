import complexplorer as cp

# Define function and domain
f = lambda z: z**3 - 1
domain = cp.Disk(2)

# Create side-by-side 3D landscapes
plotter = cp.pair_plot_landscape_pv(
    domain, f,
    cmap=cp.Phase(n_phi=12, auto_scale_r=True),
    resolution=200,
    z_scale=0.3,
    notebook=False,
    interactive=True  # For interactive viewing,
    window_size=(1600, 800)
)