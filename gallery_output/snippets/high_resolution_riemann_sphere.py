import complexplorer as cp

# Rational function with interesting features
f = lambda z: (z**2 - 1) / (z**2 + 1)

# High-resolution Riemann sphere for publication
plotter = cp.riemann_pv(
    f,
    cmap=cp.Phase(n_phi=12, auto_scale_r=True, v_base=0.4),
    modulus_mode='constant',
    resolution=800,
    window_size=(1200, 1200),
    camera_position=(2.5, 2.0, 1.5),
    show_orientation=True,
    notebook=False,
    interactive=True  # For interactive viewing
)