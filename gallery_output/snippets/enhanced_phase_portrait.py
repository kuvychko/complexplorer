import complexplorer as cp

# Define function and domain
f = lambda z: (z - 1) / (z**2 + z + 1)
domain = cp.Rectangle(4, 4)

# Enhanced phase with auto-scaled square cells
cmap = cp.Phase(n_phi=12, auto_scale_r=True, v_base=0.4)
cp.plot(domain, f, cmap=cmap)