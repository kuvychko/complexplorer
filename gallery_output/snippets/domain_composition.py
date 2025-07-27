import complexplorer as cp

# Create domain with holes
f = lambda z: (z**2 - 1) / (z**2 + 1)
rect = cp.Rectangle(4, 4)
hole1 = cp.Disk(0.5, center=1)
hole2 = cp.Disk(0.5, center=-1)
domain = rect - hole1 - hole2

# Visualize with enhanced phase
cmap = cp.Phase(n_phi=12, auto_scale_r=True)
cp.plot(domain, f, cmap=cmap)