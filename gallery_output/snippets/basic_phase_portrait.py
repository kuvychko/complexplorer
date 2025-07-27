import complexplorer as cp

# Define function and domain
f = lambda z: (z - 1) / (z**2 + z + 1)
domain = cp.Rectangle(4, 4)

# Create basic phase portrait
cp.plot(domain, f, cmap=cp.Phase())