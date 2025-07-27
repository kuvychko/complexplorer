import complexplorer as cp

# Define function and domain
f = lambda z: z**3 - 1
domain = cp.Disk(2)

# Polar chessboard pattern
cmap = cp.PolarChessboard(n_phi=12, spacing=0.3)
cp.plot(domain, f, cmap=cmap)