import complexplorer as cp
import numpy as np

# Exponential function
f = lambda z: np.exp(z)
domain = cp.Rectangle(4, 4)

# Logarithmic rings show modulus levels
cmap = cp.LogRings(log_spacing=0.3)
cp.plot(domain, f, cmap=cmap)