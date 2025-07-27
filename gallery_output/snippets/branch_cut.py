import complexplorer as cp
import numpy as np

# Complex logarithm with branch cut
f = lambda z: np.log(z)
domain = cp.Annulus(0.1, 3)

# Phase portrait shows branch cut clearly
cp.plot(domain, f, 
        cmap=cp.Phase(n_phi=12),
        resolution=400)