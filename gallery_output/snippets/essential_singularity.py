import complexplorer as cp
import numpy as np

# Essential singularity at origin
f = lambda z: np.exp(1/z)
domain = cp.Annulus(0.05, 0.5)

# High resolution needed near singularity
cp.plot(domain, f, 
        cmap=cp.Phase(n_phi=24),
        resolution=600)