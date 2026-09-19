---
name: Visual or rendering problem
about: An image, 3D view, or exported mesh looks wrong, or rendering fails
labels: bug, rendering
---

## What you see

<!-- Attach the image if you have one. Describe what you expected instead. -->

## Minimal example

```python
import complexplorer as cp

# The shortest code that produces it, including the domain, colormap and resolution.
```

## Environment

Rendering problems are rarely reproducible without all of this:

- complexplorer version:
- Python version:
- Operating system:
- PyVista and VTK versions: <!-- python -c "import pyvista; print(pyvista.__version__); import vtkmodules.all as v; print(v.vtkVersion.GetVTKVersion())" -->
- matplotlib version and backend: <!-- python -c "import matplotlib; print(matplotlib.__version__, matplotlib.get_backend())" -->
- Display situation: <!-- desktop with a GPU, remote/headless, CI, notebook, WSL -->
- If 3D: does the same call work with `interactive=False, filename="out.png"`?

## Anything else

<!-- Does it happen with a different colormap, resolution, or scaling mode? -->
