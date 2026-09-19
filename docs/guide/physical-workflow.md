# STL export and the physical workflow

The last step of the tour is an object you can hold. A Riemann relief — the modulus of a function
displacing the surface of a sphere — is already a closed 3D shape, which is what makes it
printable.

[![Riemann relief render, untextured STL mesh, and the printed ornament](../examples/gallery/view/_tour/physical_triptych.png)](../examples/gallery/_tour/physical_triptych.png)

The relief is the mathematics, the mesh is the geometry that survives losing the colour, and the
print is the object on a desk. Ten poles become ten spikes around the central zero.

## One call

```python
import complexplorer as cp

cp.create_ornament(lambda z: z / (z**10 - 1), "flower.stl", size_mm=80, resolution=200)
```

Or, when you want the generator around for more than one step:

```python
from complexplorer.export.stl import OrnamentGenerator

ornament = OrnamentGenerator(lambda z: z / (z**10 - 1), resolution=200, scaling="arctan")
ornament.generate_and_save("flower.stl", size_mm=80)
```

From the command line:

```bash
complexplorer stl preset:pole_flower_10 --size-mm 80 --resolution 200 -o flower.stl
```

## What the settings do

- **`resolution`** is the sphere's sampling density, and it is the main quality dial. 150 is a
  reasonable default; 200–300 gives crisper spikes and a proportionally larger file.
- **`scaling`** decides how `|f(z)|` becomes displacement. `arctan` is the default because it is
  bounded: a pole becomes a tall spike rather than an infinite one. `scaling_params` tunes it, and
  `poles_emphasis` from `get_scaling_preset()` is a good starting point for pole-heavy functions.
- **`size_mm`** is the printed size of the longest axis, applied at export. Geometry is scaled at
  the end, so changing it does not change the shape.

## What happens before the file is written

`generate_and_save` repairs and checks the mesh, printing what it finds:

- **Repair** (`repair=True`) cleans degenerate faces and closes small holes. Riemann sphere meshes
  are rectangular grids, so they have seams at the poles; this is where those get handled.
- **Validation** (`validate=True`) reports whether the mesh is watertight and manifold, its
  dimensions, its volume, and an estimated minimum wall thickness at the requested size.

The wall-thickness estimate is the one to read. A shape that is geometrically fine can still be
unprintable because some feature is thinner than a nozzle can lay down — the report says so, and
suggests a larger `size_mm`. A small test ornament at 30 mm will often report walls that are too
thin; the same model at 80 mm prints without complaint.

Set `verbose=False` to silence the report, and `validate=False` to skip the checks if you are
exporting in bulk and will inspect elsewhere.

## Printing notes

What comes out is a closed, roughly spherical shell with the function's poles as spikes. The
spikes are the feature that fails: they are the thinnest part of the model and the furthest from
the body, so they are where both slicing and printing go wrong first.

Two dials help, in this order:

1. **Increase `size_mm`.** Wall thickness scales with the model, and the validation report tells
   you the size at which the thin parts clear a typical nozzle.
2. **Compress the poles harder.** A scaling mode that flattens large moduli — or `arctan` with
   tuned `scaling_params` — makes shorter, thicker spikes.

Increasing `resolution` does not help here; it makes a finer mesh of the same thin geometry.

Orientation and supports are your slicer's business, and depend on the model and the printer.
Complexplorer writes the geometry; it does not decide how it is printed.

## Working with the mesh directly

```python
from complexplorer.export.stl import validate_printability, scale_to_size, center_mesh
```

These operate on the PyVista `PolyData`, so a mesh can be checked, scaled and centred as part of a
larger pipeline before it is written.
