# Command-line interface

Installing Complexplorer provides a `complexplorer` command:

```bash
complexplorer render <func> [options]   # 2D/3D image
complexplorer stl    <func> [options]   # 3D-printable STL
complexplorer list   [--tag TAG]        # browse the preset catalog
```

## The function argument

`<func>` is either a **registry preset** or an **expression**:

```bash
complexplorer render preset:pole_flower_10 -o flower.png
complexplorer render "z / (z**10 - 1)" --domain annulus:0.2:3 -o flower.png
```

- `preset:<id>` resolves through `cp.catalog` and uses the preset's recommended
  domain / colormap / scaling as defaults (any flag overrides them).
- An expression is evaluated by the safe, portable evaluator (`core/expression.py`):
  only `z`, numeric/imaginary literals, arithmetic, and curated functions
  (`sin cos tan exp log sqrt abs conj real imag …`, constants `pi e j`). Python-isms like
  `z.real` or attribute/method access are rejected — use `real(z)` instead.

## Shorthands

```
--domain  rect:RE:IM | disk:R | annulus:IN:OUT     (→ the catalog's domain factory)
--cmap    phase | phase:6                            (→ the catalog's colormap factory)
--scaling arctan | logarithmic | …                  (modulus scaling for 3D/relief)
```

## Examples

```bash
# 2D phase portrait of a registry preset
complexplorer render preset:rational_zeros_poles -o rational.png

# 3D Riemann relief of an expression
complexplorer render "z**3 - z" --mode riemann --scaling arctan -o relief.png

# 3D landscape, open an interactive window instead of saving
complexplorer render preset:square --mode 3d --show

# Export a printable ornament
complexplorer stl preset:pole_flower_10 --size-mm 80 --resolution 200 -o flower.stl

# List the presets tagged for the singularity-detective game set
complexplorer list --tag singularity-detective
```

## Backends

`render --mode 2d` and `list` work without the 3D backend. `render --mode 3d|riemann` and
`stl` require PyVista (`pip install "complexplorer[pyvista]"`) and exit with a clear message
if it is not installed — consistent with the 2D-matplotlib / 3D-PyVista backend policy.
