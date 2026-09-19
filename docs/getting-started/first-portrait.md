# Installation and your first portrait

## Install

```bash
pip install complexplorer
```

That is everything. PyVista is a required dependency as of 3.0, so 3D landscapes, Riemann spheres
and STL export work out of the box — there is no optional 3D extra to remember and no capability
flag to check. The install is large, mostly because VTK is; [the backend
policy](../development/backend-policy.md) records the measurement and the reasoning.

Optional extras:

```bash
pip install "complexplorer[qt]"       # interactive matplotlib windows in scripts
pip install "complexplorer[examples]" # tooling to run the example notebooks
```

Python 3.11 or newer is required.

## Your first portrait

A portrait needs three things: a region of the plane, a function, and a colormap.

```python
import complexplorer as cp

domain = cp.Rectangle(re_length=4, im_length=4)
func = lambda z: (z**2 - 1) / (z**2 + 1)
cmap = cp.Phase(phase_sectors=6, auto_scale_r=True)

cp.plot(domain, func, cmap=cmap, legend=True)
```

[![Phase portrait of (z^2-1)/(z^2+1) with a phase-wheel legend](../examples/gallery/view/_tour/legend_portrait.png)](../examples/gallery/_tour/legend_portrait.png)

Four things are worth naming, because they recur everywhere in the library:

- **`cmap` is keyword-only in practice.** `plot(domain, func, cmap)` will not do what you want;
  the third positional parameter is `z`, a pre-computed grid.
- **`phase_sectors=6`** cuts the colour wheel into six bands, so winding is countable rather than a
  smooth smear. `auto_scale_r=True` sizes the modulus bands to match, which makes the cells square.
- **`legend=True`** insets the same colormap applied to the identity map. It is the key to the
  picture: see [reading a phase portrait](../guide/reading-a-portrait.md).
- **`plot` returns the matplotlib `Axes`**, so you can keep styling it.

## Saving instead of showing

Every renderer takes `filename`, which writes the file instead of opening a window:

```python
cp.plot(domain, func, cmap=cmap, legend=True, filename="portrait.png")
```

This is the form to use in scripts, in CI, and on a headless machine. The phase-wheel legend is
drawn as an inset *inside* the portrait, not alongside it, so saving cannot crop it off.

## Turning it into a landscape

The same domain, the same function, one different call:

```python
cp.plot_landscape_pv(domain, func, cmap=cmap)
```

[![The same function as a 2D portrait and as a 3D analytic landscape](../examples/gallery/view/_tour/portrait_to_landscape.png)](../examples/gallery/_tour/portrait_to_landscape.png)

`|f(z)|` becomes height while the colours stay exactly as they were. That is the whole relationship
between the 2D and 3D views, and it is the subject of [3D landscapes and the Riemann
sphere](../guide/three-dimensions.md).

## One-liners

When you only want to look at something, `quick_plot` picks sensible defaults:

```python
cp.quick_plot(lambda z: 1 / z)                  # 2D
cp.quick_plot(lambda z: 1 / z, mode="3d")       # analytic landscape
cp.quick_plot(lambda z: 1 / z, mode="riemann")  # Riemann sphere
```

## Where to go next

- [Reading a phase portrait](../guide/reading-a-portrait.md) — what the colours actually encode
- [Domains and colormaps](../guide/domains-and-colormaps.md) — regions you can do arithmetic on
- [The gallery](../gallery/gallery.generated.md) — every preset and colormap, with the code
