# API reference

Everything public is re-exported from the top-level package, so `import complexplorer as cp` is all
you need — except engineering mode, which lives under `cp.ee`. The pages below group the surface by
what you are trying to do rather than by module, because the module layout is internal organisation
that the flat API deliberately hides.

| Group | What is in it |
|---|---|
| [Domains](domains.md) | `Rectangle`, `Disk`, `Annulus`, and set arithmetic |
| [Colormaps](colormaps.md) | `Phase`, the perceptual families, and the pattern maps |
| [Plotting (2D)](plotting-2d.md) | `plot`, `pair_plot`, the stereographic charts |
| [Plotting (3D)](plotting-3d.md) | landscapes, the Riemann sphere, Riemann surfaces |
| [Scaling](scaling.md) | `ModulusScaling` and `get_scaling_preset` |
| [Catalog and presets](catalog.md) | `catalog`, `FunctionPreset`, `Presets`, `quick_plot`, `generate_gallery` |
| [Export](export.md) | `OrnamentGenerator`, `create_ornament` |
| [Engineering mode](engineering.md) | `cp.ee` |
| [Functions and exceptions](core.md) | `phase`, `sawtooth`, projections, the exception types |

All of it is generated from the docstrings in the source, so it cannot drift from the code.
