# API map

One table per thing you might be trying to do, with what each entry point hands back. Everything
here is imported as `import complexplorer as cp`, except engineering mode, which lives under
`cp.ee`.

## Draw a picture

| You want | Call | You get back |
|---|---|---|
| A 2D phase portrait | `cp.plot(domain, func, cmap=...)` | a matplotlib `Axes` |
| Domain and codomain side by side | `cp.pair_plot(domain, func, cmap=...)` | a matplotlib `Figure` |
| A flat stereographic chart | `cp.riemann_chart(func)` | a matplotlib `Axes` |
| Both hemispheres as charts | `cp.riemann_hemispheres(func)` | a matplotlib `Figure` |
| A 3D analytic landscape | `cp.plot_landscape_pv(domain, func)` | a PyVista `Plotter`, or `None` |
| Two landscapes side by side | `cp.pair_plot_landscape_pv(domain, func)` | a PyVista `Plotter`, or `None` |
| The Riemann sphere | `cp.riemann_pv(func)` | a PyVista `Plotter`, or `None` |
| A Riemann surface | `cp.riemann_surface_pv("power", n=2)` | a PyVista `Plotter`, or `None` |
| Something, quickly | `cp.quick_plot(func, mode="2d")` | whichever of the above it dispatched to |

Every 3D entry point takes `filename=` and `interactive=False` to write an image instead of
opening a window, and `return_plotter=True` to hand back the `Plotter` for further composition.
The 2D entry points take `filename=` too.

## Choose what to draw

| You want | Call | You get back |
|---|---|---|
| A region of the plane | `cp.Rectangle(4, 4)`, `cp.Disk(2)`, `cp.Annulus(0.2, 3)` | a `Domain` |
| A region built from others | `a \| b`, `a & b`, `a - b` | a `CompositeDomain` |
| A colour convention | `cp.Phase(phase_sectors=6)` and the other families | a `Colormap` |
| A curated function | `cp.catalog.get("pole_flower_10")` | a `FunctionPreset` |
| Several of them | `cp.catalog.filter(tag="ornament")` | a list of `FunctionPreset` |
| Settings for a render | `cp.PlotPresets.publication_ready()` | a dict to spread as keyword arguments |

`PlotPresets` configures a render; `catalog` supplies a function.

## Engineering mode

| You want | Call | You get back |
|---|---|---|
| A transfer function | `cp.ee.TransferFunction(num, den)` | a callable object with `poles`, `zeros`, `is_stable` |
| Its phase portrait | `cp.ee.transfer_portrait(H)` | a matplotlib `Axes` |
| Poles and zeros | `cp.ee.pole_zero_plot(H)` | a matplotlib `Axes` |
| Bode or Nyquist | `cp.ee.bode_plot(H)`, `cp.ee.nyquist_plot(H)` | a matplotlib `Figure` |
| The frequency response | `H.frequency_response()` | `(omega, response)`, both arrays |

A `TransferFunction` is a plain callable, so every renderer above accepts it directly.

## Produce a file

| You want | Call | You get back |
|---|---|---|
| A printable ornament | `cp.create_ornament(func, "out.stl", size_mm=80)` | the path written |
| The same, with the mesh in hand | `cp.OrnamentGenerator(func).generate_and_save(...)` | the path written |
| A reproducible asset bundle | `cp.generate_gallery(out_dir, selection=...)` | the `index.json` manifest as a dict |
| An image from any renderer | pass `filename=` | the file is written; the return is unchanged |

## Scaling, and the pieces underneath

| You want | Call | You get back |
|---|---|---|
| To compress `\|f(z)\|` into height | `modulus_mode="arctan"` on a 3D call | — |
| The scaling modes themselves | `cp.ModulusScaling` | a class of static methods |
| A bundled scaling | `cp.get_scaling_preset("balanced")` | a settings dict |
| Phase or a sawtooth directly | `cp.phase(z)`, `cp.sawtooth(x)` | an array |
| Stereographic projection | `cp.stereographic_projection(z)`, `cp.inverse_stereographic(...)` | arrays |

## When something goes wrong

Every error the library raises derives from `cp.ComplexplorerError`, so one `except` catches all of
them. `cp.ValidationError` covers bad arguments, and `cp.ColormapError` is a `ValidationError` for
colormap configuration specifically.
