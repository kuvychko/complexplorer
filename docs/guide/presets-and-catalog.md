# The catalog and the plot presets

Two things in complexplorer are called presets, and they answer different questions.

> **`cp.catalog` supplies a function. `cp.Presets` configures a render.**

## `cp.catalog` — curated functions

The catalog is seventeen mathematical functions worth looking at, each with the metadata that makes
it teachable:

```python
import complexplorer as cp

preset = cp.catalog.get("pole_flower_10")
preset.func            # the callable
preset.expression      # 'z / (z**10 - 1)'
preset.singularities   # the answer key: ten poles and a zero, with positions and orders
preset.story           # one or two sentences on what to look for
preset.tags            # ('ornament', 'poles', 'canonical', 'singularity-detective')

cp.catalog.list()                        # every id
cp.catalog.filter(tag="branches")        # the branch-cut set
```

The `singularities` record is what distinguishes a catalog entry from a lambda: it states where the
zeros and poles are and what order they have, so a picture can be checked against the mathematics
rather than admired. `answer_key_stats()` summarises it — how many of each type, and how close
together the closest pair is.

The catalog is also what `complexplorer list`, `complexplorer gallery` and the documentation
gallery are built from, so an entry added there propagates everywhere.

## `cp.Presets` — render configurations

These are bundles of plotting arguments for a purpose, not functions:

```python
cp.quick_plot(lambda z: 1 / z, **cp.Presets.publication_ready())
cp.quick_plot(lambda z: 1 / z, **cp.Presets.high_contrast())
cp.quick_plot(lambda z: 1 / z, **cp.Presets.interactive())
```

- `publication_ready()` — high resolution, restrained styling
- `high_contrast()` — stronger separation, for projection or for readers who need it
- `interactive()` — lighter settings that stay responsive while you explore

Because they are plain dictionaries of keyword arguments, you can override any part:

```python
cp.quick_plot(f, **{**cp.Presets.publication_ready(), "resolution": 1200})
```

## Using both at once

```python
preset = cp.catalog.get("rational_zeros_poles")
cp.quick_plot(preset.func, **cp.Presets.publication_ready())
```

One supplies the mathematics, the other the presentation.
