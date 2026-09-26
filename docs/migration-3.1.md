# Migrating from 3.0 to 3.1

3.1 is about one thing: ornaments stop being blobs. That means **every ornament generated with
default settings changes shape** — deliberately, and it is the point of the release. Nothing else in
the library is affected; 2D portraits, landscapes, the Riemann sphere renderers and their regression
baselines are all untouched.

If you generate ornaments, read the two sections that follow. If you wrote a `custom` scaling
callable, read the third — it changes geometry **without raising**, which is the one thing here that
can go wrong quietly.

## If you only do two things

1. **Re-check any ornament you had dialled in**, or pin the old behaviour explicitly (below).
2. **If you wrote a `custom` scaling callable**, divide its return by your radius range — see
   [the `custom` fix](#the-custom-scaling-fix-changes-geometry-silently).

## Ornament geometry changes

Three defaults moved, and their effects compound:

| setting | 3.0 | 3.1 | why |
|---|---|---|---|
| normalization | none | `normalize="geometric"` | the constant in front of `f` changed the shape, not just the labels |
| `scaling` | `"arctan"` | `"logarithmic"` | `arctan` admits no scale, so tip sharpness was not tunable at all |
| depth (`r_min`) | 0.5 | 0.2 | every named scaling preset already said 0.2; only the STL defaults diverged |

To reproduce the 3.0 look exactly:

```python
from complexplorer.export.stl import OrnamentGenerator

ornament = OrnamentGenerator(
    lambda z: z / (z**10 - 1),
    resolution=200,
    scaling="arctan",                              # the old transfer
    scaling_params={"r_min": 0.5, "r_max": 1.0},   # the old depth
    normalize=None,                                # the old (absent) normalization
)
```

`normalize=None` on its own reproduces the old normalization behaviour; the old *look* needs all
three. Note that the useful thing to do is usually the opposite — keep the new defaults and adjust
`pointiness` and `contrast` — because normalization is what stops the arbitrary scale of your
function from deciding the geometry. The
[physical workflow guide](guide/physical-workflow.md#why-is-my-ornament-a-blob) explains the
reasoning.

## The `custom` scaling fix changes geometry silently

**This is the one to check.** In 3.0, dispatching `scaling="custom"` short-circuited
`ModulusScaling.custom` and used your callable's return value as the radius directly. Any `r_min` and
`r_max` you passed alongside it were silently discarded.

So anyone who used `custom` successfully wrote a callable returning a **radius**, because that was
the only thing that worked. In 3.1 dispatch goes through the documented contract: your output is
clipped to `[0, 1]` and then mapped onto `[r_min, r_max]`.

A radius-returning callable already satisfies `[0, 1]`, so nothing raises — the value is simply
remapped, and the geometry comes out wrong. With the default bounds of `[0.5, 1.5]`, a callable that
returned 0.2 now yields a radius of 0.7.

```python
# 3.0: the callable returned a radius, baking the depth in itself.
def transfer(moduli):
    height = 1 / (1 + np.exp(-np.log(moduli) / 2.0))
    return 0.2 + 0.8 * height          # a radius in [0.2, 1.0]

# 3.1: return [0, 1] and let the bounds do the mapping.
def transfer(moduli):
    return 1 / (1 + np.exp(-np.log(moduli) / 2.0))   # in [0, 1]

# ...with the depth where it belongs:
scaling_params = {"scaling_func": transfer, "r_min": 0.2, "r_max": 1.0}
```

The one-line version: **drop the `r_min + (r_max - r_min) *` wrapper from your callable and pass
those bounds as parameters instead.**

If your two-scale transfer was a hand-rolled mixture of logistics, `contrast=(boost, weight)` now
does it for you — see [the guide](guide/physical-workflow.md#sculpting-the-body-with-contrast).

## `validate_printability` loses two keys and gains two

| 3.0 key | 3.1 |
|---|---|
| `wall_thickness_ok` | removed |
| `estimated_min_wall_mm` | removed |
| `recommended_size_mm` | removed |
| — | `min_radius_mm` |
| — | `max_radius_mm` |

Reading a removed key by name now raises `KeyError`, so this is a visible break rather than a quiet
one. It is worth knowing that the removed keys were never meaningful: validation ran *after*
`scale_to_size`, so the scale factor was always 1.0 and `wall_thickness_ok` reduced to `0.3 >= 0.8`
— constant `False` for every mesh ever exported, with a recommendation that always suggested a size
2.67× larger. Nothing could have depended on it being right.

A relief has no walls. It is a star-shaped solid about the origin, so the meaningful measurement is
its radial extent, and it is at least `2 * min_radius_mm` thick through the centre. What genuinely
fails on a fused-deposition printer is the ridge width between two adjacent pits; that is not
measured, and the verbose report now says so instead of substituting a number for it.

## Topology reporting is fixed

If you worked around `validate_printability` reporting your closed meshes as open, you can stop.

`extract_feature_edges` enables boundary, feature, manifold and non-manifold extraction by default,
so requesting boundary edges alone returned essentially every edge in the mesh — a `pyvista.Cube`,
closed and manifold with `n_open_edges == 0`, was reported as neither watertight nor manifold,
because each of its twelve creases read as a hole. `repair_mesh_simple` and `close_mesh_holes`
carried the same mistake in their progress counters, which is why repair looked like it was failing
while it was in fact working.

Counts are now correct, which means they can be trusted and asserted on. `count_edges` is available
from `complexplorer.export.stl` if you want to do the same thing in your own code.

## Saved meshes carry oriented normals

`save_stl` now computes consistent, outward-facing normals before writing, so the facet normals
recorded in the STL mean something to a consumer that reads them rather than recomputing. If you were
running `compute_normals` yourself after export, you no longer need to.

## New parameters, all optional

None of these are required, and the defaults are what change the look:

| parameter | default | what it does |
|---|---|---|
| `normalize` | `"geometric"` | `"median"`, a float, or `None`; puts sea level where the function lives |
| `pointiness` | 2.0 | tip exponent is `1 / pointiness`; larger is sharper |
| `pole_order` | 1.0 | so a double pole prints as sharp as a simple one |
| `sharpness` | — | the log-modulus scale directly; `ln(10)` makes one unit of relief 20 dB |
| `contrast` | off | `(boost, weight)` sculpts the body between sparse features |

`cp.normalization_constant` and `cp.sampled_normalization_constant` are new public helpers. The first
is the exact closed form for a rational function and requires a **complete** divisor; the second
estimates the same quantity from samples and requires none. Read the
[warning on the first](api/scaling.md#normalization) before using it — with a partial divisor it
returns a confidently wrong number rather than an error.
