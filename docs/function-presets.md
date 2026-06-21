# Function presets (`cp.catalog`)

`cp.catalog` is a curated registry of complex functions, each described in a way that is
both **renderable** (a Python callable) and **serializable** (plain dicts + an expression
string + exact answer keys). It is the single source consumed by the gallery, CLI, STL
object cards, and — under the project's *games boundary* — Godot game prototyping, where the
math is reimplemented natively and validated against these records.

> Not to be confused with `cp.Presets` (capital P) — those are *plot-config* presets
> (resolution + colormap bundles). `cp.catalog` is the *function* registry.

## Using the registry

```python
import complexplorer as cp

cp.catalog.list()                              # -> sorted preset ids
p = cp.catalog.get("pole_flower_10")
cp.catalog.filter(tag="singularity-detective") # -> [FunctionPreset, …]

p.func(0.5 + 0.2j)        # the callable Complexplorer renders with
p.domain()                # a live Domain   (built from p.domain_spec)
p.colormap()              # a live Colormap (built from p.cmap_spec)
p.to_dict()               # JSON-ready record (everything except the callable)
```

## The model

A `FunctionPreset` carries:

| Field | Purpose |
|---|---|
| `func` | the callable Complexplorer renders with (**not** serialized) |
| `expression` | e.g. `"z / (z**10 - 1)"` — what Godot reimplements from |
| `domain_spec` / `cmap_spec` / `scaling_spec` | plain dicts whose keys mirror the constructor kwargs |
| `singularities` | hand-authored, **exact** answer keys, one record per location |
| `id` / `title` / `story` / `tags` | metadata; `tags` group presets (e.g. game sets) |

**Serialization is the design center.** Specs are dicts, not live objects, so a preset is a
clean JSON record. Every complex value (a domain `center`, a singularity `at`) is an
`[re, im]` pair, since JSON has no complex type. The factories `domain_from_spec` /
`cmap_from_spec` rebuild live objects on demand; the core `Domain`/`Colormap` classes are
untouched.

## Singularity answer keys

Each `singularities` record is exact and author-provided (never detected numerically):

```python
{"type": "pole", "at": [1.0, 0.0], "order": 1}          # type ∈ {zero, pole, essential, branch_point}
```

`order` is the multiplicity for `zero`/`pole`, the branching order for `branch_point`, and
`null` for `essential`. Multivalued presets (`sqrt`, `log`, `z**(1/3)`) use numpy's
**principal branch** and declare a `branch_point`, so the answer key and any native
reimplementation agree on the branch convention.

## Parametrized families

The base `FunctionPreset` is **static**. Parametrized "playgrounds" (Möbius, Julia,
`z^(1/n)`, resonators) are a separate, later `FunctionFamily` whose `bind(**params)` emits an
ordinary static preset — so the registry shape here is the snapshot families produce.
