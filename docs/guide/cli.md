# Command line

```
complexplorer render <func> [options]   # a 2D or 3D image
complexplorer stl    <func> [options]   # a 3D-printable STL
complexplorer list   [--tag TAG]        # browse the preset catalog
complexplorer gallery [options]         # a reproducible asset bundle
```

Every command takes a function the same way: either `preset:<id>` from the catalog, or an
expression string such as `'z/(z**10-1)'`, which is evaluated safely — it is parsed, not `eval`ed.

## `render`

```bash
complexplorer render preset:pole_flower_10 -o flower.png
complexplorer render "z / (z**10 - 1)" --domain annulus:0.2:3 -o flower.png
complexplorer render "z**3 - z" --mode riemann --scaling arctan -o relief.png
complexplorer render preset:square --mode 3d --show
```

| Option | Meaning |
|---|---|
| `--mode {2d,3d,riemann}` | flat portrait, analytic landscape, or Riemann sphere |
| `--domain` | `rect:RE:IM`, `disk:R` or `annulus:IN:OUT` |
| `--cmap` | `phase`, or `phase:6` for six sectors |
| `--scaling` | modulus scaling for `3d` and `riemann`, e.g. `arctan` |
| `--resolution` | samples per axis |
| `--output`, `-o` | write to a file |
| `--show` | open an interactive window instead |

## `stl`

```bash
complexplorer stl preset:pole_flower_10 --size-mm 80 --resolution 200 -o flower.stl
```

`--size-mm` is the printed size of the longest axis, `--scaling` defaults to `arctan`, and the
mesh is repaired and checked for printability before it is written. See
[the physical workflow](physical-workflow.md) for what those checks mean.

## `list`

```bash
complexplorer list
complexplorer list --tag singularity-detective
```

Prints the catalog: seventeen presets, each with its id, title and tags. The tags are how the
catalog is sliced — `canonical`, `poles`, `branches`, `ornament`, `transcendental` and others.

## `gallery`

```bash
complexplorer gallery -o bundle/
complexplorer gallery --tag ornament -o ornaments/
complexplorer gallery --preset identity square sqrt -o three/
```

Renders a preset selection into a self-contained bundle: a portrait and a `card.json` per preset,
plus a top-level `index.json` manifest. The manifest is byte-stable — identical for the same
selection and library version, on any platform — which is what makes it usable as an interchange
record rather than just a folder of images.

This is the library's own gallery generator. The documentation gallery on this site is produced by
`examples/showcase.py`, which adds the curated tour on top of it.

## A note on consoles

Output degrades rather than crashing on a console that cannot represent a character — a preset
title such as `z³ - z` on a legacy Windows code page will show an escape rather than ending the
command in a traceback.
