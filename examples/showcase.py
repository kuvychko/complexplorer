#!/usr/bin/env python3
"""examples/showcase.py — the high-res visual gallery producer (M2).

Renders the curated preset registry (``cp.catalog``) into the committed visual gallery:
the 2D portraits + deterministic ``index.json`` (via the library ``cp.gallery``), plus the
PyVista 3D screenshots that manifest deliberately omits, a colormap gallery, a presentation
manifest (``showcase.json``), and a generated docs gallery page.

The render set per preset follows the preset's TAGS (which encode mathematical character):

    every preset      -> portrait.png   (2D, from cp.gallery)
    canonical         -> landscape.png + sphere.png
    branches          -> surface.png    (riemann_surface_pv)
    ornament          -> ornament.png   (relief sphere)

This is a LOCAL regeneration tool — off-screen VTK screenshots crash only on headless CI.
Run it from the repo root:

    python examples/showcase.py

It is idempotent: re-running reproduces the same bundle (images best-effort, manifest stable).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import complexplorer as cp
from complexplorer import generate_gallery
from complexplorer._version import __version__
from complexplorer.core.presets import catalog
from complexplorer.plotting.matplotlib.plot_2d import plot as plot_2d

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
GALLERY_DIR = REPO_ROOT / "examples" / "gallery"
DOCS_GALLERY = REPO_ROOT / "docs" / "gallery"
COLORMAPS_DIR = GALLERY_DIR / "_colormaps"

SCHEMA_VERSION = 1  # showcase manifest schema (independent of cp.gallery's index.json)
HERO_BANNER = "Riemann_relief_map_20250726.png"  # curated, kept (not regenerated)

PORTRAIT_DPI = 390          # 2D portraits / colormaps (figsize 4x4 -> ~1560px); +30%
WINDOW_3D = (1560, 1560)    # PyVista screenshot size; +30%
RESOLUTION_3D = 260         # mesh resolution for landscapes/spheres; +30%
SURFACE_RESOLUTION = 104    # Riemann-surface radial samples; +30%

COLORMAP_REFERENCE = "rational_zeros_poles"  # zeros at +-1, poles at +-i — good contrast

# The colormap family — only the colormaps that actually exist in the public API.
def _colormap_family() -> list[tuple[str, cp.Colormap, str]]:
    """(name, colormap, snippet-constructor) tuples for the colormap gallery.

    Covers every concrete colormap the package exports, so the gallery cannot drift from the
    public API (see the `examples` capability).
    """
    import numpy as np

    return [
        ("phase_basic", cp.Phase(), "cp.Phase()"),
        ("phase_enhanced", cp.Phase(phase_sectors=6), "cp.Phase(phase_sectors=6)"),
        ("phase_modulus", cp.Phase(r_linear_step=0.6), "cp.Phase(r_linear_step=0.6)"),
        ("phase_full", cp.Phase(phase_sectors=6, auto_scale_r=True),
         "cp.Phase(phase_sectors=6, auto_scale_r=True)"),
        ("oklab_phase", cp.OklabPhase(phase_sectors=6), "cp.OklabPhase(phase_sectors=6)"),
        ("perceptual_pastel", cp.PerceptualPastel(phase_sectors=6),
         "cp.PerceptualPastel(phase_sectors=6)"),
        ("analogous_wedge", cp.AnalogousWedge(phase_sectors=6),
         "cp.AnalogousWedge(phase_sectors=6)"),
        ("diverging_warm_cool", cp.DivergingWarmCool(phase_sectors=6),
         "cp.DivergingWarmCool(phase_sectors=6)"),
        ("isoluminant", cp.Isoluminant(phase_sectors=6), "cp.Isoluminant(phase_sectors=6)"),
        ("cubehelix_phase", cp.CubehelixPhase(phase_sectors=6),
         "cp.CubehelixPhase(phase_sectors=6)"),
        ("ink_paper", cp.InkPaper(phase_sectors=6), "cp.InkPaper(phase_sectors=6)"),
        ("earth_topographic", cp.EarthTopographic(phase_sectors=6),
         "cp.EarthTopographic(phase_sectors=6)"),
        ("four_quadrant", cp.FourQuadrant(phase_sectors=6), "cp.FourQuadrant(phase_sectors=6)"),
        ("chessboard", cp.Chessboard(spacing=0.25), "cp.Chessboard(spacing=0.25)"),
        ("polar_linear", cp.PolarChessboard(phase_sectors=6, spacing=0.25),
         "cp.PolarChessboard(phase_sectors=6, spacing=0.25)"),
        ("polar_log", cp.PolarChessboard(phase_sectors=6, r_log=np.e),
         "cp.PolarChessboard(phase_sectors=6, r_log=np.e)"),
        ("logrings", cp.LogRings(log_spacing=0.2), "cp.LogRings(log_spacing=0.2)"),
    ]


# Tag -> extra render types (beyond the always-present portrait).
TAG_RENDERS = {
    "canonical": ("landscape", "sphere"),
    "branches": ("surface",),
    "ornament": ("ornament",),
}

# Multivalued preset id -> riemann_surface_pv (family, kwargs).
SURFACE_FAMILY = {
    "sqrt": ("power", {"n": 2}),
    "cbrt": ("power", {"n": 3}),
    "log": ("log", {}),
}


# ---------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------

def _write_json(path: Path, obj: dict) -> None:
    """Deterministic JSON: sorted keys, stable indent, LF, trailing newline."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, sort_keys=True, indent=2, ensure_ascii=False)
        f.write("\n")


def _renders_for(preset) -> list[str]:
    """The render types a preset gets, per the tag policy (portrait always first)."""
    types = ["portrait"]
    for tag in preset.tags:
        for rtype in TAG_RENDERS.get(tag, ()):
            if rtype not in types:
                types.append(rtype)
    return types


def _render_portrait_mpl(domain, func, cmap, path: Path) -> None:
    """A 2D portrait via matplotlib (used for the colormap gallery)."""
    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    try:
        plot_2d(domain, func, cmap=cmap, ax=ax)
        fig.savefig(path, dpi=PORTRAIT_DPI, metadata={"Software": None})
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------------------
# Per-preset 3D renders
# ---------------------------------------------------------------------------------------

def _render_landscape(preset, path: Path) -> None:
    sc = preset.scaling()
    cp.plot_landscape_pv(
        preset.domain(), preset.func, cmap=preset.colormap(),
        modulus_mode=sc["method"], modulus_params=sc["params"],
        resolution=RESOLUTION_3D, window_size=WINDOW_3D,
        interactive=False, filename=str(path),
    )


def _render_sphere(preset, path: Path) -> None:
    # domain=None -> no stereographic mask -> the FULL sphere (both poles), no "cup".
    cp.riemann_pv(
        preset.func, cmap=preset.colormap(),
        modulus_mode="constant", resolution=RESOLUTION_3D, window_size=WINDOW_3D,
        interactive=False, filename=str(path),
    )


def _render_ornament(preset, path: Path) -> None:
    sc = preset.scaling()
    # domain=None -> full sphere relief (the ∞ pole is included, not cut off).
    cp.riemann_pv(
        preset.func, cmap=preset.colormap(),
        modulus_mode=sc["method"], modulus_params=sc["params"],
        resolution=RESOLUTION_3D, window_size=WINDOW_3D,
        interactive=False, filename=str(path),
    )


def _render_surface(preset, path: Path) -> None:
    family, kw = SURFACE_FAMILY[preset.id]
    cp.riemann_surface_pv(
        family, **kw, resolution=SURFACE_RESOLUTION, window_size=WINDOW_3D,
        interactive=False, filename=str(path),
    )


_RENDERERS = {
    "landscape": _render_landscape,
    "sphere": _render_sphere,
    "ornament": _render_ornament,
    "surface": _render_surface,
}


# ---------------------------------------------------------------------------------------
# Snippets (registry-driven — the expression strings are math notation, not runnable code)
# ---------------------------------------------------------------------------------------

def _snippet(preset, rtype: str) -> str:
    head = f'preset = cp.catalog.get("{preset.id}")   # f(z) = {preset.expression}'
    if rtype == "portrait":
        body = "cp.plot(preset.domain(), preset.func, cmap=preset.colormap())"
    elif rtype == "landscape":
        body = (
            'sc = preset.scaling()\n'
            "cp.plot_landscape_pv(preset.domain(), preset.func, cmap=preset.colormap(),\n"
            '                     modulus_mode=sc["method"], modulus_params=sc["params"])'
        )
    elif rtype == "sphere":
        body = 'cp.riemann_pv(preset.func, cmap=preset.colormap())   # full sphere'
    elif rtype == "ornament":
        body = (
            'sc = preset.scaling()\n'
            "cp.riemann_pv(preset.func, cmap=preset.colormap(),\n"
            '              modulus_mode=sc["method"], modulus_params=sc["params"])  # relief'
        )
    elif rtype == "surface":
        family, kw = SURFACE_FAMILY[preset.id]
        args = f'"{family}"' + ("".join(f", {k}={v}" for k, v in kw.items()))
        return f"import complexplorer as cp\ncp.riemann_surface_pv({args})   # {preset.expression}"
    else:  # pragma: no cover
        body = ""
    return f"import complexplorer as cp\n{head}\n{body}"


def _colormap_snippet(ctor: str) -> str:
    return (
        "import complexplorer as cp\n"
        f'preset = cp.catalog.get("{COLORMAP_REFERENCE}")\n'
        f"cp.plot(preset.domain(), preset.func, cmap={ctor})"
    )


# ---------------------------------------------------------------------------------------
# Docs page generation
# ---------------------------------------------------------------------------------------

def _md_entry(title: str, story: str, images: list[str], snippet: str) -> str:
    lines = [f"### {title}", ""]
    if story:
        lines += [story, ""]
    for img in images:
        lines.append(f"![{title}](../../examples/gallery/{img})")
    lines += ["", "```python", snippet, "```", ""]
    return "\n".join(lines)


def _generate_docs_page(manifest: dict) -> None:
    out = ["<!-- GENERATED by examples/showcase.py — do not edit by hand. -->", ""]
    out += ["# Gallery (generated)", "", "## Functions", ""]
    for rec in manifest["presets"]:
        preset = catalog.get(rec["id"])
        imgs = [rec["renders"][rt] for rt in rec["renders"]]
        # one snippet per distinct render type, concatenated
        snip = "\n\n# ---\n".join(_snippet(preset, rt) for rt in rec["renders"])
        out.append(_md_entry(rec["title"], rec.get("story", ""), imgs, snip))

    out += ["## Colormaps", ""]
    ref = catalog.get(manifest["colormaps"]["reference_preset"])
    out.append(f"*Reference function:* `{ref.expression}` (`{ref.id}`)\n")
    for r in manifest["colormaps"]["renders"]:
        out.append(_md_entry(r["name"], "", [r["file"]], _colormap_snippet(r["ctor"])))

    (DOCS_GALLERY / "gallery.generated.md").write_text("\n".join(out), encoding="utf-8", newline="\n")


# ---------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------

def _existing_manifest() -> dict:
    path = GALLERY_DIR / "showcase.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main(only: str = "all") -> None:
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    COLORMAPS_DIR.mkdir(parents=True, exist_ok=True)

    if only == "colormaps":
        # Reuse the committed preset renders; only the colormap section is re-rendered.
        preset_records = _existing_manifest().get("presets", [])
    else:
        # 1. Deterministic 2D portraits + index.json (the library bundle; untouched contract).
        print("Rendering 2D portraits + index.json (cp.gallery) ...")
        generate_gallery(GALLERY_DIR, selection=None, dpi=PORTRAIT_DPI)

        # 2. Per-preset PyVista screenshots, by tag policy.
        preset_records = _render_presets()
    _finish(preset_records)


def _render_presets() -> list[dict]:
    preset_records = []
    for pid in catalog.list():
        preset = catalog.get(pid)
        rtypes = _renders_for(preset)
        renders = {"portrait": f"{pid}/portrait.png"}
        for rtype in rtypes:
            if rtype == "portrait":
                continue
            rel = f"{pid}/{rtype}.png"
            print(f"  {pid}: {rtype}")
            _RENDERERS[rtype](preset, GALLERY_DIR / rel)
            renders[rtype] = rel
        preset_records.append(
            {
                "id": pid,
                "title": preset.title,
                "expression": preset.expression,
                "story": preset.story,
                "tags": list(preset.tags),
                "renders": renders,
            }
        )

    return preset_records


def _finish(preset_records: list[dict]) -> None:
    # 3. Colormap gallery — one reference function under each implemented colormap.
    print("Rendering colormap gallery ...")
    ref = catalog.get(COLORMAP_REFERENCE)
    cmap_records = []
    for name, cmap, ctor in _colormap_family():
        rel = f"_colormaps/{name}.png"
        _render_portrait_mpl(ref.domain(), ref.func, cmap, GALLERY_DIR / rel)
        cmap_records.append({"name": name, "file": rel, "ctor": ctor})

    # 4. Presentation manifest (split from index.json).
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "complexplorer_version": __version__,
        "generator": "complexplorer showcase",
        "banner": HERO_BANNER,
        "colormaps": {"reference_preset": COLORMAP_REFERENCE, "renders": cmap_records},
        "presets": preset_records,
    }
    _write_json(GALLERY_DIR / "showcase.json", manifest)

    # 5. Generated docs gallery page.
    _generate_docs_page(manifest)
    print(f"Done. {len(preset_records)} presets, {len(cmap_records)} colormaps -> {GALLERY_DIR}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=["all", "colormaps"],
        default="all",
        help="regenerate only one section (default: everything)",
    )
    main(**vars(parser.parse_args()))
