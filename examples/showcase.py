#!/usr/bin/env python3
"""examples/showcase.py — the high-res visual gallery producer.

Renders the curated preset registry (``cp.catalog``) into the committed visual gallery:
the 2D portraits + deterministic ``index.json`` (via the library ``cp.gallery``), plus the
PyVista 3D screenshots that manifest deliberately omits, a colormap gallery, thumbnails, a
presentation manifest (``showcase.json``), and a generated docs gallery page.

The render set per preset follows the preset's TAGS (which encode mathematical character):

    every preset      -> portrait.png   (2D, from cp.gallery)
    canonical         -> landscape.png + sphere.png
    branches          -> surface.png    (riemann_surface_pv)
    ornament          -> ornament.png   (relief sphere)

Every render takes its look from ``RENDER_PROFILES`` — one locked profile per family, so the
committed gallery shares a single camera, ground, lighting and mesh resolution (the outcome of
visual-review rounds R0/R0b; see openspec/REV3_CLOSEOUT.md).

This is a LOCAL regeneration tool — off-screen VTK screenshots crash only on headless CI.
Run it from the repo root:

    python examples/showcase.py                      # everything, into examples/gallery
    python examples/showcase.py --only colormaps     # one section
    python examples/showcase.py --out /tmp/round3    # stage a review round; repo untouched

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

import tour

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# ---------------------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
GALLERY_DIR = REPO_ROOT / "examples" / "gallery"
DOCS_GALLERY = REPO_ROOT / "docs" / "gallery"

SCHEMA_VERSION = 2  # v2: adds the `tour` and `curated` sections (independent of index.json)
HERO_BANNER = "Riemann_relief_map_20250726.png"  # curated, kept (not regenerated)

PORTRAIT_DPI = 390  # 2D portraits / colormaps (figsize 4x4 -> ~1560px)
THUMB_PX = 400      # gallery thumbnails; PNG for README/PyPI compatibility

COLORMAP_REFERENCE = "rational_zeros_poles"  # zeros at +-1, poles at +-i — good contrast

# The house style, decided in visual-review round R0 ("Gallery grey"), with the mesh
# resolutions from round R0b. One entry per render family; every render reads its look from
# here, so a restyle happens in one place and the committed images cannot drift apart.
GROUND = ("#e6e9ee", "#fbfcfd")  # (bottom, top) of the vertical gradient

RENDER_PROFILES = {
    "portrait": {
        "kind": "matplotlib",
        "dpi": PORTRAIT_DPI,
        "figsize": (4.0, 4.0),
        "resolution": 1200,  # samples per axis; 400 upscaled and stair-stepped the bands
        "tight": True,  # keeps the Im(z) label inside the image
    },
    "landscape": {
        "kind": "pyvista",
        "window": (1560, 1560),
        "resolution": 600,
        "ground": GROUND,
        "lighting": "three",
        "specular": 0.3,
        "anti_aliasing": "ssaa",
        "orientation_widget": False,
        # The refit already frames the domain tightly (about a 1% margin), so anything above
        # 1.0 clips the corners. This leaves a small, even border.
        "zoom": 0.95,
    },
    "sphere": {
        "kind": "pyvista",
        "window": (1560, 1560),
        "resolution": 1000,
        "ground": GROUND,
        "lighting": "three",
        "specular": 0.3,
        "anti_aliasing": "ssaa",
        "orientation_widget": False,
        "zoom": 1.1,
    },
    "ornament": {
        "kind": "pyvista",
        "window": (1560, 1560),
        "resolution": 800,
        "ground": GROUND,
        "lighting": "three",
        "specular": 0.3,
        "anti_aliasing": "ssaa",
        "orientation_widget": False,
        "zoom": 1.1,
    },
    "surface": {
        "kind": "pyvista",
        "window": (1560, 1560),
        "resolution": 312,
        "ground": GROUND,
        "lighting": "three",
        "specular": 0.3,
        "anti_aliasing": "ssaa",
        "orientation_widget": False,
        "zoom": 1.1,
    },
}


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
    # The helicoid's height is 2*pi*turns (about 19), so at the default r_max=1.5 it is six
    # times taller than it is wide and reads as a thin ribbon. Double the radius: wider than
    # this and the upper sheets simply occlude the lower ones.
    "log": ("log", {"r_max": 3.0}),
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


def _style(plotter, profile: dict) -> None:
    """Apply a family profile to a returned (off-screen) plotter.

    The renderers expose camera, window and orientation but not ground or lighting, so the
    gallery's look is applied here rather than widening the library's signatures.
    """
    bottom, top = profile["ground"]
    plotter.set_background(bottom, top=top)
    if profile["lighting"] == "three":
        plotter.remove_all_lights()
        plotter.enable_3_lights()
    for actor in plotter.renderer.actors.values():
        prop = getattr(actor, "prop", None)
        if prop is not None and hasattr(prop, "specular"):
            prop.specular = profile["specular"]
    if profile["anti_aliasing"]:
        try:
            plotter.enable_anti_aliasing(profile["anti_aliasing"])
        except Exception as exc:  # pragma: no cover - driver dependent
            print(f"    (anti-aliasing unavailable: {exc})")
    # Fit the camera to the subject first, then zoom. Without this each family framed its
    # subject differently: the landscapes sat small inside wide empty margins.
    plotter.reset_camera()
    if profile["zoom"] != 1.0:
        plotter.camera.zoom(profile["zoom"])


def _shoot(plotter, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(path))
    plotter.close()


def _thumbnail(path: Path, gallery_dir: Path) -> str:
    """Write a THUMB_PX-wide PNG mirroring the render's path under thumb/."""
    rel = path.relative_to(gallery_dir)
    dst = gallery_dir / "thumb" / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(path) as img:
        thumb = img.copy()
    thumb.thumbnail((THUMB_PX, THUMB_PX), Image.LANCZOS)
    thumb.save(dst)
    return str(Path("thumb") / rel).replace("\\", "/")


def _render_portrait_mpl(
    domain, func, cmap, path: Path, legend: bool = False, dpi: int | None = None
) -> None:
    """A 2D portrait via matplotlib (used for the colormap gallery and the tour).

    ``dpi`` overrides the profile so a panel of a composed figure can be rendered at exactly
    the size it will occupy: downscaling a finished panel is what made its text blurry.
    """
    profile = RENDER_PROFILES["portrait"]
    fig, ax = plt.subplots(figsize=profile["figsize"])
    try:
        plot_2d(
            domain, func, cmap=cmap, ax=ax, legend=legend, resolution=profile["resolution"]
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            path,
            dpi=dpi or profile["dpi"],
            metadata={"Software": None},
            bbox_inches="tight" if profile["tight"] else None,
            pad_inches=0.05,
        )
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------------------
# Per-preset 3D renders
# ---------------------------------------------------------------------------------------

def _render_landscape(preset, path: Path) -> None:
    profile = RENDER_PROFILES["landscape"]
    sc = preset.scaling()
    plotter = cp.plot_landscape_pv(
        preset.domain(), preset.func, cmap=preset.colormap(),
        modulus_mode=sc["method"], modulus_params=sc["params"],
        resolution=profile["resolution"], window_size=profile["window"],
        interactive=False, return_plotter=True,
        show_orientation=profile["orientation_widget"],
    )
    _style(plotter, profile)
    _shoot(plotter, path)


def _render_sphere(preset, path: Path) -> None:
    # domain=None -> no stereographic mask -> the FULL sphere (both poles), no "cup".
    profile = RENDER_PROFILES["sphere"]
    plotter = cp.riemann_pv(
        preset.func, cmap=preset.colormap(), modulus_mode="constant",
        resolution=profile["resolution"], window_size=profile["window"],
        interactive=False, return_plotter=True,
        show_orientation=profile["orientation_widget"],
    )
    _style(plotter, profile)
    _shoot(plotter, path)


def _render_ornament(preset, path: Path) -> None:
    # domain=None -> full sphere relief (the infinity pole is included, not cut off).
    profile = RENDER_PROFILES["ornament"]
    sc = preset.scaling()
    plotter = cp.riemann_pv(
        preset.func, cmap=preset.colormap(),
        modulus_mode=sc["method"], modulus_params=sc["params"],
        resolution=profile["resolution"], window_size=profile["window"],
        interactive=False, return_plotter=True,
        show_orientation=profile["orientation_widget"],
    )
    _style(plotter, profile)
    _shoot(plotter, path)


# The log surface is a tall spiral ramp. Seen from the default camera its turns are edge-on
# and read as disconnected crescents; from straight above they occlude each other. This angle
# shows the ramp turning through all three levels.
SURFACE_CAMERA = {"log": (2.5, 2.5, 3.5)}


def _render_surface_family(family: str, kw: dict, path: Path) -> None:
    profile = RENDER_PROFILES["surface"]
    plotter = cp.riemann_surface_pv(
        family, **kw,
        resolution=profile["resolution"], window_size=profile["window"],
        interactive=False, return_plotter=True,
        show_orientation=profile["orientation_widget"],
        camera_position=SURFACE_CAMERA.get(family, (2.5, 2.5, 2.5)),
    )
    _style(plotter, profile)
    _shoot(plotter, path)


def _render_surface(preset, path: Path) -> None:
    family, kw = SURFACE_FAMILY[preset.id]
    _render_surface_family(family, kw, path)


def _render_callable(family: str, func, domain, path: Path, modulus_mode: str = "none") -> None:
    """Render an arbitrary callable (not a catalog preset) in a family's profile."""
    profile = RENDER_PROFILES[family]
    plotter = cp.plot_landscape_pv(
        domain, func, modulus_mode=modulus_mode,
        resolution=profile["resolution"], window_size=profile["window"],
        interactive=False, return_plotter=True,
        show_orientation=profile["orientation_widget"],
    )
    _style(plotter, profile)
    _shoot(plotter, path)


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

# Sections are ordered by idea, not by registry id. A preset lands in the first section whose
# tag it carries, so each one appears exactly once.
PAGE_SECTIONS = [
    ("phase-portraits", "Phase portraits", None,
     "Hue is the phase of f(z); the shaded cells are contour bands of |f(z)|. Zeros and poles "
     "read as opposite winding directions."),
    ("mapping-and-topology", "Mapping and topology", "canonical",
     "The same functions lifted off the plane: magnitude as height, and the sphere that "
     "compactifies the plane so infinity has a place to sit."),
    ("riemann-surfaces", "Riemann surfaces", "branches",
     "Multivalued families become single-valued on their covering surface. Branch points and "
     "cuts are geometry here, not bookkeeping."),
    ("engineering", "Engineering mode", None,
     "A transfer function is a complex function, so the whole library applies to it."),
    ("colormaps", "Colormaps", None,
     "One reference function under every colormap the package exports."),
    ("physical-output", "Physical output", "ornament",
     "Modulus-scaled relief, exported as a watertight mesh and printed."),
]

_TAG_SECTION = {"branches": "riemann-surfaces", "ornament": "physical-output",
                "canonical": "mapping-and-topology"}

RENDER_ALT = {
    "portrait": "2D phase portrait",
    "landscape": "3D analytic landscape",
    "sphere": "Riemann sphere",
    "ornament": "Riemann relief (ornament)",
    "surface": "Riemann surface",
}


def _section_of(preset) -> str:
    for tag in ("branches", "ornament", "canonical"):
        if tag in preset.tags:
            return _TAG_SECTION[tag]
    return "phase-portraits"


def _slug(title: str) -> str:
    """GitHub's heading anchor: lowercased, spaces to hyphens."""
    return title.lower().replace(" ", "-")


def _figure(full: str, thumb: str | None, alt: str, width: int = 420) -> str:
    """A figure that links to the full-resolution render.

    The thumbnail is only used when it is at least as wide as the display size; above that the
    full render is served and the browser scales it down, which stays sharp.
    """
    src = thumb if (thumb and width <= THUMB_PX) else full
    return (
        f'<a href="../../examples/gallery/{full}">'
        f'<img src="../../examples/gallery/{src}" alt="{alt}" width="{width}"></a>'
    )


def _entry(title: str, caption: str, figures: list[str], snippet: str) -> str:
    lines = [f"### {title}", ""]
    if caption:
        lines += [caption, ""]
    lines += ["<p>" + " ".join(figures) + "</p>", ""]
    if snippet:
        lines += ["<details>", "<summary>Show the code</summary>", "", "```python", snippet,
                  "```", "", "</details>", ""]
    return "\n".join(lines)


def _generate_docs_page(manifest: dict, docs_dir: Path) -> None:
    hero = manifest.get("hero") or []
    tour_by_section: dict[str, list[dict]] = {}
    for rec in manifest.get("tour", []):
        key = rec["section"].replace(" ", "-").replace("and-", "and-")
        key = {"phase-portraits": "phase-portraits", "mapping-and-topology": "mapping-and-topology",
               "riemann-surfaces": "riemann-surfaces", "engineering": "engineering",
               "physical-output": "physical-output"}.get(key, key)
        tour_by_section.setdefault(key, []).append(rec)

    presets_by_section: dict[str, list[dict]] = {}
    for rec in manifest["presets"]:
        presets_by_section.setdefault(_section_of(catalog.get(rec["id"])), []).append(rec)

    out = [
        "<!-- GENERATED by examples/showcase.py — do not edit by hand.",
        "     Regenerate with:  python examples/showcase.py  -->",
        "",
        "# Gallery",
        "",
        "Every image here is produced by `examples/showcase.py` from the curated preset registry",
        "(`cp.catalog`) and the tour recipes in `examples/tour.py`. Thumbnails link to the",
        "full-resolution render.",
        "",
    ]

    if hero:
        pick = next((h for h in hero if h["id"].endswith("labels_on_panel")), hero[0])
        out += [
            _figure(pick["file"], pick.get("thumb"),
                    "Six-panel montage: domain coloring, analytic landscape, Riemann relief, "
                    "Riemann surface, transfer functions and a 3D-printable ornament", 900),
            "",
        ]

    out += ["## What is here", ""]
    for key, title, _tag, _blurb in PAGE_SECTIONS:
        count = len(presets_by_section.get(key, [])) + len(tour_by_section.get(key, []))
        if key == "colormaps":
            count = len(manifest["colormaps"]["renders"])
        out.append(f"- [{title}](#{_slug(title)}) — {count} figures")
    out.append("")

    for key, title, _tag, blurb in PAGE_SECTIONS:
        entries = []
        for rec in tour_by_section.get(key, []):
            entries.append(
                _entry(rec["title"], rec["caption"],
                       [_figure(rec["file"], rec.get("thumb"), rec["alt"], 620)], rec["snippet"])
            )
        if key == "colormaps":
            ref = catalog.get(manifest["colormaps"]["reference_preset"])
            figures = [
                _figure(r["file"], r.get("thumb"), f"{ref.expression} rendered with {r['ctor']}", 260)
                for r in manifest["colormaps"]["renders"]
            ]
            entries.append(
                _entry(f"Every colormap on {ref.expression}",
                       "The same function under each colormap the package exports.",
                       figures,
                       _colormap_snippet(manifest["colormaps"]["renders"][0]["ctor"]))
            )
        for rec in presets_by_section.get(key, []):
            preset = catalog.get(rec["id"])
            figures = [
                _figure(rel, (rec.get("thumbs") or {}).get(rtype),
                        f"{RENDER_ALT.get(rtype, rtype)} of {preset.expression}")
                for rtype, rel in rec["renders"].items()
            ]
            snippet = "\n\n# ---\n".join(_snippet(preset, rt) for rt in rec["renders"])
            entries.append(_entry(rec["title"], rec.get("story", ""), figures, snippet))

        if not entries:
            continue
        out += [f"## {title}", "", blurb, ""] + entries

    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "gallery.generated.md").write_text("\n".join(out), encoding="utf-8", newline="\n")


# ---------------------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------------------

def _existing_manifest(gallery_dir: Path) -> dict:
    path = gallery_dir / "showcase.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _render_presets(gallery_dir: Path) -> list[dict]:
    """Deterministic 2D portraits + index.json, then the PyVista screenshots by tag policy."""
    print("Rendering 2D portraits + index.json (cp.gallery) ...")
    generate_gallery(
        gallery_dir,
        selection=None,
        dpi=PORTRAIT_DPI,
        resolution=RENDER_PROFILES["portrait"]["resolution"],
    )

    records = []
    for pid in catalog.list():
        preset = catalog.get(pid)
        renders = {"portrait": f"{pid}/portrait.png"}
        used = {"portrait": "portrait"}
        for rtype in _renders_for(preset):
            if rtype == "portrait":
                continue
            rel = f"{pid}/{rtype}.png"
            print(f"  {pid}: {rtype}")
            _RENDERERS[rtype](preset, gallery_dir / rel)
            renders[rtype] = rel
            used[rtype] = rtype
        records.append(
            {
                "id": pid,
                "title": preset.title,
                "expression": preset.expression,
                "story": preset.story,
                "tags": list(preset.tags),
                "renders": renders,
                "thumbs": {
                    rtype: _thumbnail(gallery_dir / rel, gallery_dir)
                    for rtype, rel in renders.items()
                },
                "profiles": {
                    rtype: {
                        "profile": name,
                        "resolution": RENDER_PROFILES[name].get("resolution"),
                    }
                    for rtype, name in used.items()
                },
            }
        )
    return records


def _render_colormaps(gallery_dir: Path) -> list[dict]:
    print("Rendering colormap gallery ...")
    ref = catalog.get(COLORMAP_REFERENCE)
    records = []
    for name, cmap, ctor in _colormap_family():
        rel = f"_colormaps/{name}.png"
        _render_portrait_mpl(ref.domain(), ref.func, cmap, gallery_dir / rel)
        records.append(
            {
                "name": name,
                "file": rel,
                "ctor": ctor,
                "thumb": _thumbnail(gallery_dir / rel, gallery_dir),
            }
        )
    return records


def _tour_context(gallery_dir: Path, curated: list[dict]) -> dict:
    """Everything examples/tour.py needs to render in the gallery's locked profiles."""
    photo = next(
        (gallery_dir / c["file"] for c in curated if c["kind"] == "photograph"),
        None,
    )
    return {
        "profiles": RENDER_PROFILES,
        "style": _style,
        "shoot": _shoot,
        "thumbnail": _thumbnail,
        "portrait_mpl": _render_portrait_mpl,
        "render_family": lambda family, preset, path: _RENDERERS[family](preset, path),
        "render_callable": _render_callable,
        "render_surface": _render_surface_family,
        "photo": photo,
    }


# Curated assets are committed but never regenerated; the kind decides how they are used
# (only a `photograph` stands in for the physical-output panel).
CURATED_KINDS = {
    "printed_ornament.png": (
        "photograph",
        "supplied by the author: the printed pole-flower ornament (z / (z**10 - 1))",
    ),
    "relief_and_print.jpg": (
        "composite",
        "supplied by the author: the Riemann relief beside the printed ornament, captioned "
        "with the function",
    ),
}


def _curated_records(gallery_dir: Path) -> list[dict]:
    """Assets that are committed but never regenerated: the banner and printed-object photos.

    Each entry records its kind and origin and carries no render recipe, so a photograph can
    never be mistaken for — or overwritten by — a render.
    """
    records = [
        {
            "file": HERO_BANNER,
            "kind": "banner",
            "origin": "hand-composed relief render, kept from the 2.x gallery",
        }
    ]
    photo_dir = gallery_dir / "_curated"
    if photo_dir.is_dir():
        for asset in sorted(photo_dir.glob("*")):
            if asset.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            kind, origin = CURATED_KINDS.get(
                asset.name,
                ("photograph", "supplied by the author; a physical object, not a render"),
            )
            records.append(
                {
                    "file": f"_curated/{asset.name}",
                    "kind": kind,
                    "origin": origin,
                    "thumb": _thumbnail(asset, gallery_dir),
                }
            )
    return records


# ---------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------

SECTIONS = ("presets", "colormaps", "tour", "hero", "thumbs")


def main(only: str = "all", out: str | None = None) -> None:
    gallery_dir = Path(out).resolve() if out else GALLERY_DIR
    docs_dir = (gallery_dir / "docs") if out else DOCS_GALLERY
    gallery_dir.mkdir(parents=True, exist_ok=True)

    # Sections that are not re-rendered keep the records they already have.
    previous = _existing_manifest(GALLERY_DIR if out else gallery_dir)

    if only in ("all", "presets"):
        preset_records = _render_presets(gallery_dir)
    else:
        preset_records = previous.get("presets", [])

    if only in ("all", "colormaps"):
        colormap_records = _render_colormaps(gallery_dir)
    else:
        colormap_records = previous.get("colormaps", {}).get("renders", [])

    curated = _curated_records(gallery_dir)
    ctx = _tour_context(gallery_dir, curated)
    if only in ("all", "tour"):
        print("Rendering the curated tour ...")
        tour_records = tour.render_tour(gallery_dir, ctx)
    else:
        tour_records = previous.get("tour", [])

    if only in ("all", "hero"):
        print("Rendering the hero montage ...")
        hero_records = tour.render_hero(gallery_dir, ctx)
    else:
        hero_records = previous.get("hero", [])

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "complexplorer_version": __version__,
        "generator": "complexplorer showcase",
        "banner": HERO_BANNER,  # kept until the docs page stops reading it
        "curated": curated,
        "colormaps": {"reference_preset": COLORMAP_REFERENCE, "renders": colormap_records},
        "presets": preset_records,
        "tour": tour_records,
        "hero": hero_records,
    }
    _write_json(gallery_dir / "showcase.json", manifest)
    _generate_docs_page(manifest, docs_dir)
    print(
        f"Done. {len(preset_records)} presets, {len(colormap_records)} colormaps, "
        f"{len(tour_records)} tour, {len(hero_records)} hero, {len(curated)} curated "
        f"-> {gallery_dir}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render the complexplorer visual gallery.")
    parser.add_argument(
        "--only",
        choices=["all", *SECTIONS],
        default="all",
        help="regenerate only one section (default: everything)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="write the bundle here instead of examples/gallery (for staging a review round)",
    )
    main(**vars(parser.parse_args()))
