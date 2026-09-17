"""examples/tour.py — the curated tour section of the visual gallery.

The preset registry drives most of the gallery, but several rev3 capabilities are not catalog
presets: the phase-wheel legend, engineering mode, composite domains, the sphere/surface
distinction, and the path from an STL to a printed object. Each of those is a *recipe* here:
what to render, how to frame it, what it shows, and the snippet that reproduces it.

``examples/showcase.py`` owns the render profiles and calls :func:`render_tour`.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

import complexplorer as cp
from complexplorer.core.presets import catalog
from complexplorer.export.stl import OrnamentGenerator

# The reference function for the 2D -> 3D pair: zeros at +-1, poles at +-i.
REFERENCE = "rational_zeros_poles"

# A notch filter: zeros sit exactly on the jw axis at +-2j, poles just inside the left
# half-plane, so one portrait shows the notch, the resonance and the stability margin.
NOTCH_NUM = [1.0, 0.0, 4.0]
NOTCH_DEN = [1.0, 1.2, 5.0, 2.0]

PANEL_BG = (251, 252, 253)
PANEL_INK = (26, 30, 36)

# Every panel is rendered AT this size and pasted 1:1. Scaling a finished panel down is what
# blurred the axis labels in the first pass of round R1, so nothing is resampled here.
CELL = 1200
PANEL_FIGSIZE = (5.0, 5.0)

# The hero is displayed about 830px wide on GitHub, so ~1900px across is sharp on a high-dpi
# screen without being a heavy download. Text-bearing panels are rendered AT the cell size;
# the PyVista panels are rendered large and shrunk, which only improves them.
HERO_CELL = 620


# ---------------------------------------------------------------------------------------
# Composition helpers
# ---------------------------------------------------------------------------------------


def _font(size: int) -> ImageFont.FreeTypeFont:
    """A real font, taken from matplotlib's bundled DejaVu so this works on any platform."""
    try:
        return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)
    except Exception:  # pragma: no cover - only if matplotlib ships no TTF
        return ImageFont.load_default()


def _wrap(draw, text: str, font, max_width: int) -> list[str]:
    """Greedy wrap so a caption stays inside its own cell."""
    words, lines, line = text.split(), [], ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if draw.textlength(candidate, font=font) <= max_width or not line:
            line = candidate
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def _trim(path: Path, margin: int = 12, tol: int = 12) -> None:
    """Crop a render down to its subject, so panels do not carry unequal empty margins.

    The ground is a vertical GRADIENT, so each row's background is taken from that row's own
    edge pixels; comparing against a single corner colour would flag the gradient itself as
    content and trim nothing.
    """
    import numpy as np

    with Image.open(path) as raw:
        img = raw.convert("RGB")
    arr = np.asarray(img).astype(int)
    edges = np.concatenate([arr[:, :3, :], arr[:, -3:, :]], axis=1)
    row_bg = np.median(edges, axis=1)[:, None, :]
    mask = np.abs(arr - row_bg).max(axis=2) > tol
    ys, xs = np.where(mask)
    if not len(ys):
        return
    img.crop((
        max(0, int(xs.min()) - margin),
        max(0, int(ys.min()) - margin),
        min(img.width, int(xs.max()) + 1 + margin),
        min(img.height, int(ys.max()) + 1 + margin),
    )).save(path)


def _mpl_panel(draw, path: Path, *, figsize=PANEL_FIGSIZE, cell: int = CELL) -> None:
    """Render one matplotlib panel at the pixel size it will occupy in the composition."""
    fig, ax = plt.subplots(figsize=figsize)
    try:
        draw(ax)
        fig.savefig(path, dpi=cell / figsize[0], bbox_inches="tight", pad_inches=0.06)
    finally:
        plt.close(fig)


def _compose(panels: list[tuple[Path, str]], out: Path, *, columns: int, cell: int = CELL) -> None:
    """Lay labelled panels out on a grid at native size (panels are never upscaled)."""
    font = _font(max(20, round(cell * 0.030)))
    line_h = round(font.size * 1.25)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    wrapped = [_wrap(probe, label, font, cell - 8) for _, label in panels]
    label_h = 16 + line_h * max(len(w) for w in wrapped)
    pad = 28
    rows = (len(panels) + columns - 1) // columns
    width = columns * cell + pad * (columns + 1)
    height = rows * (cell + label_h) + pad * (rows + 1)
    sheet = Image.new("RGB", (width, height), PANEL_BG)
    draw = ImageDraw.Draw(sheet)

    for index, (path, _label) in enumerate(panels):
        row, col = divmod(index, columns)
        x = pad + col * (cell + pad)
        y = pad + row * (cell + label_h + pad)
        with Image.open(path) as raw:
            img = raw.convert("RGB")
        if img.width > cell or img.height > cell:  # only ever shrink an oversized panel
            img.thumbnail((cell, cell), Image.LANCZOS)
        sheet.paste(img, (x + (cell - img.width) // 2, y + (cell - img.height) // 2))
        for line_no, line in enumerate(wrapped[index]):
            draw.text((x + 4, y + cell + 8 + line_no * line_h), line, fill=PANEL_INK, font=font)

    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


def _stack_wide(grid: Path, wide: Path, label: str, out: Path) -> None:
    """Put one already-correctly-sized wide panel under a grid, with its caption."""
    font = _font(max(20, round(CELL * 0.030)))
    with Image.open(grid) as top_img, Image.open(wide) as bottom_raw:
        top = top_img.convert("RGB")
        bottom = bottom_raw.convert("RGB")
        label_h = 16 + round(font.size * 1.25)
        width = max(top.width, bottom.width)
        sheet = Image.new("RGB", (width, top.height + bottom.height + label_h), PANEL_BG)
        sheet.paste(top, ((width - top.width) // 2, 0))
        sheet.paste(bottom, ((width - bottom.width) // 2, top.height))
        ImageDraw.Draw(sheet).text(
            (28, top.height + bottom.height + 8), label, fill=PANEL_INK, font=font
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(out)


# ---------------------------------------------------------------------------------------
# Recipes
# ---------------------------------------------------------------------------------------


def _legend_portrait(ctx, out: Path) -> None:
    preset = catalog.get(REFERENCE)
    ctx["portrait_mpl"](preset.domain(), preset.func, preset.colormap(), out, legend=True)


def _portrait_to_landscape(ctx, out: Path) -> None:
    preset = catalog.get(REFERENCE)
    with tempfile.TemporaryDirectory() as tmp:
        flat = Path(tmp) / "flat.png"
        relief = Path(tmp) / "relief.png"
        ctx["portrait_mpl"](
            preset.domain(), preset.func, preset.colormap(), flat, legend=True, dpi=CELL / 4
        )
        ctx["render_family"]("landscape", preset, relief)
        _compose(
            [(flat, "2D phase portrait — cp.plot(..., legend=True)"),
             (relief, "the same function as a landscape — cp.plot_landscape_pv(...)")],
            out,
            columns=2,
        )


def _engineering_figure(ctx, out: Path) -> None:
    H = cp.ee.TransferFunction(NOTCH_NUM, NOTCH_DEN)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        portrait, poles, nyquist, bode = (
            tmp_dir / "portrait.png",
            tmp_dir / "poles.png",
            tmp_dir / "nyquist.png",
            tmp_dir / "bode.png",
        )
        for path, draw in (
            (portrait, lambda ax: cp.ee.transfer_portrait(H, ax=ax, legend=True)),
            (poles, lambda ax: cp.ee.pole_zero_plot(H, ax=ax)),
            (nyquist, lambda ax: cp.ee.nyquist_plot(H, ax=ax)),
        ):
            _mpl_panel(draw, path)

        grid = tmp_dir / "grid.png"
        _compose(
            [
                (portrait, "transfer portrait — poles, zeros and the jw axis"),
                (poles, "pole-zero map"),
                (nyquist, "Nyquist"),
            ],
            grid,
            columns=3,
        )

        # Bode owns its figure, so render it at the grid's width rather than upscaling into it.
        fig = cp.ee.bode_plot(H)
        try:
            with Image.open(grid) as grid_img:
                target_width = grid_img.width
            fig.savefig(
                bode,
                dpi=target_width / fig.get_size_inches()[0],
                bbox_inches="tight",
                pad_inches=0.06,
            )
        finally:
            plt.close(fig)

        _stack_wide(grid, bode, "Bode — magnitude and phase", out)


def _composition_proof(ctx, out: Path) -> None:
    H = cp.ee.TransferFunction(NOTCH_NUM, NOTCH_DEN)
    with tempfile.TemporaryDirectory() as tmp:
        flat = Path(tmp) / "flat.png"
        relief = Path(tmp) / "relief.png"
        _mpl_panel(lambda ax: cp.ee.transfer_portrait(H, ax=ax, legend=True), flat)
        ctx["render_callable"]("landscape", H, cp.Rectangle(6, 6), relief, modulus_mode="arctan")
        _compose(
            [(flat, "cp.ee.transfer_portrait(H) — the engineering view"),
             (relief, "cp.plot_landscape_pv(domain, H) — the same object, a general renderer")],
            out,
            columns=2,
        )


def _composite_domain(ctx, out: Path) -> None:
    # Two overlapping disks unioned into a peanut, with a hole punched out around the pole:
    # the silhouette itself shows that the domain was composed, not drawn.
    left = cp.Disk(1.1, center=-0.75 + 0j)
    right = cp.Disk(1.1, center=0.75 + 0j)
    domain = (left | right) - cp.Disk(0.4)
    ctx["portrait_mpl"](
        domain, lambda z: 1 / z, cp.Phase(phase_sectors=12, auto_scale_r=True), out, legend=True
    )


def _sphere_vs_surface(ctx, out: Path) -> None:
    preset = catalog.get("reciprocal")
    with tempfile.TemporaryDirectory() as tmp:
        sphere = Path(tmp) / "sphere.png"
        surface = Path(tmp) / "surface.png"
        ctx["render_family"]("sphere", preset, sphere)
        ctx["render_surface"]("power", {"n": 2}, surface)
        _compose(
            [(sphere, "Riemann SPHERE — one single-valued function, including the point at infinity"),
             (surface, "Riemann SURFACE — the two-sheeted cover on which sqrt(z) is single-valued")],
            out,
            columns=2,
        )


def _clay_mesh(ctx, mesh, out: Path) -> None:
    """Render an STL mesh as an untextured clay object: the geometry, not the colours."""
    import pyvista as pv

    profile = ctx["profiles"]["ornament"]
    plotter = pv.Plotter(off_screen=True, window_size=profile["window"])
    plotter.add_mesh(mesh, color="#c9c4bb", smooth_shading=True, specular=0.2, specular_power=12)
    plotter.camera_position = "iso"
    ctx["style"](plotter, profile)
    ctx["shoot"](plotter, out)


def _physical_triptych(ctx, out: Path) -> None:
    preset = catalog.get("pole_flower_10")
    scaling = preset.scaling()
    with tempfile.TemporaryDirectory() as tmp:
        relief = Path(tmp) / "relief.png"
        mesh_png = Path(tmp) / "mesh.png"
        ctx["render_family"]("ornament", preset, relief)

        generator = OrnamentGenerator(
            preset.func,
            resolution=150,
            scaling=scaling["method"],
            scaling_params=scaling["params"],
            cmap=preset.colormap(),
        )
        _clay_mesh(ctx, generator.generate_ornament(), mesh_png)

        panels = [
            (relief, "the relief render — cp.riemann_pv(..., modulus_mode=...)"),
            (mesh_png, "the STL mesh — OrnamentGenerator(...).generate_ornament()"),
        ]
        photo = ctx.get("photo")
        panels.append(
            (photo, "the printed object")
            if photo
            else (mesh_png, "the printed object — photograph pending")
        )
        _compose(panels, out, columns=3)


def _orbit_loop(ctx, out: Path) -> None:
    """A short rotating loop: a still cannot show depth, rotation or lighting."""
    profile = ctx["profiles"]["ornament"]
    preset = catalog.get("pole_flower_10")
    scaling = preset.scaling()
    plotter = cp.riemann_pv(
        preset.func,
        cmap=preset.colormap(),
        modulus_mode=scaling["method"],
        modulus_params=scaling["params"],
        resolution=400,  # lower than the still: 36 frames, and the GIF has a size budget
        window_size=(700, 700),
        interactive=False,
        return_plotter=True,
        show_orientation=False,
    )
    ctx["style"](plotter, profile)
    out.parent.mkdir(parents=True, exist_ok=True)
    plotter.open_gif(str(out), fps=18)
    # shift lifts the camera above the focal plane; at 0.0 the loop opens edge-on.
    path = plotter.generate_orbital_path(n_points=36, shift=1.1, factor=2.4)
    plotter.orbit_on_path(path, write_frames=True, step=0.0, progress_bar=False)
    plotter.close()


# ---------------------------------------------------------------------------------------
# The hero montage
# ---------------------------------------------------------------------------------------

# Chosen in visual-review round R2: captions sit on the sheet under each panel, leaving the art
# uninterrupted. `_compose_overlay` keeps the alternative (a band on each panel) available.
HERO_VARIANT = "labels_below"

HERO_PANELS = [
    ("domain coloring", "portrait"),
    ("analytic landscape", "landscape"),
    ("Riemann relief", "relief"),
    ("Riemann surface", "surface"),
    ("transfer functions", "engineering"),
    ("3D-printable", "physical"),
]


def _hero_panels(ctx, tmp_dir: Path) -> dict[str, Path]:
    """Render the six hero panels, each at the size it will occupy."""
    reference = catalog.get(REFERENCE)
    flower = catalog.get("pole_flower_10")
    paths = {kind: tmp_dir / f"{kind}.png" for _, kind in HERO_PANELS}

    # text-bearing panels: rendered natively so nothing shrinks their labels
    ctx["portrait_mpl"](
        reference.domain(), reference.func, reference.colormap(),
        paths["portrait"], legend=True, dpi=HERO_CELL / 4,
    )
    H = cp.ee.TransferFunction(NOTCH_NUM, NOTCH_DEN)

    def _transfer_no_title(ax):
        cp.ee.transfer_portrait(H, ax=ax, legend=True)
        ax.set_title("")  # the montage supplies the label

    _mpl_panel(_transfer_no_title, paths["engineering"], cell=HERO_CELL)

    # 3D panels: rendered at full profile size, shrunk by the composition
    ctx["render_family"]("landscape", reference, paths["landscape"])
    ctx["render_family"]("ornament", flower, paths["relief"])
    ctx["render_surface"]("power", {"n": 2}, paths["surface"])

    for kind in ("landscape", "relief", "surface"):
        _trim(paths[kind])

    photo = ctx.get("photo")
    if photo:
        paths["physical"] = Path(photo)
    else:
        scaling = flower.scaling()
        generator = OrnamentGenerator(
            flower.func,
            resolution=150,
            scaling=scaling["method"],
            scaling_params=scaling["params"],
            cmap=flower.colormap(),
        )
        _clay_mesh(ctx, generator.generate_ornament(), paths["physical"])
        _trim(paths["physical"])
    return paths


def _compose_overlay(panels: list[tuple[Path, str]], out: Path, *, columns: int) -> None:
    """Variant B: labels sit on the panel itself, so the grid reads as one image."""
    cell, pad = HERO_CELL, 10
    font = _font(23)
    rows = (len(panels) + columns - 1) // columns
    sheet = Image.new(
        "RGB", (columns * cell + pad * (columns + 1), rows * cell + pad * (rows + 1)), PANEL_BG
    )
    draw = ImageDraw.Draw(sheet)
    for index, (path, label) in enumerate(panels):
        row, col = divmod(index, columns)
        x, y = pad + col * (cell + pad), pad + row * (cell + pad)
        with Image.open(path) as raw:
            img = raw.convert("RGB")
        if img.width > cell or img.height > cell:
            img.thumbnail((cell, cell), Image.LANCZOS)
        band = font.size + 18
        sheet.paste(img, (x + (cell - img.width) // 2, y + (cell - band - img.height) // 2))
        draw.rectangle([x, y + cell - band, x + cell, y + cell], fill=(238, 241, 245))
        draw.text((x + 12, y + cell - band + 8), label, fill=PANEL_INK, font=font)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


def render_hero(
    gallery_dir: Path, ctx: dict, variants: tuple[str, ...] = (HERO_VARIANT,)
) -> list[dict]:
    """The hero montage in the layout chosen in round R2 (or any variant asked for)."""
    layouts = {"labels_below": _compose, "labels_on_panel": _compose_overlay}
    records = []
    with tempfile.TemporaryDirectory() as tmp:
        paths = _hero_panels(ctx, Path(tmp))
        panels = [(paths[kind], label) for label, kind in HERO_PANELS]
        for variant in variants:
            compose = layouts[variant]
            rel = f"_tour/hero_{variant}.png"
            print(f"  hero: {variant}")
            if compose is _compose:
                compose(panels, gallery_dir / rel, columns=3, cell=HERO_CELL)
            else:
                compose(panels, gallery_dir / rel, columns=3)
            with Image.open(gallery_dir / rel) as img:
                size = list(img.size)
            records.append(
                {
                    "id": f"hero_{variant}",
                    "variant": variant,
                    "file": rel,
                    "px": size,
                    "panels": [label for label, _ in HERO_PANELS],
                    "sources": {kind: str(Path(p).name) for kind, p in paths.items()},
                    "photo_used": bool(ctx.get("photo")),
                    "thumb": ctx["thumbnail"](gallery_dir / rel, gallery_dir),
                    "view": ctx["view"](gallery_dir / rel, gallery_dir),
                }
            )
    return records


# ---------------------------------------------------------------------------------------
# The tour
# ---------------------------------------------------------------------------------------

TOUR = [
    {
        "id": "legend_portrait",
        "title": "Reading a phase portrait",
        "section": "phase portraits",
        "render": _legend_portrait,
        "family": "portrait",
        "inputs": {
            "function": "(z**2 - 1) / (z**2 + 1)",
            "domain": "Rectangle(4, 4)",
            "colormap": "Phase(phase_sectors=6, auto_scale_r=True)",
        },
        "caption": (
            "Hue is the phase of f(z) and the shaded cells are its contour bands, so zeros and "
            "poles read as opposite winding directions. The inset legend is the same colormap "
            "applied to the identity map, which is what makes the picture decodable."
        ),
        "alt": "2D phase portrait of (z^2-1)/(z^2+1) with a phase-wheel legend inset",
        "snippet": (
            "import complexplorer as cp\n"
            'preset = cp.catalog.get("rational_zeros_poles")\n'
            "cp.plot(preset.domain(), preset.func, cmap=preset.colormap(), legend=True)"
        ),
    },
    {
        "id": "portrait_to_landscape",
        "title": "From the plane to a landscape",
        "section": "mapping and topology",
        "render": _portrait_to_landscape,
        "family": "landscape",
        "inputs": {
            "function": "(z**2 - 1) / (z**2 + 1)",
            "domain": "Rectangle(4, 4)",
            "colormap": "Phase(phase_sectors=6, auto_scale_r=True)",
            "scaling": "the preset's recommended modulus scaling",
        },
        "caption": (
            "The same function twice: flat, then with |f(z)| lifted into height. The zeros sink "
            "and the poles rise, while the colours stay put — the landscape adds magnitude "
            "without changing what the hue means."
        ),
        "alt": "Side-by-side 2D phase portrait and 3D analytic landscape of the same function",
        "snippet": (
            "import complexplorer as cp\n"
            'preset = cp.catalog.get("rational_zeros_poles")\n'
            "cp.plot(preset.domain(), preset.func, cmap=preset.colormap(), legend=True)\n"
            "sc = preset.scaling()\n"
            "cp.plot_landscape_pv(preset.domain(), preset.func, cmap=preset.colormap(),\n"
            '                     modulus_mode=sc["method"], modulus_params=sc["params"])'
        ),
    },
    {
        "id": "engineering_figure",
        "title": "Engineering mode: a notch filter",
        "section": "engineering",
        "render": _engineering_figure,
        "family": "portrait",
        "inputs": {
            "function": "H(s) = (s**2 + 4) / (s**3 + 1.2*s**2 + 5*s + 2)",
            "domain": "the transfer portrait's default window around the jw axis",
            "colormap": "the transfer portrait default",
        },
        "caption": (
            "One stable transfer function in four views. The zeros sit exactly on the jw axis at "
            "+-2j — the notch — while the poles stay inside the left half-plane. The portrait "
            "shows where they are; Bode and Nyquist show what they do to a signal."
        ),
        "alt": (
            "Four-panel engineering figure: transfer portrait, pole-zero map, Nyquist plot and "
            "Bode magnitude/phase for a notch filter"
        ),
        "snippet": (
            "import complexplorer as cp\n"
            "H = cp.ee.TransferFunction([1, 0, 4], [1, 1.2, 5, 2])   # a notch at w = 2\n"
            "cp.ee.transfer_portrait(H, legend=True)\n"
            "cp.ee.pole_zero_plot(H)\n"
            "cp.ee.bode_plot(H)\n"
            "cp.ee.nyquist_plot(H)"
        ),
    },
    {
        "id": "composition_proof",
        "title": "A transfer function is just a complex function",
        "section": "engineering",
        "render": _composition_proof,
        "family": "landscape",
        "inputs": {
            "function": "H(s) = (s**2 + 4) / (s**3 + 1.2*s**2 + 5*s + 2)",
            "domain": "Rectangle(6, 6)",
            "colormap": "the renderer default",
        },
        "caption": (
            "`TransferFunction` is a plain callable, so the engineering view and the general 3D "
            "renderer are looking at the same object. Nothing converts between them: the notch "
            "that reads as a dark point on the left is the valley on the right."
        ),
        "alt": (
            "Transfer portrait beside a 3D analytic landscape of the same transfer function"
        ),
        "snippet": (
            "import complexplorer as cp\n"
            "H = cp.ee.TransferFunction([1, 0, 4], [1, 1.2, 5, 2])\n"
            "cp.ee.transfer_portrait(H, legend=True)      # the engineering view\n"
            "cp.plot_landscape_pv(cp.Rectangle(6, 6), H)  # the same object, a general renderer"
        ),
    },
    {
        "id": "composite_domain",
        "title": "Domains compose",
        "section": "mapping and topology",
        "render": _composite_domain,
        "family": "portrait",
        "inputs": {
            "function": "1 / z",
            "domain": "(Disk(1.1, -0.75) | Disk(1.1, 0.75)) - Disk(0.4)",
            "colormap": "Phase(phase_sectors=12, auto_scale_r=True)",
        },
        "caption": (
            "The outline is two overlapping disks unioned together, with a third punched out of "
            "the middle. Excluding a neighbourhood of the pole is not cosmetic: it keeps the huge "
            "values near z = 0 out of the sampling entirely."
        ),
        "alt": (
            "Phase portrait of 1/z on a peanut-shaped union of two disks with a disk removed "
            "around the pole"
        ),
        "snippet": (
            "import complexplorer as cp\n"
            "domain = (cp.Disk(1.6, center=-0.7) & cp.Disk(1.6, center=0.7)) - cp.Disk(0.35)\n"
            "cp.plot(domain, lambda z: 1 / z,\n"
            "        cmap=cp.Phase(phase_sectors=12, auto_scale_r=True), legend=True)"
        ),
    },
    {
        "id": "sphere_vs_surface",
        "title": "Sphere versus surface",
        "section": "riemann surfaces",
        "render": _sphere_vs_surface,
        "family": "sphere",
        "inputs": {
            "function": "1 / z  (sphere) and sqrt(z)  (surface)",
            "domain": "the full sphere; the surface's own radial mesh",
            "colormap": "the preset colormap / the renderer default",
        },
        "caption": (
            "Two different objects that are easy to confuse. The sphere compactifies the plane so "
            "one single-valued function can include the point at infinity. The surface is the "
            "two-sheeted cover on which the multivalued sqrt(z) becomes single-valued."
        ),
        "alt": "Riemann sphere of 1/z beside the two-sheeted Riemann surface of the square root",
        "snippet": (
            "import complexplorer as cp\n"
            'preset = cp.catalog.get("reciprocal")\n'
            "cp.riemann_pv(preset.func, cmap=preset.colormap())   # the SPHERE\n"
            'cp.riemann_surface_pv("power", n=2)                  # the SURFACE'
        ),
    },
    {
        "id": "physical_triptych",
        "title": "From function to printed object",
        "section": "physical output",
        "render": _physical_triptych,
        "family": "ornament",
        "inputs": {
            "function": "z / (z**10 - 1)",
            "domain": "the full sphere",
            "colormap": "the preset colormap",
            "scaling": "the preset's recommended modulus scaling",
        },
        "caption": (
            "The relief is the mathematics, the mesh is the geometry that survives losing the "
            "colour, and the print is the object on a desk. Ten poles become ten spikes around "
            "the central zero."
        ),
        "alt": (
            "Three panels: Riemann relief render, untextured STL mesh, and the printed ornament"
        ),
        "snippet": (
            "import complexplorer as cp\n"
            "from complexplorer.export.stl import OrnamentGenerator\n"
            'preset = cp.catalog.get("pole_flower_10")\n'
            "sc = preset.scaling()\n"
            "cp.riemann_pv(preset.func, cmap=preset.colormap(),\n"
            '              modulus_mode=sc["method"], modulus_params=sc["params"])\n'
            'OrnamentGenerator(preset.func, resolution=150, scaling=sc["method"]).generate_and_save(\n'
            '    "pole_flower.stl", size_mm=80)'
        ),
    },
    {
        "id": "orbit_loop",
        "title": "It rotates",
        "section": "physical output",
        "render": _orbit_loop,
        "family": "ornament",
        "suffix": ".gif",
        "inputs": {
            "function": "z / (z**10 - 1)",
            "domain": "the full sphere",
            "colormap": "the preset colormap",
            "scaling": "the preset's recommended modulus scaling",
        },
        "caption": (
            "A still cannot show depth or how the light moves across the spikes. Every 3D view in "
            "the library is an interactive PyVista window; this is 36 frames of one."
        ),
        "alt": "Animated loop of the pole-flower relief rotating on the Riemann sphere",
        "snippet": (
            "import complexplorer as cp\n"
            'preset = cp.catalog.get("pole_flower_10")\n'
            "sc = preset.scaling()\n"
            "# interactive=True (the default) opens a window you can rotate, zoom and light\n"
            "cp.riemann_pv(preset.func, cmap=preset.colormap(),\n"
            '              modulus_mode=sc["method"], modulus_params=sc["params"])'
        ),
    },
]


def render_tour(gallery_dir: Path, ctx: dict) -> list[dict]:
    """Render every tour asset into ``gallery_dir/_tour`` and return their manifest records."""
    records = []
    for recipe in TOUR:
        rel = f"_tour/{recipe['id']}{recipe.get('suffix', '.png')}"
        path = gallery_dir / rel
        print(f"  tour: {recipe['id']}")
        recipe["render"](ctx, path)
        record = {
            "id": recipe["id"],
            "title": recipe["title"],
            "section": recipe["section"],
            "file": rel,
            "caption": recipe["caption"],
            "alt": recipe["alt"],
            "snippet": recipe["snippet"],
            "inputs": recipe["inputs"],
            "profile": {
                "profile": recipe["family"],
                "resolution": ctx["profiles"][recipe["family"]].get("resolution"),
            },
        }
        if not rel.endswith(".gif"):
            record["thumb"] = ctx["thumbnail"](path, gallery_dir)
            record["view"] = ctx["view"](path, gallery_dir)
        records.append(record)
    return records
