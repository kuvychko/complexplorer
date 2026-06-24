"""Complexplorer CLI: ``render``, ``stl``, ``list``, ``gallery``.

Thin orchestration over the existing machinery — it resolves a function
(``preset:<id>`` via the catalog, or an expression via ``core.expression``), resolves a
domain/colormap (shorthand → spec dict → the catalog's factories), then dispatches to the
right backend (matplotlib 2D / PyVista 3D / STL ornament) and writes a file.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from .. import HAS_PYVISTA
from ..core.expression import compile_expression
from ..core.presets import catalog, cmap_from_spec, domain_from_spec
from ..plotting.matplotlib.plot_2d import plot as plot_2d
from ..utils.validation import ValidationError

# --------------------------------------------------------------------------------------
# Resolution helpers (reuse the catalog + spec factories)
# --------------------------------------------------------------------------------------


def _resolve_func(arg: str):
    """Return (callable, preset_or_None) for a ``preset:<id>`` or an expression string."""
    if arg.startswith("preset:"):
        preset = catalog.get(arg[len("preset:") :])
        return preset.func, preset
    return compile_expression(arg), None


def _parse_domain(text: str) -> dict:
    """``rect:4:4`` / ``disk:2`` / ``annulus:0.2:3`` → a domain spec dict."""
    kind, *rest = text.split(":")
    nums = [float(x) for x in rest]
    if kind in ("rect", "rectangle") and len(nums) == 2:
        return {"type": "rectangle", "re_length": nums[0], "im_length": nums[1]}
    if kind == "disk" and len(nums) == 1:
        return {"type": "disk", "radius": nums[0]}
    if kind == "annulus" and len(nums) == 2:
        return {"type": "annulus", "inner_radius": nums[0], "outer_radius": nums[1]}
    raise ValidationError(f"Bad --domain {text!r}; use rect:RE:IM, disk:R, or annulus:IN:OUT")


def _parse_cmap(text: str) -> dict:
    """``phase`` / ``phase:6`` → a colormap spec dict."""
    name, *rest = text.split(":")
    aliases = {"phase": "Phase", "chessboard": "Chessboard", "logrings": "LogRings"}
    type_name = aliases.get(name.lower(), name)
    spec: dict = {"type": type_name}
    if type_name == "Phase" and rest:
        spec["n_phi"] = int(rest[0])
    return spec


def _need_pyvista(what: str) -> None:
    if not HAS_PYVISTA:
        raise ValidationError(
            f"{what} requires the PyVista 3D backend. Install with: pip install "
            '"complexplorer[pyvista]"'
        )


# --------------------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------------------


def cmd_render(args: argparse.Namespace) -> int:
    func, preset = _resolve_func(args.func)

    domain = None
    if args.domain:
        domain = domain_from_spec(_parse_domain(args.domain))
    elif preset is not None:
        domain = preset.domain()

    cmap = None
    if args.cmap:
        cmap = cmap_from_spec(_parse_cmap(args.cmap))
    elif preset is not None:
        cmap = preset.colormap()

    if not args.output and not args.show:
        raise ValidationError("provide --output FILE or --show")

    # Dispatch directly to the right backend (matplotlib 2D, PyVista 3D — per the backend
    # policy) rather than quick_plot, whose 3D path defaults to the deprecated matplotlib
    # renderer unless backend='pyvista' is threaded through.
    if args.mode == "2d":
        kwargs = {"cmap": cmap, "filename": args.output}
        if args.resolution:
            kwargs["resolution"] = args.resolution
        plot_2d(domain, func, **{k: v for k, v in kwargs.items() if v is not None})
    else:
        _need_pyvista(f"render --mode {args.mode}")
        common = {"cmap": cmap, "filename": args.output, "interactive": bool(args.show)}
        if args.resolution:
            common["resolution"] = args.resolution
        if args.scaling:
            common["modulus_mode"] = args.scaling
        common = {k: v for k, v in common.items() if v is not None}
        if args.mode == "3d":
            from ..plotting.pyvista.plot_3d import plot_landscape_pv

            plot_landscape_pv(domain, func, **common)
        else:  # riemann
            from ..plotting.pyvista.riemann import riemann_pv

            riemann_pv(func, domain=domain, **common)

    if args.output:
        print(f"Wrote {args.output}")
    return 0


def cmd_stl(args: argparse.Namespace) -> int:
    _need_pyvista("stl")
    from ..export.stl import OrnamentGenerator

    func, preset = _resolve_func(args.func)
    scaling = args.scaling or "arctan"
    gen = OrnamentGenerator(func, resolution=args.resolution, scaling=scaling)
    gen.generate_and_save(args.output, size_mm=args.size_mm, verbose=False)
    print(f"Wrote {args.output}")
    return 0


def cmd_gallery(args: argparse.Namespace) -> int:
    from ..gallery import _resolve, generate_gallery

    if not args.output:
        raise ValidationError("provide --output DIR")
    selection: str | list[str] | None = args.preset or args.tag or None
    if not _resolve(selection):
        raise ValidationError(f"0 presets matched selection {selection!r}")
    manifest = generate_gallery(args.output, selection=selection)
    print(f"Wrote {len(manifest['presets'])} preset(s) to {args.output}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    presets = catalog.filter(tag=args.tag) if args.tag else [catalog.get(i) for i in catalog.list()]
    for p in presets:
        tags = ", ".join(p.tags)
        print(f"{p.id:22} {p.title:24} [{tags}]")
    return 0


# --------------------------------------------------------------------------------------
# Argument parser
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="complexplorer", description="Visualize and export complex functions."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_func_arg(p):
        p.add_argument("func", help="a preset:<id> or an expression like 'z/(z**10-1)'")

    pr = sub.add_parser("render", help="render a 2D/3D image")
    add_func_arg(pr)
    pr.add_argument("--mode", choices=["2d", "3d", "riemann"], default="2d")
    pr.add_argument("--domain", help="rect:RE:IM | disk:R | annulus:IN:OUT")
    pr.add_argument("--cmap", help="e.g. phase or phase:6")
    pr.add_argument("--scaling", help="modulus scaling mode for 3d/riemann (e.g. arctan)")
    pr.add_argument("--resolution", type=int)
    pr.add_argument("--output", "-o", help="output image file")
    pr.add_argument("--show", action="store_true", help="open an interactive window")
    pr.set_defaults(handler=cmd_render)

    ps = sub.add_parser("stl", help="export a 3D-printable STL")
    add_func_arg(ps)
    ps.add_argument("--size-mm", type=float, default=50.0)
    ps.add_argument("--resolution", type=int, default=150)
    ps.add_argument("--scaling", help="modulus scaling mode (default arctan)")
    ps.add_argument("--output", "-o", required=True, help="output .stl file")
    ps.set_defaults(handler=cmd_stl)

    pl = sub.add_parser("list", help="list catalog presets")
    pl.add_argument("--tag", help="only presets carrying this tag")
    pl.set_defaults(handler=cmd_list)

    pg = sub.add_parser("gallery", help="render a preset set into a reproducible asset bundle")
    pg.add_argument("--tag", help="render presets carrying this tag")
    pg.add_argument("--preset", nargs="+", metavar="ID", help="explicit preset ids to render")
    pg.add_argument("--output", "-o", help="output directory")
    pg.set_defaults(handler=cmd_gallery)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler: Callable[[argparse.Namespace], int] = args.handler
    try:
        return handler(args)
    except ValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
