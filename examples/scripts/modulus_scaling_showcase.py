#!/usr/bin/env python3
"""Showcase of modulus-scaling modes for 3D landscapes (PyVista).

Different modulus-scaling modes map ``|f(z)|`` to surface height differently, emphasizing
poles, zeros, or mid-range structure. This demo renders the same function under each mode.

PyVista is the 3D backend (the matplotlib 3D paths were removed in 3.0), so each mode opens
its own high-quality window. Run from a terminal:

    python examples/scripts/modulus_scaling_showcase.py            # interactive, one window per mode
    python examples/scripts/modulus_scaling_showcase.py --save out # save a screenshot per mode

Close a window (Q) to advance to the next mode.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import complexplorer as cp

# A function with diverse behavior: zeros from the cubic, a pair of poles.
def f(z):
    return (z**3 - 1) / (z**2 + 0.5)


DOMAIN = cp.Rectangle(4, 4)
CMAP = cp.Phase(n_phi=6, auto_scale_r=True)

# (mode, description) — the bounded/illustrative subset of ModulusScaling modes.
MODES = [
    ("none", "Direct modulus"),
    ("constant", "Phase only (flat)"),
    ("arctan", "Smooth bounded"),
    ("logarithmic", "Logarithmic"),
    ("linear_clamp", "Linear with clamping"),
    ("adaptive", "Percentile-based adaptive"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save", metavar="DIR", default=None,
        help="save a screenshot per mode into DIR instead of showing interactively",
    )
    parser.add_argument("--resolution", type=int, default=120, help="mesh resolution")
    args = parser.parse_args()

    out_dir = Path(args.save) if args.save else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    for mode, description in MODES:
        title = f"{description}  (modulus_mode='{mode}')"
        print(f"Rendering: {title}")
        filename = str(out_dir / f"modulus_{mode}.png") if out_dir else None
        cp.plot_landscape_pv(
            DOMAIN, f, cmap=CMAP,
            resolution=args.resolution,
            modulus_mode=mode,
            title=title,
            interactive=out_dir is None,
            filename=filename,
        )

    if out_dir:
        print(f"Saved {len(MODES)} screenshots to {out_dir}")


if __name__ == "__main__":
    main()
