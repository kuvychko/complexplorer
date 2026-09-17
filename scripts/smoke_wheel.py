#!/usr/bin/env python3
"""Exercise an INSTALLED complexplorer: the five things a user does in the first five minutes.

Run this from a directory that is not the repository, against an environment where the wheel is
installed. It refuses to run against a source checkout, because that is exactly the mistake it
exists to catch:

    uv build
    uv venv /tmp/smoke && /tmp/smoke/bin/python -m pip install dist/*.whl
    cd /tmp && /tmp/smoke/bin/python /path/to/repo/scripts/smoke_wheel.py

Checks: import and version, the console entry point, a 2D render, a real off-screen PyVista
render, and a small STL export.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

MIN_IMAGE_BYTES = 5_000
MIN_STL_BYTES = 50_000


def _console_script() -> Path | None:
    """Find the installed console script.

    Looked up next to the running interpreter first: invoking a venv's python directly (as CI
    does) leaves that venv's script directory off PATH, and this check is about whether the
    entry point was installed, not about how PATH happens to be set.
    """
    bin_dir = Path(sys.prefix) / ("Scripts" if os.name == "nt" else "bin")
    for name in ("complexplorer", "complexplorer.exe"):
        candidate = bin_dir / name
        if candidate.exists():
            return candidate
    found = shutil.which("complexplorer")
    return Path(found) if found else None


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")

    import complexplorer as cp

    installed_from = Path(cp.__file__).resolve().parent
    if (installed_from.parent / "pyproject.toml").exists():
        print(
            f"refusing to smoke-test the source tree at {installed_from}; "
            "install the wheel and run this from elsewhere",
            file=sys.stderr,
        )
        return 2

    print(f"  import      {cp.__version__} from {installed_from}")

    started = time.perf_counter()
    subprocess.run([sys.executable, "-c", "import complexplorer"], check=True)
    print(f"  cold import {time.perf_counter() - started:.2f} s (subprocess)")

    entry_point = _console_script()
    if entry_point is None:
        print("  entry point  NOT on PATH", file=sys.stderr)
        return 1
    listing = subprocess.run([str(entry_point), "list"], capture_output=True, text=True, check=True)
    preset_lines = [line for line in listing.stdout.splitlines() if line.strip()]
    print(f"  cli list     {len(preset_lines)} lines")

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)

        portrait = out / "portrait.png"
        cp.plot(
            cp.Rectangle(4, 4),
            lambda z: (z**2 - 1) / (z**2 + 1),
            cmap=cp.Phase(phase_sectors=6, auto_scale_r=True),
            resolution=200,
            filename=str(portrait),
        )
        print(f"  2D render    {portrait.stat().st_size / 1024:.0f} KB")

        sphere = out / "sphere.png"
        cp.riemann_pv(
            lambda z: (z**2 - 1) / (z**2 + 1),
            resolution=60,
            window_size=(400, 400),
            interactive=False,
            filename=str(sphere),
        )
        print(f"  3D render    {sphere.stat().st_size / 1024:.0f} KB (off-screen PyVista)")

        from complexplorer.export.stl import OrnamentGenerator

        ornament = out / "ornament.stl"
        OrnamentGenerator(lambda z: z / (z**10 - 1), resolution=60).generate_and_save(
            str(ornament), size_mm=30
        )
        print(f"  STL export   {ornament.stat().st_size / 1024:.0f} KB")

        failures = []
        for path, floor in (
            (portrait, MIN_IMAGE_BYTES),
            (sphere, MIN_IMAGE_BYTES),
            (ornament, MIN_STL_BYTES),
        ):
            if not path.exists() or path.stat().st_size < floor:
                failures.append(f"{path.name} is missing or suspiciously small")
        if failures:
            for failure in failures:
                print(f"  - {failure}", file=sys.stderr)
            return 1

    print("\ninstalled package works")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
