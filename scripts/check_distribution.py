#!/usr/bin/env python3
"""Inspect the built wheel and sdist before they are published.

The test suite runs against the source tree, so it cannot see what a user actually installs: a
wheel can ship without ``py.typed``, without its licenses, with a broken console entry point, or
with the tests and gallery images swept in, and every test still passes. This script checks the
artifact itself.

    uv build
    python scripts/check_distribution.py          # inspects dist/
    python scripts/check_distribution.py path/to/dist

Exits non-zero and prints every failure, so one run tells you everything that is wrong.
"""

from __future__ import annotations

import re
import sys
import tarfile
import zipfile
from email.parser import Parser
from pathlib import Path

# Import paths that must be present in the wheel: every subpackage a user can reach.
EXPECTED_MODULES = [
    "complexplorer/__init__.py",
    "complexplorer/py.typed",
    "complexplorer/api.py",
    "complexplorer/exceptions.py",
    "complexplorer/gallery.py",
    "complexplorer/core/colormap.py",
    "complexplorer/core/color_utils.py",
    "complexplorer/core/domain.py",
    "complexplorer/core/presets.py",
    "complexplorer/ee/transfer_function.py",
    "complexplorer/mesh/surface.py",
    "complexplorer/plotting/matplotlib/plot_2d.py",
    "complexplorer/plotting/pyvista/riemann.py",
    "complexplorer/export/stl/ornament_generator.py",
    "complexplorer/cli/main.py",
]

EXPECTED_LICENSES = ["LICENSE", "LICENSE.art"]

# Anything matching these has no business in a distribution.
FORBIDDEN = re.compile(
    r"(^|/)(tests?|examples|docs|openspec|site|\.github)/|\.(pyc|png|jpg|jpeg|gif|webp|stl|ipynb)$"
)

REQUIRED_DEPENDENCIES = ["numpy", "matplotlib", "scipy", "asteval", "pyvista"]

ENTRY_POINT = "complexplorer = complexplorer.cli.main:main"


class Report:
    """Collects failures so one run reports everything, not just the first problem."""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.notes: list[str] = []

    def check(self, condition: bool, failure: str) -> bool:
        if not condition:
            self.failures.append(failure)
        return condition

    def note(self, text: str) -> None:
        self.notes.append(text)


def _wheel_metadata(names: list[str], archive: zipfile.ZipFile) -> dict:
    metadata_path = next(n for n in names if n.endswith(".dist-info/METADATA"))
    return Parser().parsestr(archive.read(metadata_path).decode("utf-8"))


def check_wheel(path: Path, report: Report) -> None:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        metadata = _wheel_metadata(names, archive)
        entry_points = next((n for n in names if n.endswith("entry_points.txt")), None)
        entry_text = archive.read(entry_points).decode("utf-8") if entry_points else ""

    report.note(f"wheel {path.name} ({path.stat().st_size / 1024:.0f} KB, {len(names)} entries)")

    for module in EXPECTED_MODULES:
        report.check(module in names, f"wheel is missing {module}")

    for license_name in EXPECTED_LICENSES:
        report.check(
            any(n.endswith(f"licenses/{license_name}") for n in names),
            f"wheel is missing the {license_name} file",
        )

    intruders = [n for n in names if FORBIDDEN.search(n)]
    report.check(not intruders, f"wheel carries files it should not: {intruders[:5]}")

    top_level = {n.split("/")[0] for n in names}
    report.check(
        top_level <= {"complexplorer"} | {n for n in top_level if n.endswith(".dist-info")},
        f"wheel has unexpected top-level entries: {sorted(top_level)}",
    )

    version = metadata.get("Version", "")
    report.check(bool(version), "wheel metadata has no Version")
    report.check(
        path.name.startswith(f"complexplorer-{version}-"),
        f"wheel filename and metadata version disagree: {path.name} vs {version}",
    )
    report.note(f"version {version}")

    requires_python = metadata.get("Requires-Python", "")
    report.check(">=3.11" in requires_python, f"unexpected Requires-Python: {requires_python!r}")

    license_expression = metadata.get("License-Expression") or metadata.get("License", "")
    report.check(
        license_expression.strip() == "MIT",
        f"expected the MIT SPDX license expression, found {license_expression!r}",
    )

    declared = " ".join(metadata.get_all("Requires-Dist") or [])
    for dependency in REQUIRED_DEPENDENCIES:
        report.check(dependency in declared, f"wheel does not declare {dependency}")

    urls = metadata.get_all("Project-URL") or []
    report.check(len(urls) >= 2, f"wheel declares too few project URLs: {urls}")

    report.check(
        ENTRY_POINT in entry_text.replace('"', ""),
        f"console entry point missing; entry_points.txt said: {entry_text.strip()!r}",
    )


def check_sdist(path: Path, report: Report) -> None:
    with tarfile.open(path) as archive:
        names = archive.getnames()
    report.note(f"sdist {path.name} ({path.stat().st_size / 1024:.0f} KB, {len(names)} entries)")

    # Everything in an sdist sits under one <name>-<version>/ prefix.
    stripped = ["/".join(n.split("/")[1:]) for n in names if "/" in n]
    for required in [
        "pyproject.toml",
        "README.md",
        "complexplorer/__init__.py",
        *EXPECTED_LICENSES,
    ]:
        report.check(required in stripped, f"sdist is missing {required}")

    intruders = [n for n in stripped if FORBIDDEN.search(n)]
    report.check(not intruders, f"sdist carries files it should not: {intruders[:5]}")


def main(argv: list[str]) -> int:
    dist = Path(argv[1]) if len(argv) > 1 else Path("dist")
    if not dist.is_dir():
        print(f"no such directory: {dist}", file=sys.stderr)
        return 2

    wheels = sorted(dist.glob("*.whl"))
    sdists = sorted(dist.glob("*.tar.gz"))
    report = Report()
    report.check(len(wheels) == 1, f"expected exactly one wheel in {dist}, found {len(wheels)}")
    report.check(len(sdists) == 1, f"expected exactly one sdist in {dist}, found {len(sdists)}")

    if wheels:
        check_wheel(wheels[0], report)
    if sdists:
        check_sdist(sdists[0], report)

    for note in report.notes:
        print(f"  {note}")
    if report.failures:
        print(f"\n{len(report.failures)} problem(s) with the distribution:", file=sys.stderr)
        for failure in report.failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print("\ndistribution looks publishable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
