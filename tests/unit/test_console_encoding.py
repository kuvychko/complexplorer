"""The package must not crash when its output meets a legacy console code page.

Windows consoles default to a legacy code page (cp437 in a stock ``cmd.exe``, cp1252 for a
redirected stream under many locales), and Python encodes stdout with it. A single non-ASCII
character in printed output therefore ends a run in ``UnicodeEncodeError`` — not a garbled line,
a traceback. Both halves of this file guard a bug that shipped:

* ``complexplorer list`` died on the preset title ``z³ - z`` under cp437.
* ``OrnamentGenerator.generate_and_save`` died on a ``✗`` status marker under cp1252, after
  writing nothing.
"""

from __future__ import annotations

import ast
import io
import sys
from pathlib import Path

import numpy as np
import pytest

import complexplorer
from complexplorer.cli.main import main

PACKAGE_ROOT = Path(complexplorer.__file__).resolve().parent


def _legacy_console(encoding: str) -> io.TextIOWrapper:
    """A stdout that behaves like a console on ``encoding``: strict, and it will raise."""
    return io.TextIOWrapper(io.BytesIO(), encoding=encoding, errors="strict", line_buffering=True)


def _printed_string_literals() -> list[tuple[str, int, str]]:
    """Every string constant the package hands to ``print``, including f-string pieces."""
    literals: list[tuple[str, int, str]] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            is_print = (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "print"
            )
            if not is_print:
                continue
            for constant in ast.walk(node):
                if isinstance(constant, ast.Constant) and isinstance(constant.value, str):
                    literals.append((path.name, constant.lineno, constant.value))
    return literals


class TestPrintedLiteralsAreASCII:
    def test_the_package_prints_only_ascii(self):
        """Status markers are decoration; a check mark is not worth an exception."""
        offenders = [
            f"{name}:{line} {value!r}"
            for name, line, value in _printed_string_literals()
            if not value.isascii()
        ]
        assert not offenders, (
            "these printed literals cannot be encoded by a legacy Windows console:\n  "
            + "\n  ".join(offenders)
        )

    def test_the_check_would_notice_a_check_mark(self):
        """The scan reads print() arguments, so a regression is actually visible to it."""
        assert _printed_string_literals(), "the scan found no printed literals at all"


class TestTheCLISurvivesALegacyConsole:
    @pytest.mark.parametrize("encoding", ["cp437", "cp1252", "ascii"])
    def test_list_does_not_raise(self, encoding, monkeypatch):
        """``complexplorer list`` prints preset titles, which are mathematical text."""
        console = _legacy_console(encoding)
        monkeypatch.setattr(sys, "stdout", console)
        monkeypatch.setattr(sys, "stderr", console)

        assert main(["list"]) == 0

        console.flush()
        printed = console.buffer.getvalue().decode(encoding)
        assert "identity" in printed
        # Degraded, not dropped: the title is still there, minus what the code page lacks.
        assert "cubic_real_roots" in printed


class TestSTLStatusOutputSurvivesALegacyConsole:
    @pytest.mark.parametrize("encoding", ["cp437", "cp1252"])
    def test_validation_report_does_not_raise(self, encoding, monkeypatch):
        """The verbose STL report is printed by the library itself, not by the CLI."""
        pv = pytest.importorskip("pyvista")
        from complexplorer.export.stl.utils import validate_printability

        console = _legacy_console(encoding)
        monkeypatch.setattr(sys, "stdout", console)

        # An open surface, so the report takes its unhappy branches: the ones with the markers.
        mesh = pv.PolyData(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            faces=np.array([3, 0, 1, 2]),
        )
        validate_printability(mesh, size_mm=1.0, verbose=True)

        console.flush()
        assert "Overall Assessment" in console.buffer.getvalue().decode(encoding)
