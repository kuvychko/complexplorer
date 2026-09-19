"""Structural guard for the examples/ tree (restructure-examples, M1).

These tests enforce the `examples` capability invariants: the example scripts reference no
symbol removed at 3.0, and the obsolete layout (legacy generators, archive/old) is gone. They
are static checks — they do not import or execute the examples (notebook execution is M3's job).
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPO_ROOT / "examples"

# Symbols removed at 3.0 that must never appear in example scripts. The `\b...\b` boundaries
# deliberately do NOT match the surviving PyVista variants: `plot_landscape_pv`,
# `pair_plot_landscape_pv`, `riemann_pv`, `riemann_chart`, `riemann_hemispheres`,
# `riemann_surface_pv` (a `_` after the name is a word char, so no boundary there).
_FORBIDDEN = {
    "plot_landscape": re.compile(r"\bplot_landscape\b"),
    "pair_plot_landscape": re.compile(r"\bpair_plot_landscape\b"),
    "riemann() (3D mpl)": re.compile(r"\briemann\s*\("),
    "HAS_PYVISTA": re.compile(r"\bHAS_PYVISTA\b"),
    "HAS_STL_EXPORT": re.compile(r"\bHAS_STL_EXPORT\b"),
}


def _example_py_files():
    return [p for p in EXAMPLES.rglob("*.py") if "__pycache__" not in p.parts]


def test_no_example_script_references_a_removed_symbol():
    offenders = []
    for path in _example_py_files():
        text = path.read_text(encoding="utf-8")
        for name, pattern in _FORBIDDEN.items():
            if pattern.search(text):
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {name}")
    assert not offenders, "examples/ reference symbols removed at 3.0:\n" + "\n".join(offenders)


def test_obsolete_directories_are_gone():
    assert not (EXAMPLES / "archive").exists(), "examples/archive should be removed"
    assert not (EXAMPLES / "old").exists(), "examples/old should be removed"


def test_legacy_gallery_generators_are_gone():
    assert not (EXAMPLES / "generate_gallery.py").exists()
    assert not (EXAMPLES / "gallery" / "generate_gallery_images.py").exists()


def test_layout_directories_exist():
    assert (EXAMPLES / "notebooks").is_dir()
    assert (EXAMPLES / "scripts").is_dir()
    assert (EXAMPLES / "scripts" / "interactive_showcase.py").is_file()


def test_expected_notebooks_are_present():
    """The tutorial set the `examples` capability describes (incl. the CVD notebook)."""
    expected = {
        "getting_started.ipynb",
        "advanced_features.ipynb",
        "api_cookbook.ipynb",
        "stl_export_demo.ipynb",
        "color_and_accessibility.ipynb",
    }
    present = {p.name for p in (EXAMPLES / "notebooks").glob("*.ipynb")}
    assert expected <= present, f"missing tutorials: {sorted(expected - present)}"


def test_application_notebooks_live_in_their_own_directory():
    apps = EXAMPLES / "notebooks" / "applications"
    assert apps.is_dir()
    assert len(list(apps.glob("*.ipynb"))) >= 4


def test_notebooks_use_the_current_phase_sector_parameter():
    """`n_phi` was renamed to `phase_sectors` in 2.0; notebooks must not teach the old name."""
    offenders = [
        p.relative_to(REPO_ROOT)
        for p in (EXAMPLES / "notebooks").rglob("*.ipynb")
        if re.search(r"n_phi", p.read_text(encoding="utf-8"))
    ]
    assert not offenders, f"notebooks still reference n_phi: {offenders}"
