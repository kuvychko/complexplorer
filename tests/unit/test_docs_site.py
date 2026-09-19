"""The documentation site must keep describing the library that is actually here.

These checks are static and fast: they do not build the site (that is `mkdocs build --strict` in
CI), they check the things that silently drift -- a nav entry pointing at a page someone renamed,
a public name that never reached the API reference, a CLI page that fell behind the program.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import complexplorer as cp

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs"
MKDOCS_YML = REPO_ROOT / "mkdocs.yml"


def _nav_page_paths() -> list[str]:
    """Every `foo/bar.md` mentioned in the mkdocs nav, without needing a YAML parser."""
    text = MKDOCS_YML.read_text(encoding="utf-8")
    nav = text[text.index("nav:") : text.index("extra:", text.index("nav:"))]
    return re.findall(r":\s*([\w./-]+\.(?:md|ipynb))\s*$", nav, re.M)


class TestTheNavMatchesTheTree:
    def test_every_nav_page_exists(self):
        missing = [page for page in _nav_page_paths() if not (DOCS / page).exists()]
        assert not missing, f"mkdocs.yml lists pages that do not exist: {missing}"

    def test_the_nav_is_not_empty(self):
        """A nav that silently emptied would make every other check here vacuous."""
        assert len(_nav_page_paths()) > 10


class TestTheAPIReferenceCoversThePublicSurface:
    @staticmethod
    def _documented() -> set[str]:
        documented: set[str] = set()
        for page in (DOCS / "api").glob("*.md"):
            documented |= set(
                re.findall(
                    r"^::: complexplorer\.(?:ee\.)?(\w+)", page.read_text(encoding="utf-8"), re.M
                )
            )
        return documented

    def test_every_public_name_is_documented(self):
        """Grouping the reference by hand must not silently drop a name."""
        expected = {name for name in cp.__all__ if name not in {"__version__", "ee"}}
        missing = sorted(expected - self._documented())
        assert not missing, (
            f"these names are exported but appear in no API reference page: {missing}"
        )

    def test_every_engineering_name_is_documented(self):
        expected = {name for name in dir(cp.ee) if not name.startswith("_")}
        missing = sorted(expected - self._documented())
        assert not missing, f"these cp.ee names appear in no API reference page: {missing}"

    def test_the_reference_documents_nothing_invented(self):
        """A `:::` line naming something that does not exist renders as an error box."""
        unknown = sorted(
            name
            for name in self._documented()
            if not hasattr(cp, name) and not hasattr(cp.ee, name)
        )
        assert not unknown, f"the API reference names things that do not exist: {unknown}"


class TestTheCLIPageMatchesTheCLI:
    def test_every_subcommand_is_documented(self):
        from complexplorer.cli.main import build_parser

        page = (DOCS / "guide" / "cli.md").read_text(encoding="utf-8")
        actions = [
            action
            for action in build_parser()._subparsers._group_actions[0].choices  # type: ignore[union-attr]
        ]
        undocumented = [name for name in actions if f"complexplorer {name}" not in page]
        assert not undocumented, (
            f"these subcommands exist but the CLI page does not mention them: {undocumented}"
        )


class TestTheColormapGuidanceIsStillTrue:
    """The accessibility table in the guide is measured, so it can go stale silently."""

    def test_cubehelix_is_still_the_most_robust_family(self):
        """The guide tells readers to reach for CubehelixPhase when the audience is unknown."""
        pytest.importorskip("colorspacious")
        import numpy as np
        from colorspacious import cspace_convert

        samples = np.exp(1j * np.linspace(0, 2 * np.pi, 12, endpoint=False))
        deuteranomaly = {"name": "sRGB1+CVD", "cvd_type": "deuteranomaly", "severity": 100}

        def worst_separation(colormap_name: str) -> float:
            rgb = np.clip(getattr(cp, colormap_name)().rgb(samples), 0, 1)
            simulated = np.clip(cspace_convert(rgb, deuteranomaly, "sRGB1"), 0, 1)
            lab = cspace_convert(simulated, "sRGB1", "CAM02-UCS")
            distances = np.linalg.norm(lab[:, None, :] - lab[None, :, :], axis=-1)
            np.fill_diagonal(distances, np.inf)
            return float(distances.min())

        cubehelix = worst_separation("CubehelixPhase")
        others = {
            name: worst_separation(name)
            for name in ("Phase", "OklabPhase", "PerceptualPastel", "Isoluminant")
        }
        assert all(cubehelix > value for value in others.values()), (
            "docs/guide/reading-a-portrait.md recommends CubehelixPhase for colour-vision "
            f"deficiency, but it no longer separates best: {cubehelix:.1f} vs {others}"
        )

    def test_the_folding_families_are_still_documented_as_folding(self):
        """The guide says these two map phi and pi - phi to one colour. Check it is still so."""
        import numpy as np

        samples = np.exp(1j * np.linspace(0, 2 * np.pi, 12, endpoint=False))
        for name in ("DivergingWarmCool", "EarthTopographic"):
            rgb = np.round(np.clip(getattr(cp, name)().rgb(samples), 0, 1), 3)
            distinct = len({tuple(colour) for colour in rgb})
            assert distinct < 12, (
                f"{name} now assigns a distinct colour to every phase; the guide describes it "
                "as folding the phase circle and should be corrected"
            )
