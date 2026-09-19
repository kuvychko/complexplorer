"""Guards for the things a release promises in writing.

These are cheap checks on text that drifts silently: claims nobody measured creeping back into the
README, a citation file that still says the previous version, a removed name that never made it
into the migration guide.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import complexplorer as cp

REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / "README.md"
CHANGELOG = REPO_ROOT / "CHANGELOG.md"
CITATION = REPO_ROOT / "CITATION.cff"
MIGRATION = REPO_ROOT / "docs" / "migration-3.0.md"


class TestTheRetiredBackendHelpers:
    """They were exported by the published 2.0.0, so their removal must be documented."""

    @pytest.mark.parametrize("name", ["setup_matplotlib_backend", "ensure_interactive_plots"])
    def test_absent_from_the_public_surface(self, name):
        assert name not in cp.__all__
        assert not hasattr(cp, name), f"{name} is still reachable as complexplorer.{name}"

    @pytest.mark.parametrize("name", ["setup_matplotlib_backend", "ensure_interactive_plots"])
    def test_named_in_the_migration_guide(self, name):
        assert name in MIGRATION.read_text(encoding="utf-8"), (
            f"{name} was removed from the public API but the migration guide does not mention it"
        )

    @pytest.mark.parametrize("name", ["setup_matplotlib_backend", "ensure_interactive_plots"])
    def test_still_available_internally(self, name):
        """They were retired from the public surface, not deleted; the library still uses them."""
        import complexplorer.utils.backend as backend

        assert hasattr(backend, name)


class TestClaimsStaySupportable:
    """Phrases retired before the 3.0 release, which must not come back."""

    RETIRED = [
        "15-30x",
        "15–30x",
        "cinema-quality",
        "first library",
        "no supports needed",
        "ultimate visualization",
    ]

    @pytest.mark.parametrize("phrase", RETIRED)
    def test_the_readme_does_not_carry_it(self, phrase):
        text = README.read_text(encoding="utf-8").lower()
        assert phrase.lower() not in text, (
            f"the README claims {phrase!r} again; it was removed before 3.0 because nothing in "
            f"the repository measures or substantiates it"
        )

    @pytest.mark.parametrize("phrase", RETIRED)
    def test_the_package_description_does_not_carry_it(self, phrase):
        description = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
        assert phrase.lower() not in description

    def test_the_readme_uses_absolute_urls(self):
        """PyPI renders the README outside the repository, so relative paths break."""
        text = README.read_text(encoding="utf-8")
        links = re.findall(r"\]\(([^)]+)\)", text) + re.findall(r'src="([^"]+)"', text)
        relative = [link for link in links if not link.startswith(("http", "#"))]
        assert not relative, f"these would not resolve on PyPI: {relative}"


class TestCitationMetadata:
    def test_the_version_matches_the_package(self):
        text = CITATION.read_text(encoding="utf-8")
        match = re.search(r"^version:\s*(\S+)", text, re.M)
        assert match, "CITATION.cff has no version field"
        assert match.group(1) == cp.__version__, (
            f"CITATION.cff says {match.group(1)}, the package says {cp.__version__}"
        )

    def test_it_names_the_repository_and_licence(self):
        text = CITATION.read_text(encoding="utf-8")
        assert "repository-code:" in text
        assert "license: MIT" in text


class TestTheChangelogIsTruthful:
    def test_the_published_baseline_is_not_described_as_unpublished(self):
        """2.0.0 is on PyPI (uploaded 2025-10-19); the changelog said it never was."""
        text = CHANGELOG.read_text(encoding="utf-8")
        section = text[text.index("## [2.0.0]") : text.index("## [1.0.0]")]
        assert "never published" not in section.lower(), (
            "the 2.0.0 entry says it was never published, but it is the latest release on PyPI"
        )
        assert "2025-10-19" in section

    def test_compare_links_are_present(self):
        text = CHANGELOG.read_text(encoding="utf-8")
        for label in ("[Unreleased]:", "[3.0.0]:", "[2.0.0]:"):
            assert label in text, f"the changelog has no {label} compare link"
