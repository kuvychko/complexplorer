"""Guard for the committed showcase gallery bundle (rebuild-gallery-from-registry, M2).

These checks are static: they load the committed ``showcase.json`` and inspect the gallery
tree. They do NOT render (off-screen screenshots crash on headless CI); image *correctness* is
a local concern. They enforce that the committed bundle matches the catalog and the tag policy.
"""

import json
from pathlib import Path

from complexplorer.core.presets import catalog

REPO_ROOT = Path(__file__).resolve().parents[2]
GALLERY = REPO_ROOT / "examples" / "gallery"
HERO = "Riemann_relief_map_20250726.png"

# The tag -> extra render policy (mirrors examples/showcase.py; the guard re-derives it
# independently so a drift between policy and committed bundle fails the test).
TAG_RENDERS = {
    "canonical": ("landscape", "sphere"),
    "branches": ("surface",),
    "ornament": ("ornament",),
}


def _manifest() -> dict:
    return json.loads((GALLERY / "showcase.json").read_text(encoding="utf-8"))


def _expected_render_types(preset) -> set[str]:
    types = {"portrait"}
    for tag in preset.tags:
        types.update(TAG_RENDERS.get(tag, ()))
    return types


def test_manifest_covers_exactly_the_catalog():
    m = _manifest()
    assert [r["id"] for r in m["presets"]] == catalog.list()


def test_render_set_matches_tag_policy_and_files_exist():
    m = _manifest()
    for rec in m["presets"]:
        preset = catalog.get(rec["id"])
        assert set(rec["renders"]) == _expected_render_types(preset), rec["id"]
        for rel in rec["renders"].values():
            assert (GALLERY / rel).is_file(), rel


def test_no_render_for_an_untagged_preset():
    """A preset with none of the render tags gets only a portrait."""
    m = _manifest()
    for rec in m["presets"]:
        tags = set(catalog.get(rec["id"]).tags)
        if not (tags & set(TAG_RENDERS)):
            assert set(rec["renders"]) == {"portrait"}, rec["id"]


def test_colormap_gallery_present_and_files_exist():
    m = _manifest()
    cm = m["colormaps"]
    assert cm["reference_preset"] in catalog
    assert cm["renders"], "expected a non-empty colormap gallery"
    for r in cm["renders"]:
        assert (GALLERY / r["file"]).is_file(), r["file"]


def test_hero_banner_recorded_and_present():
    m = _manifest()
    assert m["banner"] == HERO
    assert (GALLERY / HERO).is_file()


def test_no_stl_committed():
    assert not list(GALLERY.rglob("*.stl"))


def test_only_curated_hero_is_a_top_level_image():
    """Top-level gallery dir holds only manifests + the curated hero; renders live in <id>/."""
    top_level_pngs = [p.name for p in GALLERY.glob("*.png")]
    assert top_level_pngs == [HERO], top_level_pngs
