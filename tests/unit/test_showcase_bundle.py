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


def test_colormap_gallery_covers_every_exported_colormap():
    """The gallery must not drift from the public colormap surface (examples capability)."""
    import inspect
    import re

    import complexplorer as cp
    from complexplorer.core.colormap import BasePhasePortrait, Colormap

    exported = {
        name
        for name in cp.__all__
        if inspect.isclass(getattr(cp, name))
        and issubclass(getattr(cp, name), Colormap)
        and getattr(cp, name) not in (Colormap, BasePhasePortrait)
        and not inspect.isabstract(getattr(cp, name))
    }
    rendered = set()
    for entry in _manifest()["colormaps"]["renders"]:
        match = re.match(r"cp\.(\w+)\(", entry["ctor"])
        assert match, f"unrecognised constructor snippet: {entry['ctor']}"
        rendered.add(match.group(1))

    assert not exported - rendered, (
        f"public colormaps missing from the gallery: {sorted(exported - rendered)}"
    )
    assert not rendered - exported, (
        f"gallery renders a colormap that is not exported: {sorted(rendered - exported)}"
    )


# --- the curated tour, hero and thumbnails (curate-rev3-visual-tour) ---------------------

REQUIRED_TOUR = {
    "legend_portrait",
    "portrait_to_landscape",
    "engineering_figure",
    "composition_proof",
    "composite_domain",
    "sphere_vs_surface",
    "physical_triptych",
    "orbit_loop",
}


def test_tour_covers_the_required_capabilities_and_its_files_exist():
    """The tour is what shows the capabilities the registry cannot express."""
    tour = {rec["id"]: rec for rec in _manifest().get("tour", [])}
    assert REQUIRED_TOUR <= set(tour), f"missing tour assets: {sorted(REQUIRED_TOUR - set(tour))}"
    for rec in tour.values():
        assert (GALLERY / rec["file"]).is_file(), rec["file"]


def test_every_tour_entry_carries_its_recipe_and_prose():
    for rec in _manifest().get("tour", []):
        assert rec.get("caption"), f"{rec['id']} has no caption"
        assert rec.get("alt"), f"{rec['id']} has no alt text"
        assert rec.get("snippet"), f"{rec['id']} has no snippet"
        assert rec.get("inputs"), f"{rec['id']} does not record what produced it"
        assert rec["alt"] != rec["title"], f"{rec['id']} alt text just repeats the title"


def test_every_render_has_a_thumbnail():
    manifest = _manifest()
    for rec in manifest["presets"]:
        thumbs = rec.get("thumbs", {})
        assert set(thumbs) == set(rec["renders"]), rec["id"]
        for rel in thumbs.values():
            assert (GALLERY / rel).is_file(), rel
    for entry in manifest["colormaps"]["renders"]:
        assert (GALLERY / entry["thumb"]).is_file(), entry["name"]


def test_curated_assets_are_recorded_and_carry_no_recipe():
    """A photograph must never be mistaken for - or overwritten by - a render."""
    manifest = _manifest()
    rendered = {rel for rec in manifest["presets"] for rel in rec["renders"].values()}
    rendered |= {entry["file"] for entry in manifest["colormaps"]["renders"]}
    rendered |= {rec["file"] for rec in manifest.get("tour", [])}
    for entry in manifest.get("curated", []):
        assert entry.get("kind"), entry
        assert entry.get("origin"), entry
        assert "snippet" not in entry and "inputs" not in entry, "curated assets have no recipe"
        assert (GALLERY / entry["file"]).is_file(), entry["file"]
        assert entry["file"] not in rendered, f"{entry['file']} is both curated and generated"


def test_hero_montage_is_recorded_with_its_sources():
    hero = _manifest().get("hero", [])
    assert hero, "no hero montage recorded"
    for rec in hero:
        assert (GALLERY / rec["file"]).is_file(), rec["file"]
        assert len(rec["panels"]) == 6, "the hero covers six ideas"
        assert rec.get("sources"), "the hero records the assets it composed"


def test_the_orbit_loop_stays_within_its_size_budget():
    loop = next(rec for rec in _manifest()["tour"] if rec["id"] == "orbit_loop")
    size_mb = (GALLERY / loop["file"]).stat().st_size / 1_000_000
    assert size_mb < 3.5, f"the rotating loop grew to {size_mb:.1f} MB"


def test_every_snippet_parses_and_uses_only_public_names():
    """A snippet that no longer runs is worse than no snippet."""
    import ast

    import complexplorer as cp

    manifest = _manifest()
    snippets = [(rec["id"], rec["snippet"]) for rec in manifest.get("tour", [])]
    snippets += [
        (entry["name"], "import complexplorer as cp\n" + entry["ctor"])
        for entry in manifest["colormaps"]["renders"]
    ]

    public = set(cp.__all__)
    ee_public = {name for name in dir(cp.ee) if not name.startswith("_")}
    for name, snippet in snippets:
        tree = ast.parse(snippet)  # raises SyntaxError if the snippet ever stops parsing
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            value = node.value
            if isinstance(value, ast.Name) and value.id == "cp":
                assert node.attr in public, f"{name}: cp.{node.attr} is not exported"
            elif (
                isinstance(value, ast.Attribute)
                and value.attr == "ee"
                and isinstance(value.value, ast.Name)
                and value.value.id == "cp"
            ):
                assert node.attr in ee_public, f"{name}: cp.ee.{node.attr} does not exist"
