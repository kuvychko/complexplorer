"""Tests for the gallery generator (add-gallery-generator)."""

import json
import warnings

from complexplorer import generate_gallery
from complexplorer.cli.main import main
from complexplorer.core.presets import catalog

_TAG = "ornament"  # a tag the catalog carries (a handful of presets)


def _matching_ids(tag):
    return sorted(p.id for p in catalog.filter(tag=tag))


def test_generates_bundle_structure(tmp_path):
    warnings.simplefilter("ignore")
    manifest = generate_gallery(tmp_path, selection=_TAG, dpi=60)
    ids = _matching_ids(_TAG)
    assert [r["id"] for r in manifest["presets"]] == ids  # id-sorted
    assert (tmp_path / "index.json").exists()
    for pid in ids:
        assert (tmp_path / pid / "portrait.png").stat().st_size > 0
        card = json.loads((tmp_path / pid / "card.json").read_text(encoding="utf-8"))
        assert card["files"]["portrait"] == f"{pid}/portrait.png"  # relative, forward-slash
        # record keys == to_dict() keys + the gallery-added fields
        expected = set(catalog.get(pid).to_dict()) | {"files", "schema_version"}
        assert set(card) == expected


def test_index_is_self_contained(tmp_path):
    warnings.simplefilter("ignore")
    generate_gallery(tmp_path, selection=_TAG, dpi=60)
    index = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    assert index["schema_version"] == 2
    assert "complexplorer_version" in index and "generator" in index
    # each index record equals the corresponding card.json
    for rec in index["presets"]:
        card = json.loads((tmp_path / rec["id"] / "card.json").read_text(encoding="utf-8"))
        assert rec == card


def test_manifest_and_portraits_byte_identical_across_runs(tmp_path):
    """The determinism contract: two runs produce byte-identical output (within env)."""
    warnings.simplefilter("ignore")
    a, b = tmp_path / "a", tmp_path / "b"
    generate_gallery(a, selection=_TAG, dpi=60)
    generate_gallery(b, selection=_TAG, dpi=60)
    files = ["index.json"] + [
        f"{pid}/{name}" for pid in _matching_ids(_TAG) for name in ("card.json", "portrait.png")
    ]
    for rel in files:
        assert (a / rel).read_bytes() == (b / rel).read_bytes(), f"differs: {rel}"


def test_no_timestamp_in_portrait(tmp_path):
    warnings.simplefilter("ignore")
    pid = _matching_ids(_TAG)[0]
    generate_gallery(tmp_path, selection=[pid], dpi=60)
    png = (tmp_path / pid / "portrait.png").read_bytes()
    assert b"Software" not in png and b"tIME" not in png


def test_selection_forms(tmp_path):
    warnings.simplefilter("ignore")
    # explicit id list
    m = generate_gallery(tmp_path / "ids", selection=["identity"], dpi=60)
    assert [r["id"] for r in m["presets"]] == ["identity"]
    # all presets (None)
    m_all = generate_gallery(tmp_path / "all", selection=None, dpi=60)
    assert [r["id"] for r in m_all["presets"]] == sorted(catalog.list())


# ---- CLI ----


def test_cli_gallery(tmp_path):
    warnings.simplefilter("ignore")
    rc = main(["gallery", "--tag", _TAG, "-o", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "index.json").exists()


def test_cli_gallery_requires_output():
    assert main(["gallery", "--tag", _TAG]) == 2


def test_cli_gallery_unmatched_tag_exits_2(tmp_path, capsys):
    rc = main(["gallery", "--tag", "no-such-tag-xyz", "-o", str(tmp_path)])
    assert rc == 2
    assert "0 presets matched" in capsys.readouterr().err
