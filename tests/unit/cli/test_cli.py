"""Tests for the Complexplorer CLI (driven via main([...]))."""

import warnings

import pytest

from complexplorer.cli.main import _parse_cmap, _parse_domain, main
from complexplorer.utils.validation import ValidationError


class TestShorthandParsers:
    def test_domain_shorthands(self):
        assert _parse_domain("rect:4:4") == {"type": "rectangle", "re_length": 4, "im_length": 4}
        assert _parse_domain("disk:2") == {"type": "disk", "radius": 2}
        assert _parse_domain("annulus:0.2:3") == {
            "type": "annulus",
            "inner_radius": 0.2,
            "outer_radius": 3,
        }

    def test_bad_domain_raises(self):
        with pytest.raises(ValidationError):
            _parse_domain("triangle:1:2:3")

    def test_cmap_shorthand(self):
        assert _parse_cmap("phase:6") == {"type": "Phase", "n_phi": 6}
        assert _parse_cmap("phase") == {"type": "Phase"}


class TestListCommand:
    """`list` needs no PyVista (base-lane safe)."""

    def test_list(self, capsys):
        assert main(["list"]) == 0
        out = capsys.readouterr().out
        assert "pole_flower_10" in out and "identity" in out

    def test_list_tag(self, capsys):
        assert main(["list", "--tag", "singularity-detective"]) == 0
        out = capsys.readouterr().out
        assert "pole_flower_10" in out and "identity" not in out


class TestRender2D:
    """2D render works without PyVista (matplotlib Agg via conftest)."""

    def test_render_preset(self, tmp_path):
        warnings.simplefilter("ignore")
        out = tmp_path / "a.png"
        assert main(["render", "preset:rational_zeros_poles", "-o", str(out)]) == 0
        assert out.exists() and out.stat().st_size > 0

    def test_render_expression_with_domain_and_cmap(self, tmp_path):
        warnings.simplefilter("ignore")
        out = tmp_path / "b.png"
        rc = main(
            ["render", "z**2 - 1", "--domain", "rect:4:4", "--cmap", "phase:8", "-o", str(out)]
        )
        assert rc == 0 and out.exists()

    def test_bad_expression_exits_2(self, tmp_path):
        assert main(["render", "z.real", "-o", str(tmp_path / "x.png")]) == 2

    def test_missing_output_and_show_exits_2(self):
        assert main(["render", "preset:identity"]) == 2


class TestPyVistaAbsent:
    def test_stl_without_pyvista_fails_clearly(self, monkeypatch, tmp_path, capsys):
        import complexplorer.cli.main as climain

        monkeypatch.setattr(climain, "HAS_PYVISTA", False)
        rc = main(["stl", "preset:pole_flower_10", "-o", str(tmp_path / "f.stl")])
        assert rc == 2
        assert "PyVista" in capsys.readouterr().err

    def test_riemann_render_without_pyvista_fails_clearly(self, monkeypatch, tmp_path):
        import complexplorer.cli.main as climain

        monkeypatch.setattr(climain, "HAS_PYVISTA", False)
        assert (
            main(
                [
                    "render",
                    "preset:pole_flower_10",
                    "--mode",
                    "riemann",
                    "-o",
                    str(tmp_path / "c.png"),
                ]
            )
            == 2
        )


class TestPyVistaCommands:
    """3D/STL paths require PyVista."""

    def test_render_riemann(self, tmp_path):
        pytest.importorskip("pyvista")
        warnings.simplefilter("ignore")
        out = tmp_path / "c.png"
        rc = main(
            [
                "render",
                "preset:pole_flower_10",
                "--mode",
                "riemann",
                "--scaling",
                "arctan",
                "-o",
                str(out),
            ]
        )
        assert rc == 0 and out.exists() and out.stat().st_size > 0

    def test_stl(self, tmp_path):
        pytest.importorskip("pyvista")
        warnings.simplefilter("ignore")
        out = tmp_path / "f.stl"
        rc = main(["stl", "preset:pole_flower_10", "--resolution", "40", "-o", str(out)])
        assert rc == 0 and out.exists() and out.stat().st_size > 0
