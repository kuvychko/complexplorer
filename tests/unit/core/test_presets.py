"""Unit tests for the function preset registry (cp.catalog) and spec factories."""

import json
import warnings

import numpy as np
import pytest

import complexplorer as cp
from complexplorer.core.colormap import Chessboard, Phase
from complexplorer.core.domain import Annulus, Disk, Rectangle
from complexplorer.core.presets import (
    FunctionPreset,
    catalog,
    cmap_from_spec,
    domain_from_spec,
    scaling_from_spec,
    singularity,
)
from complexplorer.utils.validation import ValidationError


class TestSpecFactories:
    def test_domain_from_spec(self):
        assert isinstance(
            domain_from_spec({"type": "rectangle", "re_length": 4, "im_length": 4}), Rectangle
        )
        assert isinstance(domain_from_spec({"type": "disk", "radius": 2}), Disk)
        d = domain_from_spec({"type": "annulus", "inner_radius": 0.2, "outer_radius": 3})
        assert isinstance(d, Annulus)

    def test_domain_complex_center_pair(self):
        # center is a complex constructor kwarg, stored as [re, im]
        d = domain_from_spec({"type": "disk", "radius": 1, "center": [1, 2]})
        assert d.center == complex(1, 2)

    def test_cmap_from_spec(self):
        assert isinstance(cmap_from_spec({"type": "Phase", "n_phi": 6}), Phase)
        assert isinstance(cmap_from_spec({"type": "Chessboard", "spacing": 0.5}), Chessboard)

    def test_unknown_types_raise(self):
        with pytest.raises(ValidationError):
            domain_from_spec({"type": "trapezoid"})
        with pytest.raises(ValidationError):
            cmap_from_spec({"type": "Rainbow"})

    def test_scaling_from_spec(self):
        named = scaling_from_spec("balanced")
        assert "method" in named and "params" in named
        inline = scaling_from_spec({"method": "arctan", "params": {"r_min": 0.5}})
        assert inline["method"] == "arctan"


class TestSingularityRecords:
    def test_valid_record(self):
        rec = singularity("pole", 1 + 2j, 1, "p")
        assert rec == {"type": "pole", "at": [1.0, 2.0], "order": 1, "label": "p"}

    def test_unknown_type_raises(self):
        with pytest.raises(ValidationError):
            singularity("vortex", 0, 1)

    def test_essential_must_have_null_order(self):
        with pytest.raises(ValidationError):
            singularity("essential", 0, 2)
        assert singularity("essential", 0, None)["order"] is None


class TestFunctionPreset:
    def test_to_dict_excludes_callable_and_is_json(self):
        p = catalog.get("pole_flower_10")
        d = p.to_dict()
        assert "func" not in d and "callable" not in d
        json.dumps(d)  # raises if not serializable
        assert d["expression"] == "z / (z**10 - 1)"

    def test_specs_round_trip_to_live_objects(self):
        p = catalog.get("reciprocal")
        assert isinstance(p.domain(), Annulus)
        assert isinstance(p.colormap(), Phase)
        assert "method" in p.scaling()

    def test_bad_singularity_rejected_at_construction(self):
        with pytest.raises(ValidationError):
            FunctionPreset(
                id="bad",
                title="bad",
                expression="z",
                func=lambda z: z,
                singularities=({"type": "wormhole", "at": [0, 0], "order": 1},),
            )


class TestCatalog:
    def test_get_and_unknown_id(self):
        assert catalog.get("identity").id == "identity"
        with pytest.raises(ValidationError):
            catalog.get("does_not_exist")

    def test_list_unique_ids(self):
        ids = catalog.list()
        assert len(ids) == len(set(ids)) >= 15

    def test_filter_by_tag(self):
        found = catalog.filter(tag="singularity-detective")
        assert found and all("singularity-detective" in p.tags for p in found)

    def test_every_preset_round_trips_and_callable_finite(self):
        warnings.simplefilter("ignore")
        sample = np.array([0.37 + 0.41j])  # generic point away from curated singularities
        for pid in catalog.list():
            p = catalog.get(pid)
            p.domain()
            p.colormap()
            p.scaling()
            json.dumps(p.to_dict())
            assert np.all(np.isfinite(p.func(sample))), pid

    def test_pole_flower_answer_key_exact(self):
        p = catalog.get("pole_flower_10")
        zeros = [s for s in p.singularities if s["type"] == "zero"]
        poles = [s for s in p.singularities if s["type"] == "pole"]
        assert len(zeros) == 1 and zeros[0]["order"] == 1
        assert len(poles) == 10 and all(s["order"] == 1 for s in poles)


class TestPublicNaming:
    def test_catalog_and_Presets_both_exist_and_differ(self):
        assert cp.catalog is catalog  # function registry
        assert cp.Presets is not cp.catalog  # plot-config presets (distinct concept)
        assert hasattr(cp.Presets, "publication_ready")  # plot config
        assert hasattr(cp.catalog, "get")  # registry


def test_presets_module_is_pyvista_free():
    import complexplorer.core.presets as presets_mod

    with open(presets_mod.__file__, encoding="utf-8") as fh:
        text = fh.read()
    assert "import pyvista" not in text and "pyvista" not in text
