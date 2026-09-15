"""Tests for the exception hierarchy (add-exception-hierarchy)."""

import pytest

import complexplorer as cp
from complexplorer.exceptions import ComplexplorerError, ValidationError


class TestHierarchy:
    def test_base_is_exception(self):
        assert issubclass(ComplexplorerError, Exception)

    def test_validation_error_is_complexplorer_error(self):
        assert issubclass(ValidationError, ComplexplorerError)

    def test_validation_error_is_value_error(self):
        # Pre-3.0 handlers caught ValueError; that must keep working.
        assert issubclass(ValidationError, ValueError)

    def test_historical_import_path(self):
        from complexplorer.utils.validation import ValidationError as LegacyValidationError

        assert LegacyValidationError is ValidationError


class TestTopLevelExports:
    def test_names_exported(self):
        assert cp.ComplexplorerError is ComplexplorerError
        assert cp.ValidationError is ValidationError

    def test_names_in_all(self):
        assert "ComplexplorerError" in cp.__all__
        assert "ValidationError" in cp.__all__


class TestLibraryErrorsDeriveFromBase:
    def test_quick_plot_unknown_mode(self):
        with pytest.raises(ComplexplorerError):
            cp.quick_plot(lambda z: z, mode="bogus")

    def test_unknown_scaling_preset(self):
        with pytest.raises(ComplexplorerError):
            cp.get_scaling_preset("bogus")

    def test_unknown_scaling_preset_still_value_error(self):
        with pytest.raises(ValueError):
            cp.get_scaling_preset("bogus")

    def test_ornament_save_before_generate(self):
        gen = cp.OrnamentGenerator(lambda z: z, resolution=20)
        with pytest.raises(ComplexplorerError):
            gen.save_stl("never_written.stl")


class TestColormapError:
    """`ColormapError` restores the 2.0.0 name for invalid colormap configuration."""

    def test_is_a_validation_error(self):
        assert issubclass(cp.ColormapError, cp.ValidationError)
        assert issubclass(cp.ColormapError, cp.ComplexplorerError)
        assert issubclass(cp.ColormapError, ValueError)

    def test_caught_by_every_documented_handler(self):
        for handler in (cp.ColormapError, cp.ValidationError, cp.ComplexplorerError, ValueError):
            with pytest.raises(handler):
                raise cp.ColormapError("bad spacing")

    def test_is_a_top_level_export(self):
        from complexplorer.exceptions import ColormapError

        assert cp.ColormapError is ColormapError
        assert "ColormapError" in cp.__all__
