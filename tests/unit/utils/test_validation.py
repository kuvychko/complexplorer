"""Tests for validation utilities.

Only ``validate_resolution`` and the ``ValidationError`` re-export remain in
``complexplorer.utils.validation`` as of 3.0 (the broader unused helper set was removed).
"""

import pytest

from complexplorer.exceptions import ComplexplorerError
from complexplorer.utils.validation import ValidationError, validate_resolution


class TestValidationError:
    """Test ValidationError re-export and hierarchy."""

    def test_validation_error_is_value_error(self):
        """ValidationError should be a subclass of ValueError (back-compat)."""
        assert issubclass(ValidationError, ValueError)

    def test_validation_error_is_complexplorer_error(self):
        """ValidationError should derive from the library base error."""
        assert issubclass(ValidationError, ComplexplorerError)

    def test_validation_error_message(self):
        """Test ValidationError with custom message."""
        with pytest.raises(ValidationError, match="Custom error"):
            raise ValidationError("Custom error")


class TestValidateResolution:
    """Test resolution validation."""

    def test_valid_resolution(self):
        """Test with valid resolution."""
        assert validate_resolution(100) == 100
        assert validate_resolution(100.5) == 100  # Float converted to int
        assert validate_resolution("200") == 200  # String converted to int

    def test_invalid_type(self):
        """Test with invalid type."""
        with pytest.raises(ValidationError, match="resolution must be an integer"):
            validate_resolution("not a number")

    def test_out_of_range(self):
        """Test with out of range value."""
        with pytest.raises(ValidationError, match="resolution must be between"):
            validate_resolution(5)  # Too small

        with pytest.raises(ValidationError, match="resolution must be between"):
            validate_resolution(2000)  # Too large

    def test_custom_range(self):
        """Test with custom range."""
        assert validate_resolution(50, min_val=1, max_val=100) == 50

        with pytest.raises(ValidationError):
            validate_resolution(150, min_val=1, max_val=100)
