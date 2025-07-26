"""Tests for validation utilities."""

import numpy as np
import pytest
import warnings
from complexplorer.utils.validation import (
    ValidationError,
    validate_domain_or_mesh,
    validate_function_or_values,
    validate_function,
    validate_colormap,
    validate_stl_parameters,
    validate_resolution,
    validate_array_shape,
    validate_file_extension,
    warn_deprecated
)


class TestValidationError:
    """Test custom ValidationError."""
    
    def test_validation_error_is_value_error(self):
        """ValidationError should be a subclass of ValueError."""
        assert issubclass(ValidationError, ValueError)
    
    def test_validation_error_message(self):
        """Test ValidationError with custom message."""
        with pytest.raises(ValidationError, match="Custom error"):
            raise ValidationError("Custom error")


class TestValidateDomainOrMesh:
    """Test domain/mesh validation."""
    
    def test_valid_domain(self):
        """Test with valid domain."""
        domain = object()  # Mock domain
        validate_domain_or_mesh(domain, None)  # Should not raise
    
    def test_valid_mesh(self):
        """Test with valid mesh."""
        mesh = np.array([1, 2, 3])
        validate_domain_or_mesh(None, mesh)  # Should not raise
    
    def test_both_provided(self):
        """Test with both domain and mesh."""
        domain = object()
        mesh = np.array([1, 2, 3])
        validate_domain_or_mesh(domain, mesh)  # Should not raise
    
    def test_both_none(self):
        """Test with both None."""
        with pytest.raises(ValidationError, match="Both domain and z parameters cannot be None"):
            validate_domain_or_mesh(None, None)
    
    def test_custom_param_names(self):
        """Test with custom parameter names."""
        with pytest.raises(ValidationError, match="Both region and points parameters"):
            validate_domain_or_mesh(None, None, param_names=('region', 'points'))


class TestValidateFunctionOrValues:
    """Test function/values validation."""
    
    def test_valid_function(self):
        """Test with valid function."""
        func = lambda z: z**2
        validate_function_or_values(func, None)  # Should not raise
    
    def test_valid_values(self):
        """Test with valid values."""
        values = np.array([1, 2, 3])
        validate_function_or_values(None, values)  # Should not raise
    
    def test_both_none(self):
        """Test with both None."""
        with pytest.raises(ValidationError, match="Both func and f parameters cannot be None"):
            validate_function_or_values(None, None)


class TestValidateFunction:
    """Test function validation."""
    
    def test_valid_function(self):
        """Test with valid function."""
        func = lambda z: z**2
        safe_func = validate_function(func)
        
        # Test it works
        result = safe_func(np.array([1+1j, 2+2j]))
        assert len(result) == 2
    
    def test_not_callable(self):
        """Test with non-callable."""
        with pytest.raises(ValidationError, match="Function parameter must be callable"):
            validate_function("not a function")
    
    def test_function_error_handling(self):
        """Test error handling in wrapped function."""
        def bad_func(z):
            raise RuntimeError("Function error")
        
        safe_func = validate_function(bad_func)
        
        with pytest.raises(ValidationError, match="Function evaluation failed"):
            safe_func(np.array([1, 2]))
    
    def test_scalar_to_array_conversion(self):
        """Test that scalar results are converted to arrays."""
        func = lambda z: 5  # Returns scalar
        safe_func = validate_function(func)
        
        result = safe_func(np.array([1, 2]))
        assert isinstance(result, np.ndarray)


class TestValidateColormap:
    """Test colormap validation."""
    
    def test_none_colormap_with_default(self):
        """Test None colormap with default class."""
        class MockCmap:
            def __init__(self, n_phi=6, auto_scale_r=True):
                self.n_phi = n_phi
                self.auto_scale_r = auto_scale_r
            def hsv(self): pass
            def rgb(self): pass
        
        cmap = validate_colormap(None, default_class=MockCmap)
        assert isinstance(cmap, MockCmap)
        assert cmap.n_phi == 6
        assert cmap.auto_scale_r is True
    
    def test_valid_colormap(self):
        """Test with valid colormap."""
        class ValidCmap:
            def hsv(self): pass
            def rgb(self): pass
        
        cmap = ValidCmap()
        result = validate_colormap(cmap)
        assert result is cmap
    
    def test_invalid_colormap_missing_method(self):
        """Test colormap missing required method."""
        class InvalidCmap:
            def hsv(self): pass
            # Missing rgb method
        
        with pytest.raises(ValidationError, match="Colormap must have 'rgb' method"):
            validate_colormap(InvalidCmap())


class TestValidateSTLParameters:
    """Test STL parameter validation."""
    
    def test_valid_parameters(self):
        """Test with valid parameters."""
        validate_stl_parameters(50.0, 2.0)  # Should not raise
    
    def test_invalid_size_type(self):
        """Test with non-numeric size."""
        with pytest.raises(ValidationError, match="Size must be numeric"):
            validate_stl_parameters("50", 2.0)
    
    def test_negative_size(self):
        """Test with negative size."""
        with pytest.raises(ValidationError, match="Size must be positive"):
            validate_stl_parameters(-10, 2.0)
    
    def test_size_out_of_range(self):
        """Test with size out of range."""
        with pytest.raises(ValidationError, match="Size must be between"):
            validate_stl_parameters(600, 2.0)
    
    def test_invalid_wall_thickness(self):
        """Test with invalid wall thickness."""
        with pytest.raises(ValidationError, match="Wall thickness must be positive"):
            validate_stl_parameters(50, -1)
    
    def test_wall_too_thick(self):
        """Test with wall thickness too large."""
        with pytest.raises(ValidationError, match="Wall thickness .* must be less than half"):
            validate_stl_parameters(50, 30)


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


class TestValidateArrayShape:
    """Test array shape validation."""
    
    def test_valid_array(self):
        """Test with valid array."""
        arr = np.array([[1, 2], [3, 4]])
        validate_array_shape(arr)  # Should not raise
    
    def test_not_array(self):
        """Test with non-array."""
        with pytest.raises(ValidationError, match="array must be a numpy array"):
            validate_array_shape([1, 2, 3])
    
    def test_expected_ndim(self):
        """Test expected dimensions."""
        arr = np.array([1, 2, 3])
        validate_array_shape(arr, expected_ndim=1)  # Should not raise
        
        with pytest.raises(ValidationError, match="array must have 2 dimensions"):
            validate_array_shape(arr, expected_ndim=2)
    
    def test_expected_shape(self):
        """Test expected shape."""
        arr = np.array([[1, 2], [3, 4]])
        validate_array_shape(arr, expected_shape=(2, 2))  # Should not raise
        
        with pytest.raises(ValidationError, match="array must have shape"):
            validate_array_shape(arr, expected_shape=(3, 3))


class TestValidateFileExtension:
    """Test file extension validation."""
    
    def test_valid_extension(self):
        """Test with valid extension."""
        assert validate_file_extension("file.stl", "stl") == "stl"
        assert validate_file_extension("FILE.STL", "stl") == "stl"  # Case insensitive
        assert validate_file_extension("file.png", ["png", "jpg"]) == "png"
    
    def test_invalid_type(self):
        """Test with non-string filename."""
        with pytest.raises(ValidationError, match="filename must be a string"):
            validate_file_extension(123, "stl")
    
    def test_empty_filename(self):
        """Test with empty filename."""
        with pytest.raises(ValidationError, match="filename cannot be empty"):
            validate_file_extension("", "stl")
    
    def test_no_extension(self):
        """Test filename without extension."""
        with pytest.raises(ValidationError, match="filename must have a file extension"):
            validate_file_extension("file", "stl")
    
    def test_invalid_extension(self):
        """Test with invalid extension."""
        with pytest.raises(ValidationError, match="Invalid file extension"):
            validate_file_extension("file.txt", "stl")
    
    def test_extension_with_dot(self):
        """Test that extensions with dots are handled."""
        assert validate_file_extension("file.stl", ".stl") == "stl"


class TestWarnDeprecated:
    """Test deprecation warnings."""
    
    def test_deprecation_warning(self):
        """Test deprecation warning is issued."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_deprecated("old_func", "new_func")
            
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "old_func is deprecated" in str(w[0].message)
            assert "Use new_func instead" in str(w[0].message)
    
    def test_custom_version(self):
        """Test with custom version."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_deprecated("old_func", "new_func", version="v2.0")
            
            assert "will be removed in v2.0" in str(w[0].message)