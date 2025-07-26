"""Tests for modulus scaling methods."""

import numpy as np
import pytest
from complexplorer.core.scaling import ModulusScaling, SCALING_PRESETS, get_scaling_preset


class TestModulusScaling:
    """Test ModulusScaling methods."""
    
    def test_constant_scaling(self):
        """Test constant scaling method."""
        moduli = np.array([0, 1, 2, 10, np.inf])
        
        # Default radius
        result = ModulusScaling.constant(moduli)
        assert np.all(result == 1.0)
        
        # Custom radius
        result = ModulusScaling.constant(moduli, radius=2.5)
        assert np.all(result == 2.5)
    
    def test_linear_scaling(self):
        """Test linear scaling method."""
        moduli = np.array([0, 1, 2, 10])
        
        # Default scale
        result = ModulusScaling.linear(moduli)
        expected = 1.0 + 0.1 * moduli
        np.testing.assert_array_almost_equal(result, expected)
        
        # Custom scale
        result = ModulusScaling.linear(moduli, scale=0.5)
        expected = 1.0 + 0.5 * moduli
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_arctan_scaling(self):
        """Test arctangent scaling method."""
        moduli = np.array([0, 1, 10, 100, np.inf])
        
        result = ModulusScaling.arctan(moduli)
        
        # Check bounds
        assert result[0] == 0.5  # r_min for modulus 0
        assert result[-1] <= 1.5  # Should approach but not exceed r_max
        
        # Check monotonicity
        finite_mask = np.isfinite(moduli)
        assert np.all(np.diff(result[finite_mask]) >= 0)
        
        # Custom range
        result = ModulusScaling.arctan(moduli, r_min=0.2, r_max=2.0)
        assert result[0] == 0.2
        assert result[-1] <= 2.0
    
    def test_logarithmic_scaling(self):
        """Test logarithmic scaling method."""
        moduli = np.array([0, 0.1, 1, 10, 100])
        
        result = ModulusScaling.logarithmic(moduli)
        
        # Check that it handles zero
        assert np.isfinite(result[0])
        
        # Check bounds
        assert np.all(result >= 0.5)
        assert np.all(result <= 1.5)
        
        # Check different base
        result_base10 = ModulusScaling.logarithmic(moduli, base=10)
        assert not np.array_equal(result, result_base10)
    
    def test_linear_clamp_scaling(self):
        """Test linear clamp scaling method."""
        moduli = np.array([0, 5, 10, 20, 100])
        
        result = ModulusScaling.linear_clamp(moduli, m_max=10)
        
        # Check clamping
        assert result[0] == 0.5  # modulus 0 -> r_min
        assert result[2] == 1.5  # modulus 10 -> r_max
        assert result[3] == 1.5  # modulus 20 -> r_max (clamped)
        assert result[4] == 1.5  # modulus 100 -> r_max (clamped)
    
    def test_power_scaling(self):
        """Test power scaling method."""
        moduli = np.array([0, 1, 2, 4])
        
        # Square root scaling (default)
        result = ModulusScaling.power(moduli)
        
        # Check normalization
        assert result[0] == 0.5  # min value
        assert result[-1] == 1.5  # max value
        
        # Different exponent
        result_squared = ModulusScaling.power(moduli, exponent=2.0)
        assert not np.array_equal(result, result_squared)
    
    def test_custom_scaling(self):
        """Test custom scaling method."""
        moduli = np.array([0, 1, 2, 3, 4])
        
        # Simple custom function
        def custom_func(m):
            return np.tanh(m)
        
        result = ModulusScaling.custom(moduli, custom_func)
        
        # Check bounds
        assert np.all(result >= 0.5)
        assert np.all(result <= 1.5)
        
        # Check that it uses the custom function
        expected_normalized = np.tanh(moduli)
        expected = 0.5 + (1.5 - 0.5) * expected_normalized
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_sigmoid_scaling(self):
        """Test sigmoid scaling method."""
        moduli = np.array([0, 0.5, 1, 2, 10])
        
        result = ModulusScaling.sigmoid(moduli)
        
        # Check bounds
        assert np.all(result >= 0.5)
        assert np.all(result <= 1.5)
        
        # Check center point (modulus=1 should be near middle)
        center_idx = 2  # modulus = 1
        assert 0.9 < result[center_idx] < 1.1
        
        # Test different steepness
        result_steep = ModulusScaling.sigmoid(moduli, steepness=5.0)
        result_gentle = ModulusScaling.sigmoid(moduli, steepness=0.5)
        
        # Steeper should have sharper transition
        diff_steep = result_steep[-1] - result_steep[0]
        diff_gentle = result_gentle[-1] - result_gentle[0]
        assert diff_steep > diff_gentle
    
    def test_adaptive_scaling(self):
        """Test adaptive percentile-based scaling."""
        # Data with outliers
        moduli = np.array([0, 1, 2, 3, 4, 5, 100, 1000])
        
        result = ModulusScaling.adaptive(moduli)
        
        # Should ignore outliers and scale based on percentiles
        assert np.all(result >= 0.5)
        assert np.all(result <= 1.5)
        
        # Test with all same values
        moduli_same = np.ones(10)
        result_same = ModulusScaling.adaptive(moduli_same)
        assert np.all(result_same == 1.0)  # Should return middle value
        
        # Test with empty/invalid data
        moduli_invalid = np.array([np.inf, np.nan, np.inf])
        result_invalid = ModulusScaling.adaptive(moduli_invalid)
        assert np.all(result_invalid == 0.5)  # Should return r_min
    
    def test_hybrid_scaling(self):
        """Test hybrid linear-logarithmic scaling."""
        moduli = np.array([0, 0.5, 1, 2, 10, 100])
        
        result = ModulusScaling.hybrid(moduli, transition=1.0)
        
        # Check bounds
        assert np.all(result >= 0.5)
        assert np.all(result <= 1.5)
        
        # Check linear part (below transition)
        assert result[0] == 0.5  # modulus 0
        assert 0.74 < result[1] < 0.76  # modulus 0.5 should be ~0.75
        
        # Check logarithmic part (above transition)
        assert result[3] > result[2]  # Should continue increasing
        assert result[-1] < 1.5  # Should approach but not exceed r_max


class TestScalingPresets:
    """Test scaling preset functionality."""
    
    def test_preset_availability(self):
        """Test that all documented presets exist."""
        expected_presets = {'balanced', 'detail_near_zero', 'auto', 
                          'high_contrast', 'poles_emphasis'}
        assert set(SCALING_PRESETS.keys()) == expected_presets
    
    def test_get_scaling_preset(self):
        """Test get_scaling_preset function."""
        # Valid preset
        preset = get_scaling_preset('balanced')
        assert preset['method'] == 'sigmoid'
        assert 'steepness' in preset['params']
        
        # Invalid preset
        with pytest.raises(ValueError) as exc_info:
            get_scaling_preset('nonexistent')
        assert 'Unknown preset' in str(exc_info.value)
        assert 'Available presets' in str(exc_info.value)
    
    def test_preset_usage(self):
        """Test using presets with ModulusScaling."""
        moduli = np.array([0, 1, 2, 10])
        
        # Test each preset
        for name in SCALING_PRESETS:
            preset = get_scaling_preset(name)
            method = getattr(ModulusScaling, preset['method'])
            
            # Should not raise any errors
            result = method(moduli, **preset['params'])
            
            # Basic checks
            assert len(result) == len(moduli)
            assert np.all(np.isfinite(result[np.isfinite(moduli)]))
    
    def test_preset_params_are_copied(self):
        """Test that preset params are copied, not referenced."""
        preset1 = get_scaling_preset('balanced')
        preset2 = get_scaling_preset('balanced')
        
        # Modify one
        preset1['params']['steepness'] = 10.0
        
        # Other should be unchanged
        assert preset2['params']['steepness'] == 2.0


class TestScalingEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_array(self):
        """Test scaling with empty arrays."""
        moduli = np.array([])
        
        # All methods should handle empty arrays
        assert len(ModulusScaling.constant(moduli)) == 0
        assert len(ModulusScaling.linear(moduli)) == 0
        assert len(ModulusScaling.arctan(moduli)) == 0
        assert len(ModulusScaling.sigmoid(moduli)) == 0
    
    def test_infinite_values(self):
        """Test scaling with infinite values."""
        moduli = np.array([0, 1, np.inf])
        
        # Most methods should handle infinities gracefully
        result_arctan = ModulusScaling.arctan(moduli)
        assert np.isfinite(result_arctan).all()
        
        result_sigmoid = ModulusScaling.sigmoid(moduli)
        assert np.isfinite(result_sigmoid).all()
    
    def test_nan_values(self):
        """Test scaling with NaN values."""
        moduli = np.array([0, 1, np.nan, 2])
        
        # Methods should propagate NaN
        result = ModulusScaling.linear(moduli)
        assert np.isnan(result[2])
        assert np.isfinite(result[0])
        assert np.isfinite(result[1])
        assert np.isfinite(result[3])
    
    def test_negative_moduli(self):
        """Test that methods handle negative moduli (shouldn't happen but be robust)."""
        moduli = np.array([-1, 0, 1])
        
        # Should still produce valid results
        result = ModulusScaling.arctan(moduli)
        assert np.all(np.isfinite(result))