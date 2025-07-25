"""Test auto-scaling functionality for Enhanced Phase portraits."""

import numpy as np
import pytest
from complexplorer import Phase


class TestPhaseAutoScale:
    """Test the auto-scaling feature for Enhanced Phase color maps."""
    
    def test_auto_scale_calculation(self):
        """Test that auto-scaling calculates correct r_linear_step."""
        # Test with n_phi = 6
        cmap = Phase(n_phi=6, auto_scale_r=True)
        expected_r_step = 2 * np.pi / 6  # ≈ 1.047
        assert np.isclose(cmap.r_linear_step, expected_r_step)
        
        # Test with n_phi = 12
        cmap = Phase(n_phi=12, auto_scale_r=True)
        expected_r_step = 2 * np.pi / 12  # ≈ 0.524
        assert np.isclose(cmap.r_linear_step, expected_r_step)
        
        # Test with n_phi = 24
        cmap = Phase(n_phi=24, auto_scale_r=True)
        expected_r_step = 2 * np.pi / 24  # ≈ 0.262
        assert np.isclose(cmap.r_linear_step, expected_r_step)
    
    def test_auto_scale_with_custom_radius(self):
        """Test auto-scaling with custom scale_radius."""
        # Test with scale_radius = 2.0
        cmap = Phase(n_phi=6, auto_scale_r=True, scale_radius=2.0)
        expected_r_step = 2 * np.pi / 6 * 2.0  # ≈ 2.094
        assert np.isclose(cmap.r_linear_step, expected_r_step)
        
        # Test with scale_radius = 0.5
        cmap = Phase(n_phi=12, auto_scale_r=True, scale_radius=0.5)
        expected_r_step = 2 * np.pi / 12 * 0.5  # ≈ 0.262
        assert np.isclose(cmap.r_linear_step, expected_r_step)
    
    def test_auto_scale_requires_n_phi(self):
        """Test that auto_scale_r=True requires n_phi to be specified."""
        with pytest.raises(ValueError, match="auto_scale_r=True requires n_phi"):
            Phase(auto_scale_r=True)
        
        with pytest.raises(ValueError, match="auto_scale_r=True requires n_phi"):
            Phase(auto_scale_r=True, r_linear_step=0.5)
    
    def test_auto_scale_conflicts_with_r_linear_step(self):
        """Test that auto_scale_r and r_linear_step cannot be used together."""
        with pytest.raises(ValueError, match="Cannot specify both auto_scale_r=True and r_linear_step"):
            Phase(n_phi=6, r_linear_step=0.5, auto_scale_r=True)
    
    def test_backward_compatibility(self):
        """Test that existing usage patterns still work."""
        # Basic phase portrait
        cmap1 = Phase()
        assert cmap1.phi is None
        assert cmap1.r_linear_step is None
        assert cmap1.auto_scale_r is False
        
        # Phase with sectors
        cmap2 = Phase(n_phi=6)
        assert cmap2.phi == np.pi / 6
        assert cmap2.r_linear_step is None
        assert cmap2.auto_scale_r is False
        
        # Enhanced phase with manual r_linear_step
        cmap3 = Phase(n_phi=6, r_linear_step=0.5, v_base=0.4)
        assert cmap3.phi == np.pi / 6
        assert cmap3.r_linear_step == 0.5
        assert cmap3.auto_scale_r is False
    
    def test_visual_output_consistency(self):
        """Test that auto-scaled and manually calculated values produce same output."""
        # Create test complex values
        z = np.array([1+0j, 0+1j, -1+0j, 0-1j, 0.5+0.5j])
        
        # Manual calculation for n_phi=6
        n_phi = 6
        manual_r_step = 2 * np.pi / n_phi
        cmap_manual = Phase(n_phi=n_phi, r_linear_step=manual_r_step, v_base=0.4)
        
        # Auto-scaled version
        cmap_auto = Phase(n_phi=n_phi, auto_scale_r=True, v_base=0.4)
        
        # Get HSV values
        hsv_manual = cmap_manual.hsv_tuple(z)
        hsv_auto = cmap_auto.hsv_tuple(z)
        
        # They should be identical
        np.testing.assert_array_almost_equal(hsv_manual, hsv_auto)
    
    def test_attributes_set_correctly(self):
        """Test that all attributes are set correctly with auto-scaling."""
        cmap = Phase(n_phi=12, auto_scale_r=True, scale_radius=1.5, v_base=0.3)
        
        # Check the phi value (pi/n_phi is stored, not n_phi itself)
        assert cmap.phi == np.pi / 12
        assert cmap.auto_scale_r is True
        assert cmap.scale_radius == 1.5
        assert cmap.v_base == 0.3
        assert np.isclose(cmap.r_linear_step, 2 * np.pi / 12 * 1.5)
    
    def test_edge_cases(self):
        """Test edge cases for auto-scaling."""
        # Very small n_phi
        cmap1 = Phase(n_phi=2, auto_scale_r=True)
        assert np.isclose(cmap1.r_linear_step, np.pi)
        
        # Large n_phi
        cmap2 = Phase(n_phi=100, auto_scale_r=True)
        assert np.isclose(cmap2.r_linear_step, 2 * np.pi / 100)
        
        # With logarithmic scaling (should still work)
        cmap3 = Phase(n_phi=6, auto_scale_r=True, r_log_base=2.0)
        assert np.isclose(cmap3.r_linear_step, 2 * np.pi / 6)
        assert cmap3.r_log_base == 2.0
    
    def test_integration_with_plots(self):
        """Test that auto-scaled colormaps work with actual plotting."""
        import complexplorer as cp
        
        # This should work without errors
        cmap = cp.Phase(n_phi=12, auto_scale_r=True, v_base=0.4)
        
        # Test with a simple domain
        domain = cp.Rectangle(2, 2)
        z = domain.mesh(50)
        
        # Get RGB values (should not raise errors)
        rgb = cmap.rgb(z**2)
        
        # Check output shape
        assert rgb.shape == (50, 50, 3)
        
        # Check values are in valid range
        assert np.all(rgb >= 0) and np.all(rgb <= 1)