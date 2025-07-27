"""Tests for texture computation utilities."""

import numpy as np
import pytest
from complexplorer.utils.texture import (
    compute_texture_from_colormap,
    compute_sphere_gradient,
    compute_hsv_gradient,
    quantize_regions
)
from complexplorer.core.colormap import (
    Chessboard, PolarChessboard, LogRings, Phase
)


class TestComputeTextureFromColormap:
    """Test texture computation from colormaps."""
    
    def test_chessboard_binary_mode(self):
        """Test binary texture mode with chessboard."""
        # Create simple test data
        n = 5
        z = np.array([i + 1j*j for i in range(n) for j in range(n)], dtype=complex)
        cmap = Chessboard(spacing=1.0)
        
        # Binary mode should give -1 or 1 values
        texture = compute_texture_from_colormap(z, cmap, mode='binary')
        
        assert texture.shape == (n*n,)
        assert np.all(np.abs(texture) == 1.0)
        # Check alternating pattern
        assert texture[0] != texture[1]  # Adjacent squares differ
    
    def test_chessboard_ridge_mode(self):
        """Test ridge detection with chessboard."""
        n = 10
        z = np.array([i*0.5 + 1j*j*0.5 for i in range(n) for j in range(n)], dtype=complex)
        cmap = Chessboard(spacing=1.0)
        
        texture = compute_texture_from_colormap(z, cmap, mode='ridges')
        
        assert texture.shape == (n*n,)
        assert np.all(texture >= -1) and np.all(texture <= 1)  # In valid range
        assert np.any(texture > 0)   # Some ridges exist
    
    def test_phase_ridge_mode(self):
        """Test ridge detection with phase colormap."""
        n = 8
        # Create grid of complex values
        x = np.linspace(-1, 1, n)
        y = np.linspace(-1, 1, n)
        X, Y = np.meshgrid(x, y)
        z = X + 1j*Y
        z = z.ravel()
        
        cmap = Phase(n_phi=6)  # 6 phase sectors
        
        texture = compute_texture_from_colormap(
            z, cmap, mode='ridges'
        )
        
        assert texture.shape == (n*n,)
        # Should have ridges at phase boundaries
        assert np.sum(texture > 0) > 0
    
    def test_direct_computation_no_mesh_shape(self):
        """Test that direct computation doesn't need mesh_shape."""
        z = np.array([1+1j, 2+2j, 3+3j])
        cmap = Phase(n_phi=4)
        
        # Should work without mesh_shape
        texture = compute_texture_from_colormap(z, cmap, mode='ridges')
        assert texture.shape == (3,)
    
    def test_groove_mode(self):
        """Test groove mode produces negative values."""
        n = 5
        z = np.array([i + 1j*j for i in range(n) for j in range(n)], dtype=complex)
        cmap = PolarChessboard(n_phi=4)
        
        texture = compute_texture_from_colormap(z, cmap, mode='grooves')
        
        assert texture.shape == (n*n,)
        assert np.all(texture <= 1)  # In valid range
        assert np.any(texture < 0)  # Some grooves exist


class TestComputeSphereGradient:
    """Test gradient computation on sphere."""
    
    def test_basic_gradient(self):
        """Test gradient of linear function."""
        n_theta, n_phi = 10, 10
        # Create linear ramp in theta direction
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2*np.pi, n_phi)
        PHI, THETA = np.meshgrid(phi, theta)
        values = THETA.ravel()  # Linear in theta
        
        grad = compute_sphere_gradient(values, (n_theta, n_phi))
        
        assert grad.shape == values.shape
        assert np.all(grad >= 0)  # Magnitude is positive
        # Check gradient is reasonable (not extreme)
        assert np.median(grad) < 5.0  # Reasonable magnitude
    
    def test_zero_gradient(self):
        """Test gradient of constant function."""
        n_theta, n_phi = 8, 8
        values = np.ones(n_theta * n_phi)
        
        grad = compute_sphere_gradient(values, (n_theta, n_phi))
        
        assert np.allclose(grad, 0)
    
    def test_pole_handling(self):
        """Test gradient handles poles correctly."""
        n_theta, n_phi = 20, 20
        # Create function that varies in phi (worst at poles)
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2*np.pi, n_phi)
        PHI, THETA = np.meshgrid(phi, theta)
        values = np.sin(PHI).ravel()
        
        grad = compute_sphere_gradient(values, (n_theta, n_phi))
        
        assert grad.shape == values.shape
        assert np.all(np.isfinite(grad))  # No infinities at poles


class TestComputeHSVGradient:
    """Test HSV gradient computation."""
    
    def test_hue_gradient(self):
        """Test gradient detects hue changes."""
        n = 10
        # Create HSV with changing hue
        hsv = np.zeros((n*n, 3))
        hsv[:, 0] = np.linspace(0, 1, n*n)  # Linear hue ramp
        hsv[:, 1] = 1.0  # Full saturation
        hsv[:, 2] = 1.0  # Full value
        
        grad = compute_hsv_gradient(hsv, (n, n))
        
        assert grad.shape == (n*n,)
        assert np.all(grad >= 0)
        # Should have non-zero gradient due to hue change
        assert np.mean(grad) > 0
    
    def test_hue_wraparound(self):
        """Test hue wraparound handling."""
        n = 10
        hsv = np.zeros((n*n, 3))
        # Create sharp hue transition (wraparound)
        hsv[:n*n//2, 0] = 0.1
        hsv[n*n//2:, 0] = 0.9
        hsv[:, 1] = 1.0
        hsv[:, 2] = 1.0
        
        grad = compute_hsv_gradient(hsv, (n, n))
        
        assert np.all(np.isfinite(grad))
        # Gradient at wraparound should be reduced
        max_grad_idx = np.argmax(grad)
        assert max_grad_idx != n*n//2  # Not at the sharp transition
    
    def test_value_gradient(self):
        """Test gradient detects value changes."""
        n = 8
        hsv = np.zeros((n*n, 3))
        hsv[:, 0] = 0.5  # Constant hue
        hsv[:, 1] = 1.0  # Constant saturation
        hsv[:, 2] = np.linspace(0, 1, n*n)  # Value ramp
        
        grad = compute_hsv_gradient(hsv, (n, n))
        
        assert grad.shape == (n*n,)
        # Should detect value changes (with lower weight)
        assert np.mean(grad) > 0
        assert np.median(grad) < 1  # But less than pure hue gradient (use median to avoid outliers)


class TestQuantizeRegions:
    """Test region quantization."""
    
    def test_basic_quantization(self):
        """Test alternating pattern generation."""
        n = 10
        hsv = np.zeros((n*n, 3))
        # Create distinct hue levels
        for i in range(n*n):
            hsv[i, 0] = (i // n) / n
        edges = np.zeros(n*n, dtype=bool)
        
        result = quantize_regions(hsv, edges, (n, n))
        
        assert result.shape == (n*n,)
        assert set(np.unique(result)) <= {-1.0, 0.0, 1.0}
        # Should alternate between rows
        assert result[0] != result[n]
    
    def test_edges_neutral(self):
        """Test that edges are set to neutral value."""
        n = 5
        hsv = np.random.rand(n*n, 3)
        edges = np.zeros(n*n, dtype=bool)
        edges[::5] = True  # Mark some edges
        
        result = quantize_regions(hsv, edges, (n, n))
        
        assert np.all(result[edges] == 0.0)  # Edges are neutral


class TestTextureIntegration:
    """Test texture integration with PyVista mesh."""
    
    @pytest.mark.skipif(True, reason="Requires PyVista")
    def test_apply_texture_to_mesh(self):
        """Test applying texture to mesh (would need PyVista)."""
        # This test would require PyVista to create actual mesh
        # Skipped in unit tests, covered by integration tests
        pass