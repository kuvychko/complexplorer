"""Tests for direct texture computation."""

import numpy as np
import pytest
from complexplorer.utils.texture_direct import (
    compute_texture_direct,
    _chessboard_texture,
    _polar_chessboard_texture,
    _log_rings_texture,
    _phase_texture
)
from complexplorer.core.colormap import (
    Chessboard, PolarChessboard, LogRings, Phase
)


class TestChessboardTexture:
    """Test direct texture computation for Chessboard."""
    
    def test_binary_mode(self):
        """Test binary height mapping."""
        cmap = Chessboard(spacing=1.0)
        # Test points in different squares
        z = np.array([0.5+0.5j, 1.5+0.5j, 0.5+1.5j, 1.5+1.5j])
        
        texture = _chessboard_texture(z, cmap, 'binary')
        
        # Should alternate based on parity
        assert texture[0] == 1.0   # (0,0) white
        assert texture[1] == -1.0  # (1,0) black
        assert texture[2] == -1.0  # (0,1) black
        assert texture[3] == 1.0   # (1,1) white
    
    def test_ridge_mode(self):
        """Test ridge detection at boundaries."""
        cmap = Chessboard(spacing=1.0)
        # Points near boundaries
        z = np.array([0.05+0.5j, 0.95+0.5j, 0.5+0.05j, 0.5+0.5j])
        
        texture = _chessboard_texture(z, cmap, 'ridges')
        
        # First three should be near boundaries
        assert texture[0] > 0  # Near x=0
        assert texture[1] > 0  # Near x=1
        assert texture[2] > 0  # Near y=0
        assert texture[3] == 0  # Center of square
    
    def test_groove_mode(self):
        """Test groove mode."""
        cmap = Chessboard(spacing=1.0)
        z = np.array([0.05+0.5j, 0.5+0.5j])
        
        texture = _chessboard_texture(z, cmap, 'grooves')
        
        assert texture[0] < 0  # Groove at boundary
        assert texture[1] == 0  # No groove at center


class TestPolarChessboardTexture:
    """Test direct texture computation for PolarChessboard."""
    
    def test_binary_mode(self):
        """Test binary height mapping."""
        cmap = PolarChessboard(n_phi=4, spacing=1.0)
        # Test points at different angles and radii
        z = np.array([
            1.0 * np.exp(1j * 0),        # r=1, angle=0
            1.0 * np.exp(1j * np.pi/2),  # r=1, angle=π/2
            2.0 * np.exp(1j * 0),        # r=2, angle=0
        ])
        
        texture = _polar_chessboard_texture(z, cmap, 'binary')
        
        # Check alternating pattern
        assert np.all(np.abs(texture) == 1.0)
    
    def test_ridge_mode(self):
        """Test ridge detection."""
        cmap = PolarChessboard(n_phi=4, spacing=1.0)
        # Point near angular boundary
        angle_boundary = np.pi / 4 + 0.05
        z = np.array([np.exp(1j * angle_boundary)])
        
        texture = _polar_chessboard_texture(z, cmap, 'ridges')
        
        assert texture[0] > 0  # Should detect ridge


class TestLogRingsTexture:
    """Test direct texture computation for LogRings."""
    
    def test_binary_mode(self):
        """Test alternating rings."""
        cmap = LogRings(log_spacing=np.log(2))
        # Powers of 2 should alternate
        z = np.array([1.0, 2.0, 4.0, 0.5])
        
        texture = _log_rings_texture(z, cmap, 'binary')
        
        # Check alternating pattern
        assert texture[0] == 1.0   # Ring 0
        assert texture[1] == -1.0  # Ring 1
        assert texture[2] == 1.0   # Ring 2
    
    def test_ridge_mode(self):
        """Test ridge at ring boundaries."""
        cmap = LogRings(log_spacing=np.log(2))
        # Points near powers of 2
        z = np.array([1.9, 2.1, 1.5])
        
        texture = _log_rings_texture(z, cmap, 'ridges')
        
        # First two should be near boundary at r=2
        assert texture[0] > 0
        assert texture[1] > 0
        assert texture[2] == 0  # Not near boundary


class TestPhaseTexture:
    """Test direct texture computation for Phase colormap."""
    
    def test_phase_sectors(self):
        """Test ridge detection at phase sectors."""
        cmap = Phase(n_phi=4)  # 4 sectors
        # Sector boundaries are at 0, π/2, π, 3π/2
        # Test points near each boundary
        angles = np.array([0.05, np.pi/2 - 0.05, np.pi + 0.05, 3*np.pi/2 - 0.05])
        z = np.exp(1j * angles)
        
        texture = _phase_texture(z, cmap, 'ridges')
        
        # All points should be near boundaries
        assert np.sum(texture > 0) >= 3
    
    def test_modulus_rings(self):
        """Test ridge detection at modulus rings."""
        cmap = Phase(r_linear_step=1.0)
        # Points at different radii
        z = np.array([0.95, 1.05, 1.5, 2.05])
        
        texture = _phase_texture(z, cmap, 'ridges')
        
        # First, second, and fourth should be near boundaries
        assert texture[0] > 0  # Near r=1
        assert texture[1] > 0  # Near r=1
        assert texture[3] > 0  # Near r=2
    
    def test_combined_pattern(self):
        """Test combined phase and modulus pattern."""
        cmap = Phase(n_phi=6, r_linear_step=1.0)
        # Various points
        z = np.array([1+0j, 0+1j, 1+1j])
        
        texture = _phase_texture(z, cmap, 'ridges')
        
        # Should detect some boundaries
        assert np.any(texture > 0)
    
    def test_binary_mode(self):
        """Test binary mode with enhanced phase."""
        cmap = Phase(n_phi=4, r_linear_step=1.0)
        z = np.array([0.5+0j, 1.5+0j, 0.5*np.exp(1j*np.pi/2)])
        
        texture = _phase_texture(z, cmap, 'binary')
        
        # Should create alternating pattern
        assert np.all(np.abs(texture) == 1.0)


class TestComputeTextureDirect:
    """Test main compute_texture_direct function."""
    
    def test_dispatch_to_correct_function(self):
        """Test that correct texture function is called."""
        z = np.array([1+1j])
        
        # Test each colormap type
        texture = compute_texture_direct(z, Chessboard(), 'binary')
        assert texture.shape == (1,)
        
        texture = compute_texture_direct(z, PolarChessboard(), 'ridges')
        assert texture.shape == (1,)
        
        texture = compute_texture_direct(z, LogRings(), 'grooves')
        assert texture.shape == (1,)
        
        texture = compute_texture_direct(z, Phase(n_phi=6), 'ridges')
        assert texture.shape == (1,)
    
    def test_unknown_colormap(self):
        """Test fallback for unknown colormap."""
        from complexplorer.core.colormap import Colormap
        
        class UnknownColormap(Colormap):
            def hsv_tuple(self, z):
                return None, None, None
        
        z = np.array([1+1j])
        texture = compute_texture_direct(z, UnknownColormap(), 'ridges')
        
        # Should return zeros
        assert np.all(texture == 0)