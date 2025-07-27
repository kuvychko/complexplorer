"""Tests for STL ornament generator."""

import numpy as np
import pytest
import os

# Check if PyVista is available
try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

from complexplorer.export.stl import OrnamentGenerator, create_ornament
from complexplorer.core.colormap import Phase, Chessboard, PolarChessboard
from complexplorer.core.domain import Disk
from complexplorer.utils.validation import ValidationError


@pytest.mark.skipif(not HAS_PYVISTA, reason="PyVista not installed")
class TestOrnamentGenerator:
    """Test OrnamentGenerator class."""
    
    def test_init_defaults(self):
        """Test initialization with defaults."""
        func = lambda z: z**2
        gen = OrnamentGenerator(func)
        
        assert gen.func is func
        assert gen.resolution == 150
        assert gen.scaling == 'arctan'
        assert isinstance(gen.cmap, Phase)
        assert gen.sphere_mesh is None
        assert gen.texture_height == 0.0
        assert gen.texture_mode == 'ridges'
        assert gen.texture_sharpness == 1.0
    
    def test_init_custom(self):
        """Test initialization with custom parameters."""
        func = lambda z: z**3 - 1
        cmap = Phase(n_phi=8)
        domain = Disk(2)
        
        gen = OrnamentGenerator(
            func, 
            resolution=100,
            scaling='logarithmic',
            scaling_params={'base': 2.0},
            cmap=cmap,
            domain=domain,
            texture_height=0.005,
            texture_mode='binary',
            texture_sharpness=0.8
        )
        
        assert gen.resolution == 100
        assert gen.scaling == 'logarithmic'
        assert gen.scaling_params['base'] == 2.0
        assert gen.cmap is cmap
        assert gen.domain is domain
        assert gen.texture_height == 0.005
        assert gen.texture_mode == 'binary'
        assert gen.texture_sharpness == 0.8
    
    def test_default_scaling_params(self):
        """Test default parameters for different scaling methods."""
        func = lambda z: z
        
        # Test each scaling method
        scalings = ['constant', 'arctan', 'logarithmic', 'linear_clamp',
                   'linear', 'power', 'sigmoid', 'adaptive', 'hybrid']
        
        for scaling in scalings:
            gen = OrnamentGenerator(func, scaling=scaling)
            params = gen.scaling_params
            assert isinstance(params, dict)
            assert len(params) > 0
    
    def test_generate_ornament_basic(self):
        """Test basic ornament generation."""
        func = lambda z: z**2
        gen = OrnamentGenerator(func, resolution=50)
        
        sphere = gen.generate_ornament()
        
        assert isinstance(sphere, pv.PolyData)
        assert sphere.n_points > 0
        assert sphere.n_cells > 0
        assert "RGB" in sphere.array_names
        assert "magnitude" in sphere.array_names
        assert "phase" in sphere.array_names
        assert "radius" in sphere.array_names
        assert gen.sphere_mesh is sphere
    
    def test_generate_ornament_with_domain(self):
        """Test ornament generation with domain constraint."""
        func = lambda z: 1 / z
        domain = Disk(radius=2.0)
        gen = OrnamentGenerator(func, resolution=40, domain=domain)
        
        sphere = gen.generate_ornament()
        
        assert isinstance(sphere, pv.PolyData)
        # Should have fewer points due to domain filtering
        gen_full = OrnamentGenerator(func, resolution=40)
        sphere_full = gen_full.generate_ornament()
        assert sphere.n_points < sphere_full.n_points
    
    def test_different_scaling_methods(self):
        """Test different scaling methods produce different results."""
        func = lambda z: (z - 1) / (z + 1)
        
        radii_by_method = {}
        
        for scaling in ['constant', 'arctan', 'linear', 'logarithmic']:
            gen = OrnamentGenerator(func, resolution=30, scaling=scaling)
            sphere = gen.generate_ornament()
            
            # Get actual radii
            radii = np.linalg.norm(sphere.points, axis=1)
            radii_by_method[scaling] = radii
        
        # Constant should have all same radius
        assert np.allclose(radii_by_method['constant'], 
                          radii_by_method['constant'][0])
        
        # Others should vary
        for method in ['arctan', 'linear', 'logarithmic']:
            assert radii_by_method[method].std() > 0.01
    
    def test_validate_mesh(self):
        """Test mesh validation."""
        func = lambda z: z**3 - z
        gen = OrnamentGenerator(func, resolution=40)
        
        # Generate ornament first
        gen.generate_ornament()
        
        results = gen.validate_mesh(size_mm=50, verbose=False)
        
        assert isinstance(results, dict)
        assert 'is_watertight' in results
        assert 'is_manifold' in results
        assert 'dimensions' in results
        assert 'volume' in results
        assert 'surface_area' in results
    
    def test_save_stl(self, tmp_path):
        """Test STL file saving."""
        func = lambda z: z**2 + 1
        gen = OrnamentGenerator(func, resolution=30)
        gen.generate_ornament()
        
        filename = tmp_path / "test_ornament.stl"
        saved_path = gen.save_stl(str(filename), size_mm=50, repair=True, verbose=False)
        
        assert os.path.exists(saved_path)
        assert saved_path == str(filename)
        
        # Check file is not empty
        assert os.path.getsize(saved_path) > 0
        
        # Load and verify
        loaded = pv.read(saved_path)
        assert isinstance(loaded, pv.PolyData)
        assert loaded.n_points > 0
        assert loaded.n_cells > 0
    
    def test_generate_and_save_pipeline(self, tmp_path):
        """Test complete pipeline."""
        func = lambda z: (z**2 - 1) / (z**2 + 1)
        gen = OrnamentGenerator(func, resolution=40, scaling='sigmoid')
        
        filename = tmp_path / "complete_test.stl"
        saved_path = gen.generate_and_save(
            str(filename),
            size_mm=60,
            repair=True,
            validate=True,
            verbose=False
        )
        
        assert os.path.exists(saved_path)
        
        # Verify the mesh
        mesh = pv.read(saved_path)
        bounds = mesh.bounds
        max_dim = max(bounds[1] - bounds[0], 
                     bounds[3] - bounds[2],
                     bounds[5] - bounds[4])
        
        # Should be approximately 60mm
        assert 55 < max_dim < 65
    
    def test_invalid_scaling_method(self):
        """Test error for invalid scaling method."""
        func = lambda z: z
        gen = OrnamentGenerator(func, scaling='invalid_method')
        
        with pytest.raises(ValidationError, match="Unknown scaling mode"):
            gen.generate_ornament()
    
    def test_handle_infinities(self):
        """Test handling of infinite function values."""
        func = lambda z: 1 / z  # Has pole at origin
        gen = OrnamentGenerator(func, resolution=30)
        
        sphere = gen.generate_ornament()
        
        # Should not have any infinite radii
        radii = np.linalg.norm(sphere.points, axis=1)
        assert np.all(np.isfinite(radii))
    
    def test_no_generated_mesh_error(self):
        """Test error when trying to save without generating."""
        func = lambda z: z
        gen = OrnamentGenerator(func)
        
        with pytest.raises(ValueError, match="No mesh generated"):
            gen.save_stl("test.stl")


@pytest.mark.skipif(not HAS_PYVISTA, reason="PyVista not installed")
class TestCreateOrnament:
    """Test create_ornament convenience function."""
    
    def test_basic_creation(self, tmp_path):
        """Test basic ornament creation."""
        func = lambda z: z**3 - 1
        filename = tmp_path / "basic.stl"
        
        saved_path = create_ornament(
            func, 
            str(filename),
            size_mm=40,
            resolution=50,
            verbose=False
        )
        
        assert os.path.exists(saved_path)
        assert saved_path == str(filename)
    
    def test_with_custom_params(self, tmp_path):
        """Test creation with custom parameters."""
        func = lambda z: np.sin(z)
        filename = tmp_path / "custom.stl"
        cmap = Phase(n_phi=16)
        
        saved_path = create_ornament(
            func,
            str(filename),
            size_mm=55,
            resolution=80,
            scaling='adaptive',
            scaling_params={'low_percentile': 5, 'high_percentile': 95},
            cmap=cmap,
            verbose=False
        )
        
        assert os.path.exists(saved_path)
        
        # Verify mesh properties
        mesh = pv.read(saved_path)
        assert mesh.n_points > 1000  # Higher resolution


@pytest.mark.skipif(not HAS_PYVISTA, reason="PyVista not installed")
class TestComplexFunctions:
    """Test with various complex functions."""
    
    def test_polynomial(self, tmp_path):
        """Test polynomial function."""
        func = lambda z: z**4 - 2*z**2 + 1
        filename = tmp_path / "polynomial.stl"
        
        gen = OrnamentGenerator(func, resolution=40)
        gen.generate_and_save(str(filename), size_mm=45, verbose=False)
        
        assert os.path.exists(filename)
    
    def test_rational_function(self, tmp_path):
        """Test rational function with poles."""
        func = lambda z: (z**2 + 1) / (z**2 - 1)
        filename = tmp_path / "rational.stl"
        
        gen = OrnamentGenerator(func, resolution=40, scaling='arctan')
        gen.generate_and_save(str(filename), size_mm=50, verbose=False)
        
        assert os.path.exists(filename)
    
    def test_transcendental(self, tmp_path):
        """Test transcendental function."""
        func = lambda z: np.exp(z) / (z + 1)
        filename = tmp_path / "transcendental.stl"
        
        gen = OrnamentGenerator(func, resolution=40, scaling='logarithmic')
        gen.generate_and_save(str(filename), size_mm=55, verbose=False)
        
        assert os.path.exists(filename)
    
    def test_essential_singularity(self, tmp_path):
        """Test function with essential singularity."""
        def func(z):
            # exp(1/z) with safety
            with np.errstate(divide='ignore', invalid='ignore'):
                result = np.exp(1/z)
            return np.where(np.isfinite(result), result, 0)
        
        filename = tmp_path / "essential.stl"
        
        gen = OrnamentGenerator(func, resolution=40, scaling='adaptive')
        gen.generate_and_save(str(filename), size_mm=50, verbose=False)
        
        assert os.path.exists(filename)


@pytest.mark.skipif(not HAS_PYVISTA, reason="PyVista not installed")
class TestOrnamentTexture:
    """Test texture functionality in ornament generation."""
    
    def test_generate_with_ridges(self):
        """Test ornament generation with ridge texture."""
        func = lambda z: (z**2 - 1) / (z**2 + 1)
        cmap = Phase(n_phi=6, r_linear_step=1.0)
        gen = OrnamentGenerator(
            func, 
            resolution=50,
            cmap=cmap,
            texture_height=0.005,
            texture_mode='ridges',
            texture_sharpness=0.9
        )
        
        sphere = gen.generate_ornament()
        
        # Should have modified positions due to texture
        assert isinstance(sphere, pv.PolyData)
        assert sphere.n_points > 0
    
    def test_generate_with_binary(self):
        """Test ornament generation with binary texture."""
        func = lambda z: z**2
        cmap = Chessboard(spacing=0.5)
        gen = OrnamentGenerator(
            func,
            resolution=40,
            cmap=cmap,
            texture_height=0.008,
            texture_mode='binary'
        )
        
        sphere = gen.generate_ornament()
        
        # Points should be displaced
        assert isinstance(sphere, pv.PolyData)
    
    def test_generate_with_grooves(self):
        """Test ornament generation with groove texture."""
        func = lambda z: z + 1/z
        cmap = PolarChessboard(n_phi=8, spacing=0.3)
        gen = OrnamentGenerator(
            func,
            resolution=40,
            cmap=cmap,
            texture_height=0.006,
            texture_mode='grooves'
        )
        
        sphere = gen.generate_ornament()
        
        assert isinstance(sphere, pv.PolyData)
    
    def test_no_texture_by_default(self):
        """Test that texture is not applied when height is 0."""
        func = lambda z: z**3 - 1
        gen = OrnamentGenerator(func, resolution=30)
        
        sphere = gen.generate_ornament()
        
        # Store original points
        original_radii = np.linalg.norm(sphere.points, axis=1)
        
        # Generate with explicit zero texture
        gen2 = OrnamentGenerator(func, resolution=30, texture_height=0.0)
        sphere2 = gen2.generate_ornament()
        radii2 = np.linalg.norm(sphere2.points, axis=1)
        
        # Should be identical
        assert np.allclose(original_radii, radii2)
    
    def test_texture_affects_points(self):
        """Test that texture actually modifies the mesh."""
        func = lambda z: z**2
        cmap = Chessboard(spacing=0.5)
        
        # Generate without texture
        gen1 = OrnamentGenerator(func, resolution=30, cmap=cmap)
        sphere1 = gen1.generate_ornament()
        radii1 = np.linalg.norm(sphere1.points, axis=1)
        
        # Generate with texture
        gen2 = OrnamentGenerator(
            func, 
            resolution=30, 
            cmap=cmap,
            texture_height=0.01,
            texture_mode='binary'
        )
        sphere2 = gen2.generate_ornament()
        radii2 = np.linalg.norm(sphere2.points, axis=1)
        
        # Radii should be different
        assert not np.allclose(radii1, radii2)
        # Some points should be raised, some lowered
        assert np.any(radii2 > radii1)
        assert np.any(radii2 < radii1)
    
    def test_save_textured_stl(self, tmp_path):
        """Test saving STL with texture."""
        func = lambda z: (z - 1) / (z + 1)
        cmap = Phase(n_phi=12)
        gen = OrnamentGenerator(
            func,
            resolution=40,
            cmap=cmap,
            texture_height=0.007,
            texture_mode='ridges'
        )
        
        gen.generate_ornament()
        filename = tmp_path / "textured.stl"
        saved_path = gen.save_stl(str(filename), size_mm=50, verbose=False)
        
        assert os.path.exists(saved_path)
        
        # Verify the mesh loads correctly
        mesh = pv.read(saved_path)
        assert mesh.n_points > 0
        assert mesh.n_cells > 0
    
    def test_create_ornament_with_texture(self, tmp_path):
        """Test convenience function with texture."""
        func = lambda z: z**3 - z
        filename = tmp_path / "textured_convenience.stl"
        
        saved_path = create_ornament(
            func,
            str(filename),
            size_mm=45,
            resolution=50,
            cmap=Chessboard(spacing=0.4),
            texture_height=0.005,
            texture_mode='binary',
            texture_sharpness=0.95,
            verbose=False
        )
        
        assert os.path.exists(saved_path)