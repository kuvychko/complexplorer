"""Tests for export base classes."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from complexplorer.export.base import (
    BaseExporter, ImageExporter, VectorExporter, 
    MeshExporter, InteractiveExporter, ExportConfig
)
from complexplorer.utils.validation import ValidationError


class ConcreteExporter(BaseExporter):
    """Concrete implementation for testing."""
    
    def export(self, filename, **kwargs):
        """Dummy export method."""
        return Path(filename)


class TestBaseExporter:
    """Test BaseExporter abstract class."""
    
    def test_cannot_instantiate_abstract(self):
        """Test that abstract class cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseExporter()
    
    def test_init(self):
        """Test initialization."""
        data = {"test": "data"}
        exporter = ConcreteExporter(data)
        
        assert exporter.data is data
        assert exporter._metadata == {}
    
    def test_validate_filename_valid(self):
        """Test filename validation with valid extension."""
        exporter = ConcreteExporter()
        
        # Single extension
        path = exporter.validate_filename("test.stl", "stl")
        assert isinstance(path, Path)
        assert path.name == "test.stl"
        
        # Multiple extensions
        path = exporter.validate_filename("test.png", ["png", "jpg"])
        assert path.name == "test.png"
    
    def test_validate_filename_invalid(self):
        """Test filename validation with invalid extension."""
        exporter = ConcreteExporter()
        
        with pytest.raises(ValidationError):
            exporter.validate_filename("test.txt", "stl")
    
    def test_validate_filename_creates_directory(self, tmp_path):
        """Test that parent directory is created if needed."""
        exporter = ConcreteExporter()
        
        new_dir = tmp_path / "new_directory"
        filepath = new_dir / "test.stl"
        
        result = exporter.validate_filename(str(filepath), "stl")
        
        assert new_dir.exists()
        assert result == filepath
    
    def test_metadata_operations(self):
        """Test metadata get/set operations."""
        exporter = ConcreteExporter()
        
        # Set metadata
        exporter.set_metadata(author="test", version="1.0")
        
        # Get metadata
        metadata = exporter.get_metadata()
        assert metadata["author"] == "test"
        assert metadata["version"] == "1.0"
        
        # Ensure it's a copy
        metadata["modified"] = True
        assert "modified" not in exporter._metadata


class TestImageExporter:
    """Test ImageExporter class."""
    
    def test_init(self):
        """Test initialization."""
        fig = Mock()
        exporter = ImageExporter(fig)
        
        assert exporter.figure is fig
        assert exporter.data is fig
    
    def test_valid_extensions(self):
        """Test valid extensions list."""
        assert 'png' in ImageExporter.VALID_EXTENSIONS
        assert 'jpg' in ImageExporter.VALID_EXTENSIONS
        assert 'jpeg' in ImageExporter.VALID_EXTENSIONS
    
    def test_export_no_figure(self):
        """Test export without figure."""
        exporter = ImageExporter()
        
        with pytest.raises(ValidationError, match="No figure to export"):
            exporter.export("test.png")
    
    def test_export_success(self, tmp_path):
        """Test successful export."""
        mock_fig = Mock()
        exporter = ImageExporter(mock_fig)
        
        filepath = tmp_path / "test.png"
        result = exporter.export(str(filepath), dpi=150, transparent=True)
        
        assert result == filepath
        mock_fig.savefig.assert_called_once_with(
            filepath,
            dpi=150,
            bbox_inches='tight',
            transparent=True
        )
    
    def test_export_invalid_extension(self):
        """Test export with invalid extension."""
        exporter = ImageExporter(Mock())
        
        with pytest.raises(ValidationError):
            exporter.export("test.xyz")


class TestVectorExporter:
    """Test VectorExporter class."""
    
    def test_valid_extensions(self):
        """Test valid extensions list."""
        assert 'svg' in VectorExporter.VALID_EXTENSIONS
        assert 'pdf' in VectorExporter.VALID_EXTENSIONS
        assert 'eps' in VectorExporter.VALID_EXTENSIONS
    
    def test_export_success(self, tmp_path):
        """Test successful export."""
        mock_fig = Mock()
        exporter = VectorExporter(mock_fig)
        
        filepath = tmp_path / "test.pdf"
        result = exporter.export(str(filepath))
        
        assert result == filepath
        mock_fig.savefig.assert_called_once_with(
            filepath,
            bbox_inches='tight'
        )


class ConcreteMeshExporter(MeshExporter):
    """Concrete implementation for testing."""
    
    def export(self, filename, **kwargs):
        """Dummy export method."""
        return Path(filename)


class TestMeshExporter:
    """Test MeshExporter class."""
    
    def test_init(self):
        """Test initialization."""
        mesh = Mock()
        exporter = ConcreteMeshExporter(mesh)
        
        assert exporter.mesh_data is mesh
        assert exporter.data is mesh
    
    def test_validate_mesh_none(self):
        """Test mesh validation with no data."""
        exporter = ConcreteMeshExporter()
        
        with pytest.raises(ValidationError, match="No mesh data"):
            exporter.validate_mesh()
    
    def test_validate_mesh_valid(self):
        """Test mesh validation with data."""
        exporter = ConcreteMeshExporter("mesh_data")
        exporter.validate_mesh()  # Should not raise
    
    def test_get_mesh_statistics(self):
        """Test mesh statistics."""
        exporter = ConcreteMeshExporter("mesh_data")
        stats = exporter.get_mesh_statistics()
        
        assert isinstance(stats, dict)
        assert 'vertices' in stats
        assert 'faces' in stats
        assert 'edges' in stats


class ConcreteInteractiveExporter(InteractiveExporter):
    """Concrete implementation for testing."""
    
    def export(self, filename, **kwargs):
        """Dummy export method."""
        return Path(filename)


class TestInteractiveExporter:
    """Test InteractiveExporter class."""
    
    def test_init(self):
        """Test initialization."""
        plotter = Mock()
        exporter = ConcreteInteractiveExporter(plotter)
        
        assert exporter.plotter is plotter
        assert exporter.data is plotter
    
    def test_valid_extensions(self):
        """Test valid extensions."""
        assert InteractiveExporter.VALID_EXTENSIONS == ['html']
    
    def test_validate_dependencies(self):
        """Test dependency validation."""
        exporter = ConcreteInteractiveExporter()
        exporter.validate_dependencies()  # Should not raise by default


class TestExportConfig:
    """Test ExportConfig class."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = ExportConfig()
        
        assert config.overwrite is True
        assert config.create_dirs is True
        assert config.verbose is False
        assert config.dpi == 100
        assert config.transparent is False
        assert config.binary is True
        assert config.include_metadata is True
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = ExportConfig(
            dpi=300,
            transparent=True,
            custom_param="value"
        )
        
        assert config.dpi == 300
        assert config.transparent is True
        assert config.get('custom_param') == "value"
    
    def test_get_method(self):
        """Test get method."""
        config = ExportConfig(extra="value")
        
        # Get existing attribute
        assert config.get('dpi') == 100
        
        # Get extra parameter
        assert config.get('extra') == "value"
        
        # Get with default
        assert config.get('nonexistent', 'default') == 'default'
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = ExportConfig(
            dpi=200,
            extra_param="extra"
        )
        
        result = config.to_dict()
        
        assert isinstance(result, dict)
        assert result['dpi'] == 200
        assert result['overwrite'] is True
        assert result['extra_param'] == "extra"
        assert '_extra' not in result  # Private attributes excluded