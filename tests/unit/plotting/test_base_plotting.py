"""Tests for plotting base classes."""

import numpy as np
import pytest
from unittest.mock import Mock, MagicMock, patch
from complexplorer.plotting.base import (
    BasePlotter, Base2DPlotter, Base3DPlotter, PlotConfig
)
from complexplorer.utils.validation import ValidationError


class ConcretePlotter(BasePlotter):
    """Concrete implementation for testing."""
    
    def plot(self, **kwargs):
        """Dummy plot method."""
        return "plotted"


class Concrete2DPlotter(Base2DPlotter):
    """Concrete 2D implementation for testing."""
    
    def plot(self, **kwargs):
        """Dummy plot method."""
        return "2d_plotted"


class Concrete3DPlotter(Base3DPlotter):
    """Concrete 3D implementation for testing."""
    
    def plot(self, **kwargs):
        """Dummy plot method."""
        self._mesh_data = "mesh_data"
        return "3d_plotted"


class TestBasePlotter:
    """Test BasePlotter abstract class."""
    
    def test_cannot_instantiate_abstract(self):
        """Test that abstract class cannot be instantiated."""
        with pytest.raises(TypeError):
            BasePlotter()
    
    def test_init_with_domain_and_func(self):
        """Test initialization with domain and function."""
        domain = Mock()
        domain.mesh = Mock(return_value=np.array([1+2j, 3+4j]))
        func = lambda z: z**2
        cmap = Mock()
        cmap.hsv = Mock()
        cmap.rgb = Mock(return_value=np.array([[1, 0, 0], [0, 1, 0]]))
        
        plotter = ConcretePlotter(domain=domain, func=func, cmap=cmap)
        
        assert plotter.domain is domain
        assert plotter.func is not None  # Wrapped function
        assert plotter.cmap is cmap
        assert plotter._data is None
    
    def test_init_with_mesh_and_values(self):
        """Test initialization with pre-computed values."""
        z = np.array([1+2j, 3+4j])
        f = np.array([2+3j, 4+5j])
        cmap = Mock()
        cmap.hsv = Mock()
        cmap.rgb = Mock(return_value=np.array([[1, 0, 0], [0, 1, 0]]))
        
        plotter = ConcretePlotter(z=z, f=f, cmap=cmap)
        
        assert np.array_equal(plotter.z, z)
        assert np.array_equal(plotter.f, f)
        assert plotter.domain is None
        assert plotter.func is None
    
    def test_init_validation_errors(self):
        """Test validation errors during initialization."""
        # No domain or mesh
        with pytest.raises(ValidationError):
            ConcretePlotter()
        
        # No function or values
        domain = Mock()
        with pytest.raises(ValidationError):
            ConcretePlotter(domain=domain)
    
    def test_prepare_data_with_domain_func(self):
        """Test data preparation with domain and function."""
        # Setup domain
        domain = Mock()
        z_mesh = np.array([[1+1j, 2+1j], [1+2j, 2+2j]])
        domain.mesh = Mock(return_value=z_mesh)
        
        # Setup function
        func = lambda z: z**2
        
        # Setup colormap
        cmap = Mock()
        cmap.rgb = Mock(return_value=np.ones((4, 3)))  # 4 points, RGB
        
        plotter = ConcretePlotter(domain=domain, func=func, cmap=cmap)
        data = plotter._prepare_data()
        
        assert 'z' in data
        assert 'f' in data
        assert 'x' in data
        assert 'y' in data
        assert 'u' in data
        assert 'v' in data
        assert 'colors' in data
        
        # Check caching
        data2 = plotter._prepare_data()
        assert data is data2
    
    def test_prepare_data_with_precomputed(self):
        """Test data preparation with pre-computed values."""
        z = np.array([1+1j, 2+2j])
        f = np.array([2+3j, 4+5j])
        
        cmap = Mock()
        cmap.rgb = Mock(return_value=np.ones((2, 3)))
        
        plotter = ConcretePlotter(z=z, f=f, cmap=cmap)
        data = plotter._prepare_data()
        
        assert np.array_equal(data['z'], z)
        assert np.array_equal(data['f'], f)
        assert data['colors'].shape == (2, 3)
    
    def test_get_plot_limits(self):
        """Test plot limit calculation."""
        data = {
            'x': np.array([1, 2, 3]),
            'y': np.array([4, 5, 6]),
            'u': np.array([0, 1, np.inf]),
            'v': np.array([-1, 0, 1])
        }
        
        plotter = ConcretePlotter(z=[1], f=[1])  # Dummy init
        limits = plotter._get_plot_limits(data)
        
        # Check x limits
        assert limits['x'][0] < 1
        assert limits['x'][1] > 3
        
        # Check that infinities are handled
        assert np.isfinite(limits['u'][0])
        assert np.isfinite(limits['u'][1])
    
    def test_handle_infinities(self):
        """Test infinity handling."""
        values = np.array([1, 2, np.inf, -np.inf, np.nan, 3])
        
        plotter = ConcretePlotter(z=[1], f=[1])  # Dummy init
        result = plotter._handle_infinities(values, replacement=0)
        
        expected = np.array([1, 2, 0, 0, 0, 3])
        np.testing.assert_array_equal(result, expected)


class TestBase2DPlotter:
    """Test Base2DPlotter class."""
    
    def test_prepare_axes_new(self):
        """Test axes preparation with new figure."""
        plotter = Concrete2DPlotter(z=[1], f=[1])
        
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig = Mock()
            mock_ax = Mock()
            mock_ax.figure = mock_fig
            mock_subplots.return_value = (mock_fig, mock_ax)
            
            fig, ax = plotter._prepare_axes()
            
            assert fig is mock_fig
            assert ax is mock_ax
            mock_subplots.assert_called_once_with(figsize=(8, 8))
    
    def test_prepare_axes_existing(self):
        """Test axes preparation with existing axes."""
        plotter = Concrete2DPlotter(z=[1], f=[1])
        
        mock_fig = Mock()
        mock_ax = Mock()
        mock_ax.figure = mock_fig
        
        fig, ax = plotter._prepare_axes(ax=mock_ax)
        
        assert fig is mock_fig
        assert ax is mock_ax
    
    def test_apply_2d_styling(self):
        """Test 2D styling application."""
        plotter = Concrete2DPlotter(z=[1], f=[1])
        
        mock_ax = Mock()
        plotter._apply_2d_styling(mock_ax, title="Test", xlabel="X", ylabel="Y")
        
        mock_ax.set_aspect.assert_called_once_with('equal')
        mock_ax.grid.assert_called_once_with(True, alpha=0.3)
        mock_ax.set_title.assert_called_once_with("Test")
        mock_ax.set_xlabel.assert_called_once_with("X")
        mock_ax.set_ylabel.assert_called_once_with("Y")


class TestBase3DPlotter:
    """Test Base3DPlotter class."""
    
    def test_init(self):
        """Test 3D plotter initialization."""
        plotter = Concrete3DPlotter(z=[1], f=[1])
        assert plotter._mesh_data is None
    
    def test_get_mesh_data_before_plot(self):
        """Test getting mesh data before plotting."""
        plotter = Concrete3DPlotter(z=[1], f=[1])
        
        with pytest.raises(ValidationError):
            plotter.get_mesh_data()
    
    def test_get_mesh_data_after_plot(self):
        """Test getting mesh data after plotting."""
        plotter = Concrete3DPlotter(z=[1], f=[1])
        plotter.plot()  # Sets _mesh_data
        
        mesh_data = plotter.get_mesh_data()
        assert mesh_data == "mesh_data"
    
    def test_prepare_3d_axes(self):
        """Test 3D axes preparation."""
        plotter = Concrete3DPlotter(z=[1], f=[1])
        
        with patch('matplotlib.pyplot.figure') as mock_figure:
            mock_fig = Mock()
            mock_ax = Mock()
            mock_fig.add_subplot = Mock(return_value=mock_ax)
            mock_ax.figure = mock_fig
            mock_figure.return_value = mock_fig
            
            fig, ax = plotter._prepare_3d_axes()
            
            assert fig is mock_fig
            assert ax is mock_ax
            mock_fig.add_subplot.assert_called_once_with(111, projection='3d')
    
    def test_apply_3d_styling(self):
        """Test 3D styling application."""
        plotter = Concrete3DPlotter(z=[1], f=[1])
        
        mock_ax = Mock()
        plotter._apply_3d_styling(
            mock_ax, 
            title="3D Test",
            xlabel="X", 
            ylabel="Y", 
            zlabel="Z",
            elev=45,
            azim=60
        )
        
        mock_ax.set_title.assert_called_once_with("3D Test")
        mock_ax.set_xlabel.assert_called_once_with("X")
        mock_ax.set_ylabel.assert_called_once_with("Y")
        mock_ax.set_zlabel.assert_called_once_with("Z")
        mock_ax.view_init.assert_called_once_with(elev=45, azim=60)


class TestPlotConfig:
    """Test PlotConfig class."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = PlotConfig()
        
        assert config.figsize == (8, 8)
        assert config.title is None
        assert config.colorbar is True
        assert config.grid is True
        assert config.dpi == 100
        assert config.interactive is True
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = PlotConfig(
            figsize=(10, 10),
            title="Custom",
            colorbar=False,
            custom_param="value"
        )
        
        assert config.figsize == (10, 10)
        assert config.title == "Custom"
        assert config.colorbar is False
        assert config.get('custom_param') == "value"
    
    def test_get_method(self):
        """Test get method."""
        config = PlotConfig(extra="value")
        
        # Get existing attribute
        assert config.get('figsize') == (8, 8)
        
        # Get extra parameter
        assert config.get('extra') == "value"
        
        # Get with default
        assert config.get('nonexistent', 'default') == 'default'
    
    def test_update_method(self):
        """Test update method."""
        config = PlotConfig()
        
        config.update(
            figsize=(12, 12),
            title="Updated",
            new_param="new"
        )
        
        assert config.figsize == (12, 12)
        assert config.title == "Updated"
        assert config.get('new_param') == "new"