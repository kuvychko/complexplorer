"""
pytest configuration and shared fixtures for complexplorer tests.
"""

import pytest
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to the path so we can import complexplorer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure matplotlib for testing
matplotlib.use('Agg')  # Use non-interactive backend

# Set random seed for reproducibility
np.random.seed(42)


@pytest.fixture
def simple_domain():
    """Provide a simple rectangular domain for testing."""
    from complexplorer import Rectangle
    return Rectangle(re_length=4, im_length=4, center=0)


@pytest.fixture
def test_functions():
    """Common test functions used across multiple tests."""
    return {
        'identity': lambda z: z,
        'quadratic': lambda z: z**2,
        'rational': lambda z: (z - 1) / (z**2 + z + 1),
        'exponential': lambda z: np.exp(z),
        'sine': lambda z: np.sin(z),
        'reciprocal': lambda z: 1/z if z != 0 else np.inf,
        'mobius': lambda z: (z - 1j) / (z + 1j),
        'constant': lambda z: 1 + 2j,
    }


@pytest.fixture
def complex_points():
    """Standard complex test points."""
    return {
        'origin': 0 + 0j,
        'real_positive': 1 + 0j,
        'real_negative': -1 + 0j,
        'imag_positive': 0 + 1j,
        'imag_negative': 0 - 1j,
        'first_quadrant': 1 + 1j,
        'second_quadrant': -1 + 1j,
        'third_quadrant': -1 - 1j,
        'fourth_quadrant': 1 - 1j,
        'unit_circle': np.exp(2j * np.pi * np.linspace(0, 1, 8, endpoint=False)),
    }


@pytest.fixture
def tolerance():
    """Numerical tolerance for floating point comparisons."""
    return {
        'absolute': 1e-10,
        'relative': 1e-10,
        'image': 10.0,  # RMS tolerance for image comparisons
    }


@pytest.fixture(autouse=True)
def cleanup_plots():
    """Automatically close all matplotlib figures after each test."""
    yield
    plt.close('all')


@pytest.fixture
def figure_comparison(request):
    """Helper for visual regression testing."""
    import matplotlib.testing.compare as compare
    
    def _compare(fig, baseline_name, tolerance=10):
        test_image = f"test_{baseline_name}.png"
        baseline_image = os.path.join(
            os.path.dirname(__file__), 
            'fixtures', 
            'reference_images', 
            f"{baseline_name}.png"
        )
        
        # Save test figure
        fig.savefig(test_image, dpi=100)
        plt.close(fig)
        
        if not os.path.exists(baseline_image):
            # If baseline doesn't exist, save current as baseline
            import shutil
            os.makedirs(os.path.dirname(baseline_image), exist_ok=True)
            shutil.copy(test_image, baseline_image)
            pytest.skip(f"Generated baseline image: {baseline_image}")
        
        # Compare images
        result = compare.compare_images(baseline_image, test_image, tolerance)
        
        # Clean up test image
        if os.path.exists(test_image) and result is None:
            os.remove(test_image)
        
        assert result is None, f"Image comparison failed: {result}"
    
    return _compare


class Helpers:
    """Collection of test helper functions."""
    
    @staticmethod
    def assert_complex_equal(z1, z2, tolerance=1e-10):
        """Assert two complex numbers are equal within tolerance."""
        assert np.abs(z1 - z2) < tolerance, f"{z1} != {z2}"
    
    @staticmethod
    def assert_array_complex_equal(arr1, arr2, tolerance=1e-10):
        """Assert two complex arrays are equal within tolerance."""
        np.testing.assert_allclose(arr1, arr2, rtol=tolerance, atol=tolerance)
    
    @staticmethod
    def assert_in_range(value, min_val, max_val, inclusive=True):
        """Assert value is in range."""
        if inclusive:
            assert min_val <= value <= max_val, f"{value} not in [{min_val}, {max_val}]"
        else:
            assert min_val < value < max_val, f"{value} not in ({min_val}, {max_val})"
    
    @staticmethod
    def assert_colors_valid(colors):
        """Assert color array has valid RGB values."""
        assert colors.ndim == 3, "Colors should be 3D array"
        assert colors.shape[2] == 3, "Colors should have 3 channels (RGB)"
        assert np.all(colors >= 0) and np.all(colors <= 1), "Colors should be in [0, 1]"


@pytest.fixture
def helpers():
    """Provide test helper functions."""
    return Helpers