# Complexplorer Test Suite

This directory contains the comprehensive test suite for the complexplorer library.

## Structure

```
tests/
├── unit/                    # Unit tests for individual components
│   ├── test_domain.py      # Tests for domain classes
│   ├── test_cmap.py        # Tests for color map classes
│   ├── test_funcs.py       # Tests for utility functions
│   ├── test_plots_2d.py    # Tests for 2D plotting functions
│   └── test_plots_3d.py    # Tests for 3D plotting functions
├── fixtures/               # Test fixtures and utilities
│   ├── test_functions.py   # Common complex functions for testing
│   └── reference_images/   # Reference images for visual regression tests
├── conftest.py            # pytest configuration and shared fixtures
├── test_requirements.txt  # Testing dependencies
└── README.md             # This file
```

## Running Tests

### Install test dependencies

```bash
pip install -r tests/test_requirements.txt
```

### Run all tests

```bash
pytest tests/
```

### Run with coverage

```bash
pytest tests/ --cov=complexplorer --cov-report=html
```

### Run specific test file

```bash
pytest tests/unit/test_domain.py
```

### Run specific test class or function

```bash
pytest tests/unit/test_domain.py::TestRectangle
pytest tests/unit/test_domain.py::TestRectangle::test_rectangle_creation
```

### Run with verbose output

```bash
pytest tests/ -v
```

## Test Categories

### Domain Tests (`test_domain.py`)
- Domain base class functionality
- Rectangle, Disk, and Annulus domains
- Domain operations (union, intersection)
- Mesh generation and masking
- Edge cases and error handling

### Color Map Tests (`test_cmap.py`)
- Phase coloring (basic and enhanced)
- Chessboard patterns
- Polar chessboard patterns
- Logarithmic rings
- HSV/RGB conversions
- Special value handling (NaN, infinity)

### Function Tests (`test_funcs.py`)
- Phase calculation and range
- Sawtooth function
- Stereographic projection
- Array handling
- Edge cases

### 2D Plot Tests (`test_plots_2d.py`)
- Basic plotting functionality
- Pair plots
- Riemann charts and hemispheres
- Different color maps
- File saving
- Error handling

### 3D Plot Tests (`test_plots_3d.py`)
- Analytic landscapes
- Pair landscapes
- Riemann sphere visualization
- Logarithmic z-axis
- Performance with large meshes
- File saving

## Writing New Tests

When adding new functionality to complexplorer, please add corresponding tests:

1. Create test functions following the naming convention `test_<functionality>`
2. Use descriptive names that explain what is being tested
3. Include docstrings explaining the test purpose
4. Test both normal cases and edge cases
5. Use parametrized tests for testing multiple similar cases
6. Add visual regression tests for new plot types

## Visual Regression Testing

For plotting functions, we use visual regression testing:

1. First run generates reference images in `fixtures/reference_images/`
2. Subsequent runs compare against these references
3. If plots change intentionally, delete the reference images to regenerate them

## Coverage Goals

We aim for >90% code coverage. Areas that are particularly important:
- Mathematical correctness of domain operations
- Color map accuracy
- Proper handling of complex function singularities
- Memory efficiency for large meshes