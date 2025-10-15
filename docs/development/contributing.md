# Contributing to Complexplorer

Thank you for your interest in contributing to Complexplorer! This guide will help you get started.

## Code of Conduct

Be respectful, collaborative, and professional. We value contributions from everyone and strive to create a welcoming environment.

## Getting Started

### Prerequisites

- Python 3.11 or higher
- uv package manager (recommended) or pip
- Git

### Development Setup

1. **Fork and clone the repository:**
```bash
git clone https://github.com/yourusername/complexplorer.git
cd complexplorer
```

2. **Create virtual environment and install dependencies:**
```bash
# Using uv (recommended)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

# Or using pip
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

3. **Verify installation:**
```bash
# Run tests
pytest tests/ -v

# Check code quality
ruff check .
mypy complexplorer/
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

**Branch naming conventions:**
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation improvements
- `refactor/` - Code refactoring
- `test/` - Test additions or improvements

### 2. Make Changes

- Write clean, readable code
- Follow existing code style
- Add tests for new functionality
- Update documentation as needed

### 3. Run Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/unit/test_domain.py -v

# Run with coverage
pytest tests/ --cov=complexplorer --cov-report=html

# View coverage report
open htmlcov/index.html  # On macOS
# or
start htmlcov/index.html  # On Windows
```

### 4. Check Code Quality

```bash
# Lint with ruff
ruff check .

# Auto-fix simple issues
ruff check . --fix

# Type checking
mypy complexplorer/

# Format code (if using black)
black complexplorer/ tests/
```

### 5. Update Documentation

If you've added features or changed APIs:

```bash
# Build documentation locally
uv run mkdocs serve

# View at http://127.0.0.1:8000
```

### 6. Commit Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: add new colormap for scientific visualization"
# or
git commit -m "fix: correct stereographic projection at poles"
```

**Commit message format:**
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Test additions/changes
- `refactor:` - Code refactoring
- `style:` - Code style changes (formatting, etc.)
- `chore:` - Maintenance tasks

### 7. Push and Create Pull Request

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create pull request on GitHub
```

## What to Contribute

### Good First Issues

- Add new colormaps
- Improve documentation with examples
- Add tests for existing functionality
- Fix typos or clarify documentation
- Add type hints to untyped functions

### Feature Requests

Before implementing a major feature:

1. Open an issue to discuss the feature
2. Wait for maintainer feedback
3. Implement once approved
4. Submit pull request

### Bug Reports

When reporting bugs, include:

- Python version
- Complexplorer version
- Minimal reproducible example
- Expected vs actual behavior
- Error messages (full traceback)

## Code Guidelines

### Python Style

- Follow PEP 8
- Use type hints for function signatures
- Maximum line length: 100 characters
- Use docstrings for public functions

**Example:**
```python
def stereographic_to_sphere(z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map complex numbers to Riemann sphere via stereographic projection.

    Parameters
    ----------
    z : np.ndarray
        Complex numbers to project.

    Returns
    -------
    x, y, z_coord : np.ndarray
        3D coordinates on unit sphere.

    Notes
    -----
    Uses projection from north pole. The mapping is:

    .. math::
        x = 2 \\text{Re}(z) / (|z|^2 + 1)
        y = 2 \\text{Im}(z) / (|z|^2 + 1)
        z = (|z|^2 - 1) / (|z|^2 + 1)
    """
    r_sq = np.abs(z)**2
    denom = r_sq + 1

    x = 2 * z.real / denom
    y = 2 * z.imag / denom
    z_coord = (r_sq - 1) / denom

    return x, y, z_coord
```

### Docstring Format

Use NumPy-style docstrings:

```python
def function_name(param1: type1, param2: type2 = default) -> return_type:
    """Brief one-line description.

    More detailed description if needed. Can span multiple paragraphs.

    Parameters
    ----------
    param1 : type1
        Description of param1.
    param2 : type2, optional
        Description of param2. Default is {default}.

    Returns
    -------
    return_type
        Description of return value.

    Raises
    ------
    ValueError
        When invalid input is provided.

    Examples
    --------
    >>> result = function_name(value1, value2)
    >>> print(result)
    expected_output

    See Also
    --------
    related_function : Brief description.
    """
```

### Testing Guidelines

- Write tests for all new functionality
- Aim for >80% code coverage
- Use pytest fixtures for common setup
- Test edge cases and error conditions

**Example test:**
```python
import pytest
import numpy as np
from complexplorer.core.domain import Disk

def test_disk_contains_center():
    """Test that disk contains its center."""
    disk = Disk(radius=2, center=1+1j)
    assert disk.contains(1+1j)

def test_disk_contains_boundary():
    """Test points on boundary."""
    disk = Disk(radius=1, center=0)
    # Points exactly on boundary should be included
    assert disk.contains(1+0j)
    assert disk.contains(0+1j)

def test_disk_excludes_exterior():
    """Test that disk excludes exterior points."""
    disk = Disk(radius=1, center=0)
    assert not disk.contains(2+0j)

@pytest.mark.parametrize("center,radius,point,expected", [
    (0+0j, 1, 0+0j, True),
    (0+0j, 1, 2+0j, False),
    (1+1j, 2, 1+1j, True),
    (1+1j, 2, 4+4j, False),
])
def test_disk_contains_parametrized(center, radius, point, expected):
    """Parametrized tests for disk containment."""
    disk = Disk(radius=radius, center=center)
    assert disk.contains(point) == expected
```

### Adding a New Colormap

1. **Create class in `complexplorer/core/colormap.py`:**

```python
class MyNewColormap(Colormap):
    """Brief description of the colormap.

    Longer description explaining the color scheme,
    its purpose, and when to use it.

    Parameters
    ----------
    param1 : type
        Description.
    param2 : type, optional
        Description. Default is value.
    """

    def __init__(self, param1: type, param2: type = default):
        super().__init__()
        self.param1 = param1
        self.param2 = param2

    def hsv(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Map complex values to HSV.

        Parameters
        ----------
        z : np.ndarray
            Complex values.

        Returns
        -------
        h, s, v : np.ndarray
            Hue, saturation, value in [0, 1].
        """
        # Implement color mapping logic
        phase = np.angle(z) / (2 * np.pi) % 1
        modulus = np.abs(z)

        h = phase  # Hue from phase
        s = np.ones_like(h)  # Full saturation
        v = np.ones_like(h)  # Full brightness

        return h, s, v
```

2. **Add to `__init__.py` exports:**

```python
from .colormap import MyNewColormap

__all__ = [..., 'MyNewColormap']
```

3. **Add tests in `tests/unit/core/test_colormap.py`:**

```python
def test_mynewcolormap_basic():
    """Test basic MyNewColormap functionality."""
    cmap = MyNewColormap(param1=value)
    z = np.array([1+0j, 0+1j, -1+0j, 0-1j])
    rgb = cmap(z)

    assert rgb.shape == (4, 3)
    assert np.all(rgb >= 0) and np.all(rgb <= 1)

def test_mynewcolormap_parameters():
    """Test parameter variations."""
    cmap1 = MyNewColormap(param1=value1)
    cmap2 = MyNewColormap(param1=value2)

    z = np.array([1+1j])
    rgb1 = cmap1(z)
    rgb2 = cmap2(z)

    # Should produce different colors
    assert not np.allclose(rgb1, rgb2)
```

4. **Add documentation:**
   - Update `docs/user-guide/colormaps.md`
   - Add example to `docs/examples/gallery.md`

### Adding a New Domain Type

1. **Create class in `complexplorer/core/domain.py`:**

```python
class MyNewDomain(Domain):
    """Brief description of the domain.

    Parameters
    ----------
    param1 : type
        Description.
    param2 : type
        Description.
    """

    def __init__(self, param1: type, param2: type):
        super().__init__()
        self.param1 = param1
        self.param2 = param2

    def contains(self, z: np.ndarray) -> np.ndarray:
        """Check if points are in the domain.

        Parameters
        ----------
        z : np.ndarray
            Complex numbers to test.

        Returns
        -------
        np.ndarray
            Boolean array indicating membership.
        """
        # Implement containment logic
        return condition(z, self.param1, self.param2)

    def __repr__(self) -> str:
        return f"MyNewDomain(param1={self.param1}, param2={self.param2})"
```

2. **Add tests in `tests/unit/core/test_domain.py`**

3. **Update documentation in `docs/user-guide/domains.md`**

## Documentation

### Building Documentation

```bash
# Install documentation dependencies
uv pip install -e ".[docs]"

# Serve documentation locally
uv run mkdocs serve

# Build static site
uv run mkdocs build

# Deploy to GitHub Pages (maintainers only)
uv run mkdocs gh-deploy
```

### Documentation Structure

```
docs/
   getting-started/     # Installation, quickstart
   user-guide/          # Feature documentation
   api/                 # API reference
   examples/            # Gallery, notebooks
   development/         # Contributing, architecture
```

### Writing Documentation

- Use clear, concise language
- Include code examples
- Add images/diagrams where helpful
- Link to related sections
- Keep examples self-contained

## Release Process

(For maintainers)

1. **Update version:**
   - Bump version in `pyproject.toml`
   - Update `CHANGELOG.md`

2. **Tag release:**
```bash
git tag -a v2.1.0 -m "Release v2.1.0"
git push origin v2.1.0
```

3. **Build and publish:**
```bash
python -m build
twine upload dist/*
```

4. **Update documentation:**
```bash
mkdocs gh-deploy
```

## Getting Help

- **Issues:** GitHub Issues for bugs and features
- **Discussions:** GitHub Discussions for questions
- **Email:** Maintainer email (see pyproject.toml)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors are recognized in:
- `CONTRIBUTORS.md` file
- Release notes
- Documentation acknowledgments

Thank you for contributing to Complexplorer!
