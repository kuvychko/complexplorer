"""Constants and default values for complexplorer.

This module centralizes all magic numbers and hard-coded values used throughout
the library for better maintainability and documentation.
"""

# =============================================================================
# Colormap Constants
# =============================================================================

# Phase portrait default appearance
DEFAULT_BASE_BRIGHTNESS = 0.6  # Base V (value) in HSV for phase portraits
DEFAULT_BRIGHTNESS_CONTRAST = 0.3  # Modulus-based brightness variation
DEFAULT_AUTO_SCALE_RADIUS = 0.8  # Scale factor for auto-sized square cells

# Out-of-domain masking
OUT_OF_DOMAIN_HUE = 0.0  # Hue for masked regions (red)
OUT_OF_DOMAIN_SATURATION = 0.0  # Saturation for masked regions (gray)
OUT_OF_DOMAIN_VALUE = 0.5  # Brightness for masked regions (medium gray)

# =============================================================================
# Resolution and Quality Settings
# =============================================================================

# Resolution limits (number of points along longest edge)
MIN_RESOLUTION = 10  # Below this, severe visual artifacts occur
MAX_RESOLUTION = 1000  # Above this, memory/performance issues on most systems
RECOMMENDED_2D_RESOLUTION = 500  # Good balance for 2D plots
RECOMMENDED_3D_RESOLUTION = 100  # Good balance for 3D plots
DEFAULT_RIEMANN_RESOLUTION = 100  # Default for Riemann sphere plots

# =============================================================================
# Numerical Tolerances
# =============================================================================

# Stereographic projection
STEREOGRAPHIC_POLE_TOLERANCE = 1e-8  # Offset from poles to avoid singularities
POLE_DETECTION_THRESHOLD = 1e-10  # For detecting points at poles

# Mesh cleaning and validation
MESH_CLEAN_TOLERANCE = 1e-9  # Tolerance for removing duplicate vertices

# =============================================================================
# STL Export and 3D Printing
# =============================================================================

# Physical printing constraints (in millimeters)
MIN_PRINTABLE_SIZE_MM = 1.0  # Below this, most printers struggle with detail
MAX_PRINTABLE_SIZE_MM = 500.0  # Above this, exceeds most printer build volumes
MIN_WALL_THICKNESS_MM = 0.8  # Minimum wall thickness for FDM printers
RECOMMENDED_ORNAMENT_SIZE_MM = 50.0  # Good default size for desk ornaments

# STL mesh quality
DEFAULT_STL_RESOLUTION = 150  # Higher resolution for printable quality
MAX_STL_FILE_SIZE_MB = 100.0  # Warning threshold for STL file size

# =============================================================================
# PyVista Rendering
# =============================================================================

# Material and lighting parameters (Phong shading model)
DEFAULT_MATERIAL_PARAMS = {
    'specular': 0.5,  # Specular reflection strength (shininess)
    'specular_power': 15,  # Specular highlight sharpness (Phong exponent)
    'diffuse': 0.7,  # Diffuse reflection (matte appearance)
    'ambient': 0.3,  # Ambient light contribution (minimum brightness)
}

# Alternative material presets
MATERIAL_PRESETS = {
    'matte': {
        'specular': 0.1,
        'specular_power': 5,
        'diffuse': 0.9,
        'ambient': 0.3,
    },
    'glossy': {
        'specular': 0.8,
        'specular_power': 30,
        'diffuse': 0.5,
        'ambient': 0.2,
    },
    'metallic': {
        'specular': 0.9,
        'specular_power': 50,
        'diffuse': 0.3,
        'ambient': 0.1,
    },
}

# Window sizes (pixels)
DEFAULT_WINDOW_SIZE = (800, 600)  # Standard window
DEFAULT_LANDSCAPE_PAIR_SIZE = (1200, 600)  # Side-by-side landscapes
DEFAULT_RIEMANN_WINDOW_SIZE = (800, 800)  # Square window for sphere

# =============================================================================
# Modulus Scaling
# =============================================================================

# Default parameters for various scaling modes
# (Note: Full presets are in scaling.py - these are component defaults)
DEFAULT_SIGMOID_STEEPNESS = 2.0  # Controls sigmoid curve steepness
DEFAULT_SIGMOID_CENTER = 1.0  # Center point of sigmoid transition
DEFAULT_ARCTAN_SCALE = 2.0  # Scale factor for arctan scaling
DEFAULT_POWER_EXPONENT = 0.5  # Power for root scaling (sqrt by default)

# Adaptive scaling percentiles
ADAPTIVE_LOW_PERCENTILE = 5  # Lower percentile for adaptive scaling
ADAPTIVE_HIGH_PERCENTILE = 95  # Upper percentile for adaptive scaling

# =============================================================================
# Riemann Sphere Grid
# =============================================================================

# Latitude/longitude grid defaults
DEFAULT_LATITUDE_LINES = 10  # Number of latitude circles
DEFAULT_LONGITUDE_LINES = 12  # Number of longitude meridians
GRID_LINE_WIDTH = 1.0  # Default grid line width
GRID_LINE_OPACITY = 0.5  # Default grid line opacity

# Sphere mesh generation
AVOID_POLES_OFFSET = 0.01  # Offset from exact poles (in radians)

# =============================================================================
# Domain and Plotting
# =============================================================================

# Domain defaults
DEFAULT_RECTANGLE_SIZE = 4.0  # Default rectangle side length
DEFAULT_DISK_RADIUS = 2.0  # Default disk radius

# Plotting margins
DEFAULT_RIEMANN_CHART_MARGIN = 0.05  # Margin around unit disk
DEFAULT_UNIT_CIRCLE_WIDTH = 1.0  # Width of unit circle highlight

# Aspect ratio tolerance
ASPECT_RATIO_TOLERANCE = 1e-6  # For floating point aspect ratio comparisons

# =============================================================================
# Validation
# =============================================================================

# Parameter bounds
MAX_PHASE_SECTORS = 100  # Reasonable upper limit for phase sectors
MIN_PHASE_SECTORS = 1  # Must have at least one sector

# File validation
MAX_FILENAME_LENGTH = 255  # Maximum filename length (OS limit)
VALID_STL_EXTENSIONS = ['.stl', '.STL']  # Valid STL file extensions
VALID_IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.svg', '.pdf', '.eps']

# =============================================================================
# Performance
# =============================================================================

# Memory management
MAX_ARRAY_SIZE_WARNING = 10_000_000  # Warn for arrays larger than this
LARGE_MESH_THRESHOLD = 1_000_000  # Threshold for "large" meshes

# Parallel processing
DEFAULT_NUM_WORKERS = 4  # Default number of worker processes
