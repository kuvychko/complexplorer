# Interactive Complexplorer Demo

The `interactive_demo.py` script provides an interactive command-line interface to explore complex functions with PyVista 3D visualizations.

## Features

- **Interactive Menu System**: Choose from multiple pre-defined functions, color schemes, domains, and visualization types
- **Orientation Axes**: All visualizations include Re/Im/Z orientation axes for spatial awareness
- **High-Quality Rendering**: Utilizes PyVista's advanced rendering capabilities with anti-aliasing

## Usage

Run the script from the command line:

```bash
python examples/interactive_demo.py
```

Or make it executable and run directly:

```bash
chmod +x examples/interactive_demo.py
./examples/interactive_demo.py
```

## Available Options

### Visualization Types
1. **3D Landscape**: Single 3D surface plot of the complex function
2. **Domain/Codomain Pair**: Side-by-side comparison of domain and codomain
3. **Riemann Sphere**: Visualization on the Riemann sphere with optional modulus scaling

### Complex Functions
1. Möbius transformation: f(z) = (z - 1) / (z + 1)
2. Quadratic: f(z) = z²
3. Cubic: f(z) = z³ - 1
4. Rational function: f(z) = (z² - 1) / (z² + 1)
5. Sine: f(z) = sin(z)
6. Exponential: f(z) = e^z
7. Complex logarithm: f(z) = log(z + 0.1i)
8. Reciprocal: f(z) = 1/z

### Color Schemes
1. Basic phase portrait
2. Enhanced phase (6 colors)
3. Enhanced phase (12 colors)
4. Enhanced phase (6 colors, auto-scaled)
5. Enhanced phase (24 colors)
6. Chessboard pattern
7. Polar chessboard
8. Logarithmic rings

### Domains
1. Square (4×4)
2. Wide rectangle (6×3)
3. Disk (radius 2)
4. Annulus (0.5 to 2)
5. Large square (8×8)

### Resolution Options
1. Low (50 points) - Fast
2. Medium (100 points) - Balanced
3. High (150 points) - Quality
4. Very High (200 points) - Slow

### Z-axis Scaling Options (for landscape plots)
1. Standard (max height = 10) - Good for most functions
2. Extended (max height = 20) - For functions with larger values
3. Full range (no limit) - Shows all values without clipping
4. Custom value - Set your own maximum height

## Interactive Controls

Once a visualization window opens:
- **Left Mouse**: Rotate the view
- **Middle Mouse**: Pan the view
- **Right Mouse/Scroll**: Zoom in/out
- **R**: Reset camera to default position
- **S**: Take a screenshot
- **Q**: Close the window and return to menu

## Tips

- Start with lower resolutions to quickly explore different functions
- The orientation axes (Re/Im/Z) help you maintain spatial awareness while rotating
- For Riemann sphere visualizations, try both constant radius and modulus scaling
- Close the visualization window to return to the menu and try another combination

## Requirements

- Python 3.11+
- complexplorer (with PyVista support)
- PyVista
- NumPy

## Troubleshooting

If you encounter shader errors or black screens:
1. Update your graphics drivers
2. Try reducing the resolution
3. Disable anti-aliasing by modifying the script

For the best experience, ensure you have a GPU with OpenGL support.