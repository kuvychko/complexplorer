# Application Examples

This folder contains application notebooks demonstrating Complexplorer in various mathematical contexts beyond the core tutorials.

## Applications

### 1. FFT/DFT Matrices (app_01_fft_matrices.ipynb) (~15 min)
**Discrete Fourier Transform Matrix Visualization**
- DFT matrix structure: W_N^{jk} where W_N = e^{-2πi/N}
- Visualization grid: 2 rows × 4 columns (N = 32, 64, 128, 256, 512, 1024, 2048, 4096)
- Circular symmetry and frequency bin structure
- Extensions: Hadamard, Toeplitz, DCT matrices
- **Prerequisites**: Notebook #1 (basic visualization)

### 2. Special Functions (app_02_special_functions.ipynb) (~20 min)
**Gamma, Zeta, Bessel, and Elliptic Functions**
- Gamma function Γ(z): poles at negative integers
- Riemann zeta function ζ(z): famous zeros on critical line
- Bessel functions J_n(z): oscillatory behavior
- Elliptic functions: doubly periodic structures
- Pedagogical value: different singularity types
- **Prerequisites**: Notebooks #1, #6 (Riemann sphere)

### 3. Conformal Maps (app_03_conformal_maps.ipynb) (~20 min)
**Important Conformal Transformations**
- Joukowsky transformation: z + 1/z (airfoil design)
- Möbius transformations: (az+b)/(cz+d)
- Exponential and logarithm branches
- Grid transformation visualization
- Applications: fluid dynamics, electrostatics
- **Prerequisites**: Notebook #1

### 4. Complex Dynamics (app_04_complex_dynamics.ipynb) (~30 min)
**Newton's Method and Iteration**
- Newton's method basins of attraction
- Fractal boundaries between basins
- Iteration of rational functions
- Parameter space exploration
- Connection to Julia sets and fractals
- **Prerequisites**: Notebook #1

## Mathematical Context

These applications demonstrate:
- **Structured matrices** (FFT): Linear algebra meets complex analysis
- **Special functions**: Classical analysis and number theory
- **Conformal maps**: Geometric function theory
- **Dynamics**: Iteration and chaos theory

## Extension Ideas

- Other matrix structures (Vandermonde, Circulant)
- Theta functions and modular forms (number theory)
- Schwarz-Christoffel mapping (polygon transformations)
- Mandelbrot set parameter space exploration
- Analytic continuation demonstrations

## See Also

- `../notebooks/` - Tutorial notebooks (core concepts)
- `../../docs/examples/` - Additional examples in documentation
- External resources:
  - Needham: "Visual Complex Analysis"
  - Wegert: "Visual Complex Functions"
  - Peitgen & Richter: "The Beauty of Fractals"
