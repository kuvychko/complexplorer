# Gallery Image Generation Scripts

This directory contains Python scripts to generate the example images used in the complexplorer documentation.

## Scripts

### `generate_gallery_images.py`
Generates the main gallery images showcasing different color maps in 2D and 3D:
- Phase portraits (basic, phase-enhanced, modulus-enhanced, fully enhanced)
- Chessboard patterns (Cartesian, polar with linear/log spacing)
- Logarithmic rings
- Riemann sphere visualizations

### `generate_additional_examples.py`
Generates supplementary example images:
- Basic plot examples with different resolutions
- Domain examples (annulus)
- Analytic vs non-analytic function comparison
- 3D landscape variations

## Usage

To regenerate all gallery images:

```bash
cd examples/gallery
python generate_gallery_images.py
python generate_additional_examples.py
```

The scripts will create all PNG files in the current directory.

## Note

These scripts were extracted from the original `plots_example.ipynb` notebook to make gallery image generation more maintainable and reproducible.