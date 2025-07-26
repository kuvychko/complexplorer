#!/usr/bin/env python3
"""
Generate Gallery Images for Complexplorer Documentation

This script generates all example images for the documentation gallery,
including code snippets that show how to recreate each visualization.

Features:
- Generates 2D phase portraits
- Creates 3D PyVista landscapes (saved as screenshots)
- Produces Riemann sphere visualizations
- Outputs code snippets for each example
- Creates an index file with all examples

Usage:
    python generate_gallery.py [output_dir]

If no output directory is specified, creates 'gallery_output' in current directory.
"""

import numpy as np
import complexplorer as cp
import matplotlib.pyplot as plt
import pyvista as pv
import os
import sys
import json
from datetime import datetime


def ensure_dir(path):
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)
    return path


def sanitize_filename(name):
    """Convert a display name to a safe filename."""
    import re
    # Remove special characters and replace spaces
    name = re.sub(r'[^\w\s-]', '', name)
    name = re.sub(r'[-\s]+', '_', name)
    return name.lower()


class GalleryGenerator:
    """Generate gallery images with associated code snippets."""
    
    def __init__(self, output_dir='gallery_output'):
        self.output_dir = ensure_dir(output_dir)
        self.images_dir = ensure_dir(os.path.join(output_dir, 'images'))
        self.snippets_dir = ensure_dir(os.path.join(output_dir, 'snippets'))
        self.gallery_data = []
        
        # Configure matplotlib
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 150
        plt.rcParams['figure.figsize'] = (6, 6)
        
        # Configure PyVista for off-screen rendering
        pv.global_theme.multi_samples = 8
        pv.global_theme.smooth_shading = True
    
    def add_example(self, category, name, description, image_file, code_snippet):
        """Add an example to the gallery data."""
        self.gallery_data.append({
            'category': category,
            'name': name,
            'description': description,
            'image': image_file,
            'code': code_snippet
        })
    
    def save_snippet(self, name, code):
        """Save code snippet to file."""
        filename = os.path.join(self.snippets_dir, f"{sanitize_filename(name)}.py")
        with open(filename, 'w') as f:
            f.write(code)
        return filename
    
    def generate_2d_examples(self):
        """Generate 2D phase portrait examples."""
        print("\nGenerating 2D Phase Portrait Examples...")
        
        # Example 1: Basic phase portrait
        name = "Basic Phase Portrait"
        print(f"  {name}")
        
        code = """import complexplorer as cp

# Define function and domain
f = lambda z: (z - 1) / (z**2 + z + 1)
domain = cp.Rectangle(4, 4)

# Create basic phase portrait
cp.plot(domain, f, cmap=cp.Phase())"""
        
        # Execute the code
        f = lambda z: (z - 1) / (z**2 + z + 1)
        domain = cp.Rectangle(4, 4)
        
        fig, ax = plt.subplots(figsize=(6, 6))
        cp.plot(domain, f, cmap=cp.Phase(), ax=ax)
        ax.set_title(name)
        
        image_file = os.path.join(self.images_dir, f"{sanitize_filename(name)}.png")
        plt.savefig(image_file, bbox_inches='tight')
        plt.close()
        
        self.save_snippet(name, code)
        self.add_example("2D Visualizations", name, 
                        "Basic phase portrait showing zeros and poles",
                        image_file, code)
        
        # Example 2: Enhanced phase portrait
        name = "Enhanced Phase Portrait"
        print(f"  {name}")
        
        code = """import complexplorer as cp

# Define function and domain
f = lambda z: (z - 1) / (z**2 + z + 1)
domain = cp.Rectangle(4, 4)

# Enhanced phase with auto-scaled square cells
cmap = cp.Phase(n_phi=12, auto_scale_r=True, v_base=0.4)
cp.plot(domain, f, cmap=cmap)"""
        
        fig, ax = plt.subplots(figsize=(6, 6))
        cmap = cp.Phase(n_phi=12, auto_scale_r=True, v_base=0.4)
        cp.plot(domain, f, cmap=cmap, ax=ax)
        ax.set_title(name)
        
        image_file = os.path.join(self.images_dir, f"{sanitize_filename(name)}.png")
        plt.savefig(image_file, bbox_inches='tight')
        plt.close()
        
        self.save_snippet(name, code)
        self.add_example("2D Visualizations", name,
                        "Enhanced phase portrait with modulus information",
                        image_file, code)
        
        # Example 3: Chessboard pattern
        name = "Conformal Chessboard"
        print(f"  {name}")
        
        code = """import complexplorer as cp

# Define function and domain
f = lambda z: (z - 1) / (z**2 + z + 1)
domain = cp.Rectangle(4, 4)

# Cartesian chessboard pattern
cmap = cp.Chessboard(n_squares=20)
cp.plot(domain, f, cmap=cmap)"""
        
        fig, ax = plt.subplots(figsize=(6, 6))
        cmap = cp.Chessboard(n_squares=20)
        cp.plot(domain, f, cmap=cmap, ax=ax)
        ax.set_title(name)
        
        image_file = os.path.join(self.images_dir, f"{sanitize_filename(name)}.png")
        plt.savefig(image_file, bbox_inches='tight')
        plt.close()
        
        self.save_snippet(name, code)
        self.add_example("2D Visualizations", name,
                        "Chessboard pattern showing conformal mapping",
                        image_file, code)
        
        # Example 4: Polar chessboard
        name = "Polar Chessboard"
        print(f"  {name}")
        
        code = """import complexplorer as cp

# Define function and domain
f = lambda z: z**3 - 1
domain = cp.Disk(2)

# Polar chessboard pattern
cmap = cp.PolarChessboard(n_phi=12, n_r=8)
cp.plot(domain, f, cmap=cmap)"""
        
        f = lambda z: z**3 - 1
        domain = cp.Disk(2)
        
        fig, ax = plt.subplots(figsize=(6, 6))
        cmap = cp.PolarChessboard(n_phi=12, n_r=8)
        cp.plot(domain, f, cmap=cmap, ax=ax)
        ax.set_title(name)
        
        image_file = os.path.join(self.images_dir, f"{sanitize_filename(name)}.png")
        plt.savefig(image_file, bbox_inches='tight')
        plt.close()
        
        self.save_snippet(name, code)
        self.add_example("2D Visualizations", name,
                        "Polar chessboard showing radial structure",
                        image_file, code)
        
        # Example 5: Logarithmic rings
        name = "Logarithmic Rings"
        print(f"  {name}")
        
        code = """import complexplorer as cp
import numpy as np

# Exponential function
f = lambda z: np.exp(z)
domain = cp.Rectangle(4, 4)

# Logarithmic rings show modulus levels
cmap = cp.LogRings(base=2)
cp.plot(domain, f, cmap=cmap)"""
        
        f = lambda z: np.exp(z)
        domain = cp.Rectangle(4, 4)
        
        fig, ax = plt.subplots(figsize=(6, 6))
        cmap = cp.LogRings(base=2)
        cp.plot(domain, f, cmap=cmap, ax=ax)
        ax.set_title(name)
        
        image_file = os.path.join(self.images_dir, f"{sanitize_filename(name)}.png")
        plt.savefig(image_file, bbox_inches='tight')
        plt.close()
        
        self.save_snippet(name, code)
        self.add_example("2D Visualizations", name,
                        "Logarithmic rings showing modulus growth",
                        image_file, code)
        
        # Example 6: Domain composition
        name = "Domain Composition"
        print(f"  {name}")
        
        code = """import complexplorer as cp

# Create domain with holes
f = lambda z: (z**2 - 1) / (z**2 + 1)
rect = cp.Rectangle(4, 4)
hole1 = cp.Disk(0.5, center=1)
hole2 = cp.Disk(0.5, center=-1)
domain = rect - hole1 - hole2

# Visualize with enhanced phase
cmap = cp.Phase(n_phi=12, auto_scale_r=True)
cp.plot(domain, f, cmap=cmap)"""
        
        f = lambda z: (z**2 - 1) / (z**2 + 1)
        rect = cp.Rectangle(4, 4)
        hole1 = cp.Disk(0.5, center=1)
        hole2 = cp.Disk(0.5, center=-1)
        domain = rect - hole1 - hole2
        
        fig, ax = plt.subplots(figsize=(6, 6))
        cmap = cp.Phase(n_phi=12, auto_scale_r=True)
        cp.plot(domain, f, cmap=cmap, ax=ax)
        ax.set_title(name)
        
        image_file = os.path.join(self.images_dir, f"{sanitize_filename(name)}.png")
        plt.savefig(image_file, bbox_inches='tight')
        plt.close()
        
        self.save_snippet(name, code)
        self.add_example("2D Visualizations", name,
                        "Complex domain with excluded regions",
                        image_file, code)
    
    def generate_3d_examples(self):
        """Generate 3D landscape examples using PyVista."""
        print("\nGenerating 3D Landscape Examples...")
        
        # Example 1: Basic 3D landscape
        name = "3D Landscape"
        print(f"  {name}")
        
        code = """import complexplorer as cp

# Define function and domain
f = lambda z: (z - 1) / (z**2 + z + 1)
domain = cp.Rectangle(4, 4)

# Create 3D landscape with PyVista
# Use notebook=False for high-quality window
plotter = cp.plot_landscape_pv(
    domain, f,
    cmap=cp.Phase(n_phi=12, auto_scale_r=True),
    resolution=200,
    z_scale=0.4,
    notebook=False,
    show=True
)"""
        
        f = lambda z: (z - 1) / (z**2 + z + 1)
        domain = cp.Rectangle(4, 4)
        
        # Create off-screen plotter
        plotter = cp.plot_landscape_pv(
            domain, f,
            cmap=cp.Phase(n_phi=12, auto_scale_r=True),
            resolution=200,
            z_scale=0.4,
            notebook=False,
            show=False,
            off_screen=True
        )
        
        # Set camera for good angle
        plotter.camera_position = [(6, -6, 4), (0, 0, 0), (0, 0, 1)]
        plotter.set_background('white')
        
        image_file = os.path.join(self.images_dir, f"{sanitize_filename(name)}.png")
        plotter.screenshot(image_file)
        plotter.close()
        
        self.save_snippet(name, code)
        self.add_example("3D Visualizations", name,
                        "3D landscape showing function magnitude",
                        image_file, code)
        
        # Example 2: Side-by-side landscapes
        name = "Domain Codomain Pair"
        print(f"  {name}")
        
        code = """import complexplorer as cp

# Define function and domain
f = lambda z: z**3 - 1
domain = cp.Disk(2)

# Create side-by-side 3D landscapes
plotter = cp.pair_plot_landscape_pv(
    domain, f,
    cmap=cp.Phase(n_phi=12, auto_scale_r=True),
    resolution=200,
    z_scale=0.3,
    notebook=False,
    show=True,
    window_size=(1600, 800)
)"""
        
        f = lambda z: z**3 - 1
        domain = cp.Disk(2)
        
        plotter = cp.pair_plot_landscape_pv(
            domain, f,
            cmap=cp.Phase(n_phi=12, auto_scale_r=True),
            resolution=200,
            z_scale=0.3,
            notebook=False,
            show=False,
            off_screen=True,
            window_size=(1600, 800)
        )
        
        plotter.set_background('white')
        
        image_file = os.path.join(self.images_dir, f"{sanitize_filename(name)}.png")
        plotter.screenshot(image_file)
        plotter.close()
        
        self.save_snippet(name, code)
        self.add_example("3D Visualizations", name,
                        "Side-by-side domain and codomain landscapes",
                        image_file, code)
    
    def generate_riemann_examples(self):
        """Generate Riemann sphere examples."""
        print("\nGenerating Riemann Sphere Examples...")
        
        # Example 1: Traditional Riemann sphere
        name = "Riemann Sphere"
        print(f"  {name}")
        
        code = """import complexplorer as cp

# Rational function with poles
f = lambda z: (z**2 - 1) / (z**2 + 1)

# Create Riemann sphere visualization
plotter = cp.riemann_pv(
    f,
    cmap=cp.Phase(n_phi=16),
    modulus_scaling='constant',
    resolution=150,
    notebook=False,
    show=True
)"""
        
        f = lambda z: (z**2 - 1) / (z**2 + 1)
        
        plotter = cp.riemann_pv(
            f,
            cmap=cp.Phase(n_phi=16),
            modulus_scaling='constant',
            resolution=150,
            notebook=False,
            show=False,
            off_screen=True
        )
        
        plotter.camera_position = [(2, -2, 1.5), (0, 0, 0), (0, 0, 1)]
        plotter.set_background('white')
        
        image_file = os.path.join(self.images_dir, f"{sanitize_filename(name)}.png")
        plotter.screenshot(image_file)
        plotter.close()
        
        self.save_snippet(name, code)
        self.add_example("Riemann Sphere", name,
                        "Traditional Riemann sphere projection",
                        image_file, code)
        
        # Example 2: Modulus-scaled Riemann sphere
        name = "Modulus Scaled Sphere"
        print(f"  {name}")
        
        code = """import complexplorer as cp

# Function with interesting modulus behavior
f = lambda z: (z - 1) * (z + 1) / (z**2 + 0.5)

# Riemann sphere with arctan modulus scaling
plotter = cp.riemann_pv(
    f,
    cmap=cp.Phase(n_phi=12, auto_scale_r=True),
    modulus_scaling='arctan',
    scaling_params={'r_min': 0.3, 'r_max': 1.0},
    resolution=200,
    notebook=False,
    show=True
)"""
        
        f = lambda z: (z - 1) * (z + 1) / (z**2 + 0.5)
        
        plotter = cp.riemann_pv(
            f,
            cmap=cp.Phase(n_phi=12, auto_scale_r=True),
            modulus_scaling='arctan',
            scaling_params={'r_min': 0.3, 'r_max': 1.0},
            resolution=200,
            notebook=False,
            show=False,
            off_screen=True
        )
        
        plotter.camera_position = [(2, -2, 1.5), (0, 0, 0), (0, 0, 1)]
        plotter.set_background('white')
        
        image_file = os.path.join(self.images_dir, f"{sanitize_filename(name)}.png")
        plotter.screenshot(image_file)
        plotter.close()
        
        self.save_snippet(name, code)
        self.add_example("Riemann Sphere", name,
                        "Riemann sphere with modulus-based distortion",
                        image_file, code)
    
    def generate_special_examples(self):
        """Generate examples of special functions."""
        print("\nGenerating Special Function Examples...")
        
        # Example 1: Essential singularity
        name = "Essential Singularity"
        print(f"  {name}")
        
        code = """import complexplorer as cp
import numpy as np

# Essential singularity at origin
f = lambda z: np.exp(1/z)
domain = cp.Annulus(0.05, 0.5)

# High resolution needed near singularity
cp.plot(domain, f, 
        cmap=cp.Phase(n_phi=24),
        resolution=600)"""
        
        f = lambda z: np.exp(1/z)
        domain = cp.Annulus(0.05, 0.5)
        
        fig, ax = plt.subplots(figsize=(6, 6))
        cp.plot(domain, f, cmap=cp.Phase(n_phi=24), resolution=600, ax=ax)
        ax.set_title(name)
        
        image_file = os.path.join(self.images_dir, f"{sanitize_filename(name)}.png")
        plt.savefig(image_file, bbox_inches='tight')
        plt.close()
        
        self.save_snippet(name, code)
        self.add_example("Special Functions", name,
                        "Chaotic behavior near essential singularity",
                        image_file, code)
        
        # Example 2: Branch cut
        name = "Branch Cut"
        print(f"  {name}")
        
        code = """import complexplorer as cp
import numpy as np

# Complex logarithm with branch cut
f = lambda z: np.log(z)
domain = cp.Annulus(0.1, 3)

# Phase portrait shows branch cut clearly
cp.plot(domain, f, 
        cmap=cp.Phase(n_phi=12),
        resolution=400)"""
        
        f = lambda z: np.log(z)
        domain = cp.Annulus(0.1, 3)
        
        fig, ax = plt.subplots(figsize=(6, 6))
        cp.plot(domain, f, cmap=cp.Phase(n_phi=12), resolution=400, ax=ax)
        ax.set_title(name)
        
        image_file = os.path.join(self.images_dir, f"{sanitize_filename(name)}.png")
        plt.savefig(image_file, bbox_inches='tight')
        plt.close()
        
        self.save_snippet(name, code)
        self.add_example("Special Functions", name,
                        "Branch cut of complex logarithm",
                        image_file, code)
    
    def generate_index(self):
        """Generate index file with all examples."""
        print("\nGenerating gallery index...")
        
        # Group by category
        categories = {}
        for item in self.gallery_data:
            cat = item['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(item)
        
        # Generate HTML index
        html = """<!DOCTYPE html>
<html>
<head>
    <title>Complexplorer Gallery</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .gallery { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .item { border: 1px solid #ddd; padding: 10px; }
        .item img { width: 100%; height: auto; }
        .item h3 { margin-top: 10px; }
        .item p { color: #666; }
        pre { background: #f4f4f4; padding: 10px; overflow-x: auto; }
        h2 { color: #333; margin-top: 40px; }
    </style>
</head>
<body>
    <h1>Complexplorer Gallery</h1>
    <p>Generated on {date}</p>
""".format(date=datetime.now().strftime("%Y-%m-%d %H:%M"))
        
        for category, items in categories.items():
            html += f"\n    <h2>{category}</h2>\n    <div class='gallery'>\n"
            
            for item in items:
                # Make paths relative
                img_rel = os.path.relpath(item['image'], self.output_dir)
                
                html += f"""        <div class='item'>
            <img src='{img_rel}' alt='{item['name']}'>
            <h3>{item['name']}</h3>
            <p>{item['description']}</p>
            <details>
                <summary>Show code</summary>
                <pre>{item['code']}</pre>
            </details>
        </div>
"""
            
            html += "    </div>\n"
        
        html += """
</body>
</html>"""
        
        index_file = os.path.join(self.output_dir, 'index.html')
        with open(index_file, 'w') as f:
            f.write(html)
        
        # Also save as JSON for programmatic access
        json_file = os.path.join(self.output_dir, 'gallery.json')
        with open(json_file, 'w') as f:
            json.dump(self.gallery_data, f, indent=2)
        
        print(f"  Created: {index_file}")
        print(f"  Created: {json_file}")
    
    def generate_all(self):
        """Generate all gallery images and index."""
        print(f"Generating gallery in: {self.output_dir}")
        
        self.generate_2d_examples()
        self.generate_3d_examples()
        self.generate_riemann_examples()
        self.generate_special_examples()
        self.generate_index()
        
        print(f"\nGallery generation complete!")
        print(f"Total examples: {len(self.gallery_data)}")
        print(f"Output directory: {self.output_dir}")
        print(f"View gallery: file://{os.path.abspath(os.path.join(self.output_dir, 'index.html'))}")


def main():
    """Main entry point."""
    # Get output directory from command line or use default
    output_dir = sys.argv[1] if len(sys.argv) > 1 else 'gallery_output'
    
    # Create gallery generator and run
    generator = GalleryGenerator(output_dir)
    generator.generate_all()


if __name__ == "__main__":
    main()