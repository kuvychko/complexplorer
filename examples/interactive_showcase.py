#!/usr/bin/env python3
"""
Complexplorer Interactive Showcase

A comprehensive interactive demonstration of complexplorer's capabilities including:
- 2D phase portraits
- 3D landscapes (PyVista)
- Riemann sphere projections
- STL export for 3D printing
- Batch processing capabilities

This script provides a menu-driven interface for exploring complex functions
with various visualization types and export options.
"""

import complexplorer as cp
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import os
import sys


# Pre-defined complex functions (expanded collection)
FUNCTIONS = {
    "1": {
        "name": "Möbius transformation",
        "func": lambda z: (z - 1) / (z + 1),
        "description": "f(z) = (z - 1) / (z + 1)",
        "category": "rational"
    },
    "2": {
        "name": "Quadratic polynomial",
        "func": lambda z: z**2 - 1,
        "description": "f(z) = z² - 1",
        "category": "polynomial"
    },
    "3": {
        "name": "Cubic polynomial",
        "func": lambda z: z**3 - 1,
        "description": "f(z) = z³ - 1",
        "category": "polynomial"
    },
    "4": {
        "name": "Rational with poles",
        "func": lambda z: (z**2 - 1) / (z**2 + z + 1),
        "description": "f(z) = (z² - 1) / (z² + z + 1)",
        "category": "rational"
    },
    "5": {
        "name": "Sine function",
        "func": lambda z: np.sin(z),
        "description": "f(z) = sin(z)",
        "category": "transcendental"
    },
    "6": {
        "name": "Exponential function",
        "func": lambda z: np.exp(z),
        "description": "f(z) = e^z",
        "category": "transcendental"
    },
    "7": {
        "name": "Complex logarithm",
        "func": lambda z: np.log(z + 0.1j),
        "description": "f(z) = log(z + 0.1i)",
        "category": "transcendental"
    },
    "8": {
        "name": "Reciprocal (pole at origin)",
        "func": lambda z: 1 / z,
        "description": "f(z) = 1/z",
        "category": "rational"
    },
    "9": {
        "name": "Essential singularity",
        "func": lambda z: np.exp(1/z),
        "description": "f(z) = e^(1/z)",
        "category": "essential"
    },
    "10": {
        "name": "Composite function",
        "func": lambda z: np.sin(z**2),
        "description": "f(z) = sin(z²)",
        "category": "composite"
    }
}

# Enhanced color schemes
COLOR_SCHEMES = {
    "1": {
        "name": "Basic phase portrait",
        "cmap": cp.Phase(),
        "best_for": "zeros and poles"
    },
    "2": {
        "name": "Phase with 6 sectors",
        "cmap": cp.Phase(n_phi=6),
        "best_for": "simple structure"
    },
    "3": {
        "name": "Phase with 12 sectors",
        "cmap": cp.Phase(n_phi=12),
        "best_for": "detailed phase"
    },
    "4": {
        "name": "Enhanced phase (auto-scaled)",
        "cmap": cp.Phase(n_phi=12, auto_scale_r=True, v_base=0.4),
        "best_for": "phase and modulus"
    },
    "5": {
        "name": "Fine enhanced phase",
        "cmap": cp.Phase(n_phi=24, auto_scale_r=True, scale_radius=0.5),
        "best_for": "high detail"
    },
    "6": {
        "name": "Cartesian chessboard",
        "cmap": cp.Chessboard(n_squares=30),
        "best_for": "conformal mapping"
    },
    "7": {
        "name": "Polar chessboard",
        "cmap": cp.PolarChessboard(n_phi=16, n_r=8),
        "best_for": "radial structure"
    },
    "8": {
        "name": "Logarithmic rings",
        "cmap": cp.LogRings(base=2),
        "best_for": "modulus levels"
    }
}

# Domain presets
DOMAINS = {
    "1": {
        "name": "Square (4x4)",
        "domain": cp.Rectangle(4, 4)
    },
    "2": {
        "name": "Wide rectangle (6x3)",
        "domain": cp.Rectangle(6, 3)
    },
    "3": {
        "name": "Disk (radius 2)",
        "domain": cp.Disk(2)
    },
    "4": {
        "name": "Annulus (0.5 to 2)",
        "domain": cp.Annulus(0.5, 2)
    },
    "5": {
        "name": "Large square (8x8)",
        "domain": cp.Rectangle(8, 8)
    },
    "6": {
        "name": "Disk with hole",
        "domain": cp.Disk(3) - cp.Disk(0.5)
    },
    "7": {
        "name": "Custom (user input)",
        "domain": None  # Will be created based on user input
    }
}


def print_menu(title, options):
    """Print a formatted menu with categories if available."""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print('='*60)
    
    # Group by category if available
    categories = {}
    for key, value in options.items():
        category = value.get('category', 'Other')
        if category not in categories:
            categories[category] = []
        categories[category].append((key, value))
    
    # Print grouped options
    for category, items in categories.items():
        if len(categories) > 1:
            print(f"\n{category.upper()}:")
        for key, value in items:
            if 'description' in value:
                print(f"  {key}. {value['name']}: {value['description']}")
            elif 'best_for' in value:
                print(f"  {key}. {value['name']} (best for: {value['best_for']})")
            else:
                print(f"  {key}. {value['name']}")
    
    print('='*60)


def get_choice(prompt, valid_choices):
    """Get a valid choice from the user."""
    while True:
        choice = input(f"\n{prompt}: ").strip()
        if choice in valid_choices:
            return choice
        print(f"Invalid choice. Please enter one of: {', '.join(valid_choices)}")


def create_custom_domain():
    """Create a custom domain based on user input."""
    print("\nCustom Domain Creation:")
    print("1. Rectangle")
    print("2. Disk")
    print("3. Annulus")
    
    domain_type = get_choice("Select domain type (1-3)", ["1", "2", "3"])
    
    try:
        if domain_type == "1":
            width = float(input("Enter width (real axis): "))
            height = float(input("Enter height (imaginary axis): "))
            center_re = float(input("Enter center real part (default 0): ") or "0")
            center_im = float(input("Enter center imaginary part (default 0): ") or "0")
            return cp.Rectangle(width, height, center=center_re + 1j*center_im)
        
        elif domain_type == "2":
            radius = float(input("Enter radius: "))
            center_re = float(input("Enter center real part (default 0): ") or "0")
            center_im = float(input("Enter center imaginary part (default 0): ") or "0")
            return cp.Disk(radius, center=center_re + 1j*center_im)
        
        elif domain_type == "3":
            inner = float(input("Enter inner radius: "))
            outer = float(input("Enter outer radius: "))
            center_re = float(input("Enter center real part (default 0): ") or "0")
            center_im = float(input("Enter center imaginary part (default 0): ") or "0")
            return cp.Annulus(inner, outer, center=center_re + 1j*center_im)
    
    except ValueError:
        print("Invalid input. Using default Rectangle(4, 4).")
        return cp.Rectangle(4, 4)


def visualization_2d(func_data, cmap, domain, resolution):
    """Create 2D phase portrait visualization."""
    print("\n2D Visualization Options:")
    print("1. Single plot")
    print("2. Side-by-side (domain and codomain)")
    
    plot_type = get_choice("Select plot type (1-2)", ["1", "2"])
    
    if plot_type == "1":
        fig, ax = plt.subplots(figsize=(8, 8))
        cp.plot(domain, func_data["func"], cmap=cmap, resolution=resolution, ax=ax)
        ax.set_title(f"{func_data['name']}: {func_data['description']}")
        plt.show()
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        cp.pair_plot(domain, func_data["func"], cmap=cmap, resolution=resolution, 
                     ax_domain=ax1, ax_codomain=ax2)
        fig.suptitle(f"{func_data['name']}: {func_data['description']}")
        plt.show()


def visualization_3d_landscape(func_data, cmap, domain, resolution):
    """Create 3D landscape visualization."""
    print("\nZ-axis scaling options:")
    print("1. Automatic")
    print("2. Fixed scale (z_scale = 0.3)")
    print("3. Extended scale (z_scale = 0.5)")
    print("4. Custom scale")
    
    z_choice = get_choice("Select Z scaling (1-4)", ["1", "2", "3", "4"])
    
    z_scale = None
    if z_choice == "2":
        z_scale = 0.3
    elif z_choice == "3":
        z_scale = 0.5
    elif z_choice == "4":
        z_scale = float(input("Enter z_scale value (0.1 to 1.0): "))
    
    print("\nCreating 3D landscape (high-quality external window)...")
    
    plotter = cp.plot_landscape_pv(
        domain,
        func_data["func"],
        cmap=cmap,
        resolution=resolution,
        z_scale=z_scale,
        notebook=False,  # High quality
        show=True,
        title=f"{func_data['name']}: {func_data['description']}",
        window_size=(1000, 800)
    )
    
    return plotter


def visualization_riemann_sphere(func_data, cmap, resolution):
    """Create Riemann sphere visualization."""
    print("\nModulus Scaling Options:")
    print("1. Constant radius (traditional)")
    print("2. Arctan scaling (smooth compression)")
    print("3. Logarithmic scaling")
    print("4. Linear clamping")
    
    scaling_choice = get_choice("Select scaling (1-4)", ["1", "2", "3", "4"])
    
    scaling_map = {
        "1": ("constant", {'radius': 1.0}),
        "2": ("arctan", {'r_min': 0.3, 'r_max': 1.0}),
        "3": ("logarithmic", {'base': 2, 'r_min': 0.3, 'r_max': 1.2}),
        "4": ("linear_clamp", {'m_max': 5, 'r_min': 0.4, 'r_max': 1.0})
    }
    
    modulus_scaling, scaling_params = scaling_map[scaling_choice]
    
    print("\nCreating Riemann sphere (high-quality external window)...")
    
    plotter = cp.riemann_pv(
        func_data["func"],
        cmap=cmap,
        modulus_scaling=modulus_scaling,
        scaling_params=scaling_params,
        resolution=resolution,
        notebook=False,  # High quality
        show=True,
        title=f"{func_data['name']} on Riemann Sphere",
        window_size=(900, 900)
    )
    
    return plotter


def stl_export(func_data, cmap):
    """Export function as STL for 3D printing."""
    print("\nSTL Export Options:")
    
    # Domain selection for STL
    print("\nDomain for STL (avoiding singularities is recommended):")
    print("1. Full sphere")
    print("2. Annulus (exclude origin)")
    print("3. Custom annulus")
    
    domain_choice = get_choice("Select domain (1-3)", ["1", "2", "3"])
    
    domain = None
    if domain_choice == "2":
        domain = cp.Annulus(0.1, 5)
    elif domain_choice == "3":
        inner = float(input("Inner radius: "))
        outer = float(input("Outer radius: "))
        domain = cp.Annulus(inner, outer)
    
    # Resolution
    print("\nSTL Resolution:")
    print("1. Draft (100) - Fast")
    print("2. Standard (150)")
    print("3. High (200)")
    print("4. Ultra (300) - Slow")
    
    res_choice = get_choice("Select resolution (1-4)", ["1", "2", "3", "4"])
    resolution = {"1": 100, "2": 150, "3": 200, "4": 300}[res_choice]
    
    # Scaling
    print("\nModulus scaling for STL:")
    print("1. Arctan (recommended)")
    print("2. Logarithmic")
    print("3. Linear clamp")
    
    scale_choice = get_choice("Select scaling (1-3)", ["1", "2", "3"])
    
    scaling_map = {
        "1": ("arctan", {'r_min': 0.3, 'r_max': 1.0}),
        "2": ("logarithmic", {'base': 2, 'r_min': 0.3, 'r_max': 1.2}),
        "3": ("linear_clamp", {'m_max': 5, 'r_min': 0.4, 'r_max': 1.0})
    }
    
    scaling, scaling_params = scaling_map[scale_choice]
    
    # Size
    size_mm = float(input("\nOrnament size in mm (default 80): ") or "80")
    
    # Filename
    base_name = func_data['name'].lower().replace(' ', '_')
    filename = input(f"\nFilename (default: {base_name}.stl): ") or f"{base_name}.stl"
    
    print(f"\nGenerating STL file: {filename}")
    
    try:
        from complexplorer.export.stl import OrnamentGenerator
        
        generator = OrnamentGenerator(
            func=func_data["func"],
            resolution=resolution,
            scaling=scaling,
            scaling_params=scaling_params,
            cmap=cmap,
            domain=domain
        )
        
        output_file = generator.generate_ornament(
            output_file=filename,
            size_mm=size_mm,
            smooth=True,
            verbose=True
        )
        
        print(f"\nSTL file successfully created: {output_file}")
        print("You can now import this into your 3D printing software.")
        
    except Exception as e:
        print(f"\nError generating STL: {e}")


def batch_processing():
    """Batch process multiple functions."""
    print("\nBatch Processing Mode")
    print("This will generate visualizations for multiple functions.")
    
    # Select functions
    print("\nSelect functions to process (comma-separated, e.g., 1,3,5):")
    print_menu("Available Functions", FUNCTIONS)
    
    func_choices = input("Enter function numbers: ").strip().split(',')
    func_choices = [c.strip() for c in func_choices if c.strip() in FUNCTIONS]
    
    if not func_choices:
        print("No valid functions selected.")
        return
    
    # Select colormap
    print_menu("Select Color Scheme", COLOR_SCHEMES)
    cmap_choice = get_choice("Select color scheme", COLOR_SCHEMES.keys())
    selected_cmap = COLOR_SCHEMES[cmap_choice]["cmap"]
    
    # Select output type
    print("\nOutput type:")
    print("1. PNG images (2D)")
    print("2. STL files (3D printing)")
    print("3. Both")
    
    output_type = get_choice("Select output (1-3)", ["1", "2", "3"])
    
    # Create output directory
    output_dir = "batch_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each function
    for func_id in func_choices:
        func_data = FUNCTIONS[func_id]
        print(f"\nProcessing: {func_data['name']}")
        
        if output_type in ["1", "3"]:
            # Generate 2D PNG
            fig, ax = plt.subplots(figsize=(8, 8))
            domain = cp.Rectangle(4, 4)
            
            # Special handling for functions with singularities
            if func_id in ["8", "9"]:  # Reciprocal or essential singularity
                domain = cp.Annulus(0.1, 3)
            
            cp.plot(domain, func_data["func"], cmap=selected_cmap, resolution=300, ax=ax)
            ax.set_title(f"{func_data['name']}: {func_data['description']}")
            
            png_file = os.path.join(output_dir, f"{func_id}_{func_data['name'].lower().replace(' ', '_')}.png")
            plt.savefig(png_file, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {png_file}")
        
        if output_type in ["2", "3"]:
            # Generate STL
            try:
                from complexplorer.export.stl import OrnamentGenerator
                
                # Use annulus for functions with singularities
                domain = None
                if func_id in ["8", "9"]:
                    domain = cp.Annulus(0.1, 5)
                
                generator = OrnamentGenerator(
                    func=func_data["func"],
                    resolution=150,
                    scaling='arctan',
                    scaling_params={'r_min': 0.3, 'r_max': 1.0},
                    cmap=selected_cmap,
                    domain=domain
                )
                
                stl_file = os.path.join(output_dir, f"{func_id}_{func_data['name'].lower().replace(' ', '_')}.stl")
                generator.generate_ornament(
                    output_file=stl_file,
                    size_mm=80,
                    smooth=True,
                    verbose=False
                )
                print(f"  Saved: {stl_file}")
                
            except Exception as e:
                print(f"  Error generating STL: {e}")
    
    print(f"\nBatch processing complete! Files saved in: {output_dir}/")


def main():
    """Main interactive showcase."""
    print("\n" + "="*70)
    print("COMPLEXPLORER INTERACTIVE SHOWCASE".center(70))
    print("High-Quality Visualizations & 3D Printing".center(70))
    print("="*70)
    
    # Configure PyVista for better quality
    pv.global_theme.multi_samples = 8
    pv.global_theme.smooth_shading = True
    
    while True:
        # Main menu
        print("\nMain Menu:")
        print("1. 2D Phase Portraits")
        print("2. 3D Landscapes (PyVista)")
        print("3. Riemann Sphere")
        print("4. STL Export (3D Printing)")
        print("5. Batch Processing")
        print("6. Help & Tips")
        print("7. Quit")
        
        main_choice = get_choice("Select option (1-7)", ["1", "2", "3", "4", "5", "6", "7"])
        
        if main_choice == "7":
            print("\nThank you for using Complexplorer!")
            break
        
        if main_choice == "6":
            # Help section
            print("\n" + "="*60)
            print("HELP & TIPS".center(60))
            print("="*60)
            print("\nVisualization Tips:")
            print("- Use enhanced phase portraits to see both phase and modulus")
            print("- PyVista windows are interactive: rotate, zoom, and pan")
            print("- Press 'q' to close PyVista windows")
            print("- Press 's' in PyVista to save screenshots")
            print("\nFunction Categories:")
            print("- Polynomials: Show zeros as phase vortices")
            print("- Rational: Show both zeros and poles")
            print("- Transcendental: Often periodic or have branch cuts")
            print("- Essential singularities: Chaotic behavior near singularity")
            print("\nSTL Export Tips:")
            print("- Use annular domains to exclude singularities")
            print("- Higher resolution = smoother surfaces but larger files")
            print("- Arctan scaling usually gives best results")
            print("- 80mm size works well for ornaments")
            continue
        
        if main_choice == "5":
            # Batch processing
            batch_processing()
            continue
        
        # For visualization options, choose function and colormap
        print_menu("Select a Complex Function", FUNCTIONS)
        func_choice = get_choice("Select function", FUNCTIONS.keys())
        selected_func = FUNCTIONS[func_choice]
        
        print_menu("Select a Color Scheme", COLOR_SCHEMES)
        color_choice = get_choice("Select color scheme", COLOR_SCHEMES.keys())
        selected_cmap = COLOR_SCHEMES[color_choice]["cmap"]
        
        # Handle different visualization types
        if main_choice == "1":
            # 2D visualization
            print_menu("Select a Domain", DOMAINS)
            domain_choice = get_choice("Select domain", DOMAINS.keys())
            
            if domain_choice == "7":
                selected_domain = create_custom_domain()
            else:
                selected_domain = DOMAINS[domain_choice]["domain"]
            
            # Resolution for 2D
            res = int(input("\nResolution (100-800, default 300): ") or "300")
            
            visualization_2d(selected_func, selected_cmap, selected_domain, res)
        
        elif main_choice == "2":
            # 3D landscape
            print_menu("Select a Domain", DOMAINS)
            domain_choice = get_choice("Select domain", DOMAINS.keys())
            
            if domain_choice == "7":
                selected_domain = create_custom_domain()
            else:
                selected_domain = DOMAINS[domain_choice]["domain"]
            
            # Resolution for 3D
            res = int(input("\nResolution (50-500, default 200): ") or "200")
            
            visualization_3d_landscape(selected_func, selected_cmap, selected_domain, res)
        
        elif main_choice == "3":
            # Riemann sphere
            res = int(input("\nResolution (50-300, default 150): ") or "150")
            visualization_riemann_sphere(selected_func, selected_cmap, res)
        
        elif main_choice == "4":
            # STL export
            stl_export(selected_func, selected_cmap)
        
        # Continue prompt
        continue_choice = input("\nReturn to main menu? (y/n): ").strip().lower()
        if continue_choice != 'y':
            print("\nThank you for using Complexplorer!")
            break


if __name__ == "__main__":
    # Ensure interactive matplotlib backend
    cp.ensure_interactive_plots()
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please report this issue if it persists.")