#!/usr/bin/env python3
"""
Interactive Complexplorer Demo with PyVista

This script provides an interactive menu to explore complex functions
with different visualization types and color schemes.
"""

import complexplorer as cp
import numpy as np
import pyvista as pv


# Pre-defined complex functions
FUNCTIONS = {
    "1": {
        "name": "Möbius transformation",
        "func": lambda z: (z - 1) / (z + 1),
        "description": "f(z) = (z - 1) / (z + 1)"
    },
    "2": {
        "name": "Quadratic",
        "func": lambda z: z**2,
        "description": "f(z) = z²"
    },
    "3": {
        "name": "Cubic",
        "func": lambda z: z**3 - 1,
        "description": "f(z) = z³ - 1"
    },
    "4": {
        "name": "Rational function",
        "func": lambda z: (z**2 - 1) / (z**2 + 1),
        "description": "f(z) = (z² - 1) / (z² + 1)"
    },
    "5": {
        "name": "Sine",
        "func": lambda z: np.sin(z),
        "description": "f(z) = sin(z)"
    },
    "6": {
        "name": "Exponential",
        "func": lambda z: np.exp(z),
        "description": "f(z) = e^z"
    },
    "7": {
        "name": "Complex logarithm",
        "func": lambda z: np.log(z + 0.1j),  # Small offset to avoid branch cut at origin
        "description": "f(z) = log(z + 0.1i)"
    },
    "8": {
        "name": "Reciprocal",
        "func": lambda z: 1 / z,
        "description": "f(z) = 1/z"
    }
}

# Pre-defined color schemes
COLOR_SCHEMES = {
    "1": {
        "name": "Basic phase",
        "cmap": cp.Phase()
    },
    "2": {
        "name": "Phase (6 colors)",
        "cmap": cp.Phase(n_phi=6)
    },
    "3": {
        "name": "Phase (12 colors)",
        "cmap": cp.Phase(n_phi=12)
    },
    "4": {
        "name": "Enhanced phase (6 colors, modulus)",
        "cmap": cp.Phase(n_phi=6, r_linear_step=0.5, v_base=0.4)
    },
    "5": {
        "name": "Enhanced phase (12 colors, modulus)",
        "cmap": cp.Phase(n_phi=12, r_linear_step=0.5, v_base=0.4)
    },
    "6": {
        "name": "Chessboard",
        "cmap": cp.Chessboard(spacing=0.5)
    },
    "7": {
        "name": "Polar chessboard",
        "cmap": cp.PolarChessboard(n_phi=12, spacing=0.5)
    },
    "8": {
        "name": "Logarithmic rings",
        "cmap": cp.LogRings(log_spacing=0.3)
    }
}

# Pre-defined domains
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
    }
}


def print_menu(title, options):
    """Print a formatted menu."""
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print('='*50)
    for key, value in options.items():
        if 'description' in value:
            print(f"{key}. {value['name']}: {value['description']}")
        else:
            print(f"{key}. {value['name']}")
    print('='*50)


def get_choice(prompt, valid_choices):
    """Get a valid choice from the user."""
    while True:
        choice = input(f"\n{prompt}: ").strip()
        if choice in valid_choices:
            return choice
        print(f"Invalid choice. Please enter one of: {', '.join(valid_choices)}")


def main():
    """Main interactive demo."""
    print("\n" + "="*60)
    print("COMPLEXPLORER INTERACTIVE DEMO".center(60))
    print("PyVista 3D Visualizations with Orientation Axes".center(60))
    print("="*60)
    
    # Configure PyVista for better quality
    pv.global_theme.multi_samples = 4
    pv.global_theme.smooth_shading = True
    
    while True:
        # Choose visualization type
        print("\nVisualization Types:")
        print("1. 3D Landscape")
        print("2. Domain/Codomain Pair")
        print("3. Riemann Sphere")
        print("4. Quit")
        
        viz_type = get_choice("Select visualization type (1-4)", ["1", "2", "3", "4"])
        
        if viz_type == "4":
            print("\nThank you for using Complexplorer!")
            break
        
        # Choose function
        print_menu("Select a Complex Function", FUNCTIONS)
        func_choice = get_choice("Select function (1-8)", FUNCTIONS.keys())
        selected_func = FUNCTIONS[func_choice]
        
        # Choose color scheme
        print_menu("Select a Color Scheme", COLOR_SCHEMES)
        color_choice = get_choice("Select color scheme (1-8)", COLOR_SCHEMES.keys())
        selected_cmap = COLOR_SCHEMES[color_choice]["cmap"]
        
        # Choose domain (not for Riemann sphere)
        if viz_type != "3":
            print_menu("Select a Domain", DOMAINS)
            domain_choice = get_choice("Select domain (1-5)", DOMAINS.keys())
            selected_domain = DOMAINS[domain_choice]["domain"]
        
        # Set resolution
        print("\nResolution options:")
        print("1. Low (100 points) - Fast")
        print("2. Medium (250 points) - Balanced")
        print("3. High (500 points) - Quality")
        print("4. Very High (800 points) - Slow")
        
        res_choice = get_choice("Select resolution (1-4)", ["1", "2", "3", "4"])
        resolution = {"1": 100, "2": 250, "3": 500, "4": 800}[res_choice]

        # Set z_max for landscape plots
        z_max = 10  # Default value
        if viz_type in ["1", "2"]:  # Landscape or pair plot
            print("\nZ-axis scaling options:")
            print("1. Standard (max height = 10)")
            print("2. Extended (max height = 20)")
            print("3. Full range (no limit)")
            print("4. Custom value")
            
            z_choice = get_choice("Select Z-axis scaling (1-4)", ["1", "2", "3", "4"])
            
            if z_choice == "1":
                z_max = 10
            elif z_choice == "2":
                z_max = 20
            elif z_choice == "3":
                z_max = None
            elif z_choice == "4":
                while True:
                    try:
                        custom_z = input("Enter maximum Z value (e.g., 5, 15, 30): ").strip()
                        z_max = float(custom_z)
                        if z_max > 0:
                            break
                        else:
                            print("Please enter a positive number.")
                    except ValueError:
                        print("Invalid input. Please enter a number.")
        
        # Create visualization
        print(f"\nCreating {selected_func['name']} visualization...")
        print("(Close the window to return to menu)")
        
        try:
            if viz_type == "1":
                # 3D Landscape
                cp.plot_landscape_pv(
                    selected_domain,
                    selected_func["func"],
                    n=resolution,
                    cmap=selected_cmap,
                    z_max=z_max,  # Control Z scaling
                    show_orientation=True,
                    interactive=True,
                    title=f"{selected_func['name']}: {selected_func['description']}",
                    camera_position='iso'
                )
            
            elif viz_type == "2":
                # Domain/Codomain Pair
                cp.pair_plot_landscape_pv(
                    selected_domain,
                    selected_func["func"],
                    n=resolution,
                    cmap=selected_cmap,
                    z_max=z_max,  # Control Z scaling
                    show_orientation=True,
                    interactive=True,
                    title=f"{selected_func['name']}: {selected_func['description']}",
                    camera_position='iso'
                )
            
            elif viz_type == "3":
                # Riemann Sphere
                print("\nRiemann Sphere options:")
                print("1. Traditional (constant radius)")
                print("2. Modulus scaling (arctan)")
                
                sphere_choice = get_choice("Select sphere type (1-2)", ["1", "2"])
                
                if sphere_choice == "1":
                    scaling = 'constant'
                    scaling_params = {'radius': 1.0}
                else:
                    scaling = 'arctan'
                    scaling_params = {'r_min': 0.2, 'r_max': 1.0}
                
                cp.riemann_pv(
                    selected_func["func"],
                    n_theta=resolution,
                    n_phi=resolution,
                    cmap=selected_cmap,
                    scaling=scaling,
                    scaling_params=scaling_params,
                    show_orientation=True,
                    show_grid=True,
                    interactive=True,
                    title=f"{selected_func['name']} on Riemann Sphere",
                    camera_position='iso',
                    high_quality=True,
                    anti_aliasing=True
                )
                
        except Exception as e:
            print(f"\nError creating visualization: {e}")
            print("This might happen with certain function/domain combinations.")
            print("Try a different selection.")
        
        # Ask if user wants to continue
        continue_choice = input("\nCreate another visualization? (y/n): ").strip().lower()
        if continue_choice != 'y':
            print("\nThank you for using Complexplorer!")
            break


if __name__ == "__main__":
    main()