#!/usr/bin/env python3
"""
STL Export Demo - Generate 3D-printable mathematical ornaments.

This script demonstrates the STL export functionality of complexplorer.
It generates complete watertight meshes that can be sliced at any angle
in your 3D printing software.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from complexplorer.cmap import Phase
from complexplorer.stl_export import OrnamentGenerator
from complexplorer import Disk, Annulus


def generate_examples():
    """Generate a collection of example ornaments."""
    
    print("Complexplorer STL Export Demo")
    print("=" * 60)
    print("Generating complete watertight meshes for 3D printing.")
    print("You can slice these at any angle in your slicer software!\n")
    
    # Example 1: Simple rational function
    print("1. Möbius transformation")
    print("-" * 30)
    func1 = lambda z: (z - 1) / (z + 1)
    
    gen1 = OrnamentGenerator(
        func=func1,
        resolution=120,
        scaling='arctan',
        scaling_params={'r_min': 0.3, 'r_max': 1.0},
        cmap=Phase(12)
    )
    
    file1 = gen1.generate_ornament(
        output_file='mobius_ornament.stl',
        size_mm=80,
        smooth=True,
        verbose=False
    )
    print(f"✓ Generated: {file1}\n")
    
    # Example 2: Function with pole at origin
    print("2. Simple pole at origin")
    print("-" * 30)
    func2 = lambda z: 1 / z
    
    # Use domain to avoid numerical issues at origin
    domain2 = Annulus(radius_inner=0.1, radius_outer=5)
    
    gen2 = OrnamentGenerator(
        func=func2,
        resolution=100,
        scaling='logarithmic',
        scaling_params={'base': 2, 'r_min': 0.3, 'r_max': 1.2},
        cmap=Phase(12),
        domain=domain2
    )
    
    file2 = gen2.generate_ornament(
        output_file='pole_ornament.stl',
        size_mm=80,
        smooth=True,
        verbose=False
    )
    print(f"✓ Generated: {file2}\n")
    
    # Example 3: Polynomial
    print("3. Cubic polynomial")
    print("-" * 30)
    func3 = lambda z: z**3 - 1
    
    gen3 = OrnamentGenerator(
        func=func3,
        resolution=150,
        scaling='linear_clamp',
        scaling_params={'m_max': 5, 'r_min': 0.4, 'r_max': 1.0},
        cmap=Phase(12)
    )
    
    file3 = gen3.generate_ornament(
        output_file='cubic_ornament.stl',
        size_mm=80,
        smooth=True,
        verbose=False
    )
    print(f"✓ Generated: {file3}\n")
    
    # Example 4: Transcendental function
    print("4. Sine function")
    print("-" * 30)
    func4 = lambda z: np.sin(z)
    
    # Restrict domain for better visualization
    domain4 = Disk(radius=3)
    
    gen4 = OrnamentGenerator(
        func=func4,
        resolution=120,
        scaling='arctan',
        scaling_params={'r_min': 0.3, 'r_max': 1.0},
        cmap=Phase(12),
        domain=domain4
    )
    
    file4 = gen4.generate_ornament(
        output_file='sine_ornament.stl',
        size_mm=80,
        smooth=True,
        verbose=False
    )
    print(f"✓ Generated: {file4}\n")
    
    # Summary
    print("=" * 60)
    print("All ornaments generated successfully!")
    print("\nNext steps:")
    print("1. Import STL files into your slicer (PrusaSlicer, Cura, etc.)")
    print("2. Orient and position as desired")
    print("3. Use the cut tool to slice at any angle")
    print("4. Print with 20-30% infill, no supports needed")
    print("\nTip: Try different cutting angles for unique ornament designs!")


def show_advanced_usage():
    """Demonstrate advanced features."""
    
    print("\n\nAdvanced Example: Custom Parameters")
    print("=" * 60)
    
    # Complex function with multiple singularities
    func = lambda z: (z**2 - 1) / (z**2 + z + 1)
    
    # Create generator with custom settings
    generator = OrnamentGenerator(
        func=func,
        resolution=200,  # High resolution for detail
        scaling='arctan',
        scaling_params={'r_min': 0.25, 'r_max': 1.1},
        cmap=Phase(n_phi=16)  # More phase divisions
    )
    
    # Generate with detailed output
    print("Generating high-resolution ornament with custom parameters...")
    filename = generator.generate_ornament(
        output_file='advanced_ornament.stl',
        size_mm=100,  # Larger size
        smooth=True,
        smooth_iterations=30,  # More smoothing
        verbose=True  # Show detailed progress
    )
    
    print(f"\nAdvanced ornament saved to: {filename}")


if __name__ == "__main__":
    generate_examples()
    show_advanced_usage()
    
    print("\n" + "=" * 60)
    print("Demo complete! Check the generated STL files.")