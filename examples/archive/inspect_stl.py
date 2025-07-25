#!/usr/bin/env python3
"""
Inspect STL files to verify spherical healing results.
"""

import pyvista as pv
import sys

def inspect_stl(filename: str):
    """Load and display statistics for an STL file."""
    print(f"\nInspecting: {filename}")
    print("-" * 50)
    
    mesh = pv.read(filename)
    
    print(f"Vertices: {mesh.n_points}")
    print(f"Faces: {mesh.n_cells}")
    
    # Check bounds
    bounds = mesh.bounds
    print(f"Bounds:")
    print(f"  X: [{bounds[0]:.3f}, {bounds[1]:.3f}]")
    print(f"  Y: [{bounds[2]:.3f}, {bounds[3]:.3f}]")
    print(f"  Z: [{bounds[4]:.3f}, {bounds[5]:.3f}]")
    
    # Check if bottom is flat
    z_min = bounds[4]
    bottom_points = mesh.points[abs(mesh.points[:, 2] - z_min) < 0.001]
    if len(bottom_points) > 0:
        z_var = bottom_points[:, 2].var()
        print(f"Bottom flatness: {z_var:.2e}")
    
    # Check for boundary edges
    edges = mesh.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False
    )
    print(f"Boundary edges: {edges.n_cells}")
    
    # Non-manifold edges
    nm_edges = mesh.extract_feature_edges(
        boundary_edges=False,
        feature_edges=False, 
        manifold_edges=False,
        non_manifold_edges=True
    )
    print(f"Non-manifold edges: {nm_edges.n_cells}")
    
    return mesh


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for filename in sys.argv[1:]:
            inspect_stl(filename)
    else:
        # Default files to check
        files = [
            "pole_test_top.stl",
            "pole_test_bottom.stl",
            "z_squared_test_top.stl",
            "z_squared_test_bottom.stl"
        ]
        
        for f in files:
            try:
                inspect_stl(f)
            except FileNotFoundError:
                print(f"\nFile not found: {f}")