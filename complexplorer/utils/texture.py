"""Texture computation utilities for physical STL exports.

This module provides functions to convert colormap boundaries into
physical texture (ridges, grooves, or binary height variations) for
3D printing.
"""

from typing import Tuple, Optional
import numpy as np
from ..core.colormap import Colormap, Chessboard, PolarChessboard, LogRings
from .texture_direct import compute_texture_direct


def compute_texture_from_colormap(
    f_values: np.ndarray,
    cmap: Colormap,
    mode: str = 'ridges',
    sharpness: float = 1.0,
    mesh_shape: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """Convert colormap boundaries to physical texture displacement.
    
    Uses direct computation based on colormap patterns rather than
    gradient detection for more reliable boundary detection.
    
    Parameters
    ----------
    f_values : ndarray
        Complex function values at mesh points (1D array).
    cmap : Colormap
        Colormap instance to extract boundaries from.
    mode : str
        Texture mode: 'ridges', 'grooves', or 'binary'.
    sharpness : float
        Edge detection sensitivity (0-1). Currently unused but kept
        for API compatibility.
    mesh_shape : tuple, optional
        Shape of the mesh grid (n_theta, n_phi). Currently unused
        but kept for API compatibility.
        
    Returns
    -------
    ndarray
        Displacement values in range [-1, 1] to be scaled by texture_height.
    """
    # Ensure f_values is 1D
    f_values = np.asarray(f_values).ravel()
    
    # Use direct texture computation
    return compute_texture_direct(f_values, cmap, mode)


def compute_sphere_gradient(
    values: np.ndarray,
    mesh_shape: Tuple[int, int]
) -> np.ndarray:
    """Compute gradient magnitude on spherical mesh.
    
    Parameters
    ----------
    values : ndarray
        Scalar values at mesh points (1D array).
    mesh_shape : tuple
        Shape of the mesh grid (n_theta, n_phi).
        
    Returns
    -------
    ndarray
        Gradient magnitude at each point (1D array).
    """
    n_theta, n_phi = mesh_shape
    
    # Reshape to 2D grid
    # Note: mesh is created with meshgrid(theta, phi), so shape is (n_phi, n_theta)
    values_2d = values.reshape(n_phi, n_theta)
    
    # Compute gradients with proper boundary handling
    # Note: np.gradient handles boundaries automatically
    # gradient returns (axis0, axis1) = (phi, theta) gradients
    grad_phi, grad_theta = np.gradient(values_2d)
    
    # Account for spherical metric
    # Near poles (theta near 0 or pi), phi gradients need scaling
    theta = np.linspace(0.01, np.pi - 0.01, n_theta)  # Avoid exact poles
    # Create grid matching the shape (n_phi, n_theta)
    theta_grid = theta[np.newaxis, :]
    sin_theta = np.sin(theta_grid)
    
    # Avoid division by zero at poles
    sin_theta = np.maximum(sin_theta, 0.1)  # More conservative limit
    
    # Scale phi gradient by 1/sin(theta)
    grad_phi_scaled = grad_phi / sin_theta
    
    # Suppress extreme values near poles
    grad_phi_scaled = np.minimum(grad_phi_scaled, 10.0)
    
    # Compute magnitude
    gradient_magnitude = np.sqrt(grad_theta**2 + grad_phi_scaled**2)
    
    return gradient_magnitude.ravel()


def compute_hsv_gradient(
    hsv: np.ndarray,
    mesh_shape: Tuple[int, int]
) -> np.ndarray:
    """Compute perceptually weighted gradient in HSV space.
    
    Parameters
    ----------
    hsv : ndarray
        HSV values at mesh points (N x 3 array).
    mesh_shape : tuple
        Shape of the mesh grid (n_theta, n_phi).
        
    Returns
    -------
    ndarray
        Gradient magnitude at each point (1D array).
    """
    # Extract channels
    H = hsv[:, 0]  # Hue [0, 1]
    S = hsv[:, 1]  # Saturation [0, 1]
    V = hsv[:, 2]  # Value [0, 1]
    
    # Compute gradients for each channel
    grad_h = compute_sphere_gradient(H, mesh_shape)
    grad_s = compute_sphere_gradient(S, mesh_shape)
    grad_v = compute_sphere_gradient(V, mesh_shape)
    
    # Handle hue wraparound (0-1 boundary)
    # If gradient is large, check if it's due to wraparound
    n_theta, n_phi = mesh_shape
    h_2d = H.reshape(n_phi, n_theta)
    h_diff_phi = np.diff(h_2d, axis=0, prepend=h_2d[-1:, :])
    h_diff_theta = np.diff(h_2d, axis=1, prepend=h_2d[:, -1:])
    
    # Detect wraparound (large jumps)
    wrap_phi = np.abs(h_diff_phi) > 0.5
    wrap_theta = np.abs(h_diff_theta) > 0.5
    
    # Correct gradient where wraparound occurs
    grad_h_2d = grad_h.reshape(n_phi, n_theta)
    # Reduce gradient at wraparound locations
    # Handle phi wraparound
    if wrap_phi.shape[0] > 1:
        phi_mask = wrap_phi[:-1, :] | wrap_phi[1:, :]
        if phi_mask.shape[0] == grad_h_2d.shape[0]:
            grad_h_2d[phi_mask] *= 0.1
    # Handle theta wraparound
    if wrap_theta.shape[1] > 1:
        theta_cols = np.where(wrap_theta[:, :-1] | wrap_theta[:, 1:])[1]
        if len(theta_cols) > 0:
            grad_h_2d[:, theta_cols] *= 0.1
    grad_h = grad_h_2d.ravel()
    
    # Weighted combination
    # Hue changes are most perceptually important
    gradient_magnitude = np.sqrt(
        grad_h**2 +           # Hue changes (most important)
        0.3 * grad_s**2 +     # Saturation changes
        0.5 * grad_v**2       # Value changes
    )
    
    return gradient_magnitude


def quantize_regions(
    hsv: np.ndarray,
    edges: np.ndarray,
    mesh_shape: Tuple[int, int]
) -> np.ndarray:
    """Quantize regions between edges for binary texture mode.
    
    Parameters
    ----------
    hsv : ndarray
        HSV values at mesh points (N x 3 array).
    edges : ndarray
        Boolean array marking edge locations.
    mesh_shape : tuple
        Shape of the mesh grid (n_theta, n_phi).
        
    Returns
    -------
    ndarray
        Binary displacement values (-1 or 1).
    """
    # Use connected components to identify regions
    # For now, use a simple alternating pattern based on hue
    H = hsv[:, 0]
    
    # Quantize hue into levels
    n_levels = 12
    hue_level = np.floor(H * n_levels).astype(int)
    
    # Alternate between raised and lowered
    displacement = np.where(hue_level % 2 == 0, 1.0, -1.0)
    
    # Edges themselves can be neutral
    displacement[edges] = 0.0
    
    return displacement


def apply_texture_to_mesh(
    mesh: 'pv.PolyData',
    f_values: np.ndarray,
    cmap: Colormap,
    texture_height: float,
    texture_mode: str,
    texture_sharpness: float,
    texture_preview_scale: float,
    mesh_shape: Optional[Tuple[int, int]] = None
) -> 'pv.PolyData':
    """Apply texture displacement to a mesh.
    
    This function modifies the mesh in-place by displacing vertices
    along their normals based on colormap boundaries.
    
    Parameters
    ----------
    mesh : pv.PolyData
        The mesh to apply texture to. Must have computed normals.
    f_values : ndarray
        Complex function values at mesh points.
    cmap : Colormap
        Colormap for boundary detection.
    texture_height : float
        Physical texture height as fraction of sphere radius.
    texture_mode : str
        Texture mode: 'ridges', 'grooves', or 'binary'.
    texture_sharpness : float
        Edge detection sensitivity.
    texture_preview_scale : float
        Preview scale factor.
    mesh_shape : tuple, optional
        Shape of the mesh grid (n_theta, n_phi). If None or if the
        mesh has been filtered by a domain, gradient-based textures
        will be disabled.
        
    Returns
    -------
    pv.PolyData
        The mesh with texture applied.
    """
    if texture_height == 0:
        return mesh
    
    # Direct texture computation works for all mesh types
    displacement = compute_texture_from_colormap(
        f_values, cmap, texture_mode, texture_sharpness, mesh_shape
    )
    
    # Scale for preview
    scaled_displacement = displacement * texture_height * texture_preview_scale
    
    # Apply displacement along normals
    if not hasattr(mesh, 'point_normals') or mesh.point_normals is None:
        mesh.compute_normals(point_normals=True, inplace=True)
    
    normals = mesh.point_normals
    mesh.points += normals * scaled_displacement[:, np.newaxis]
    
    # Store actual (non-preview-scaled) displacement for STL export
    mesh.point_data['texture_displacement'] = displacement * texture_height
    # Store metadata as field data (single values)
    mesh.field_data['texture_mode'] = np.array([0])  # Will store mode separately
    mesh.field_data['texture_height'] = np.array([texture_height])
    
    return mesh