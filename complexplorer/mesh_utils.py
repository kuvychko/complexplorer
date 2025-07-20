"""
Mesh utilities for complex function visualization on the Riemann sphere.

This module provides two sphere meshing approaches:

1. **Rectangular/UV Meshing** (default): Uses latitude-longitude grid
   - Better rendering quality in PyVista due to structured grid format
   - Smoother appearance with standard shading
   - May have slight distortion near poles
   - Recommended for most visualizations

2. **Icosahedral Meshing**: Uses triangular subdivision
   - Uniform point distribution without pole singularities
   - Theoretically superior but may appear less smooth in PyVista
   - Better for mathematical accuracy
   - Use when uniform sampling is critical

Note: While icosahedral meshing is mathematically superior, rectangular meshing
often produces better visual results in PyVista due to how the rendering engine
handles structured grids vs triangular meshes.
"""

import numpy as np
import pyvista as pv
from typing import Optional, Tuple, Dict, List, Set, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from .domain import Domain


@dataclass
class IcosphereData:
    """Data structure for icosahedral sphere mesh."""
    vertices: np.ndarray  # (N, 3) coordinates
    faces: np.ndarray     # (M, 3) vertex indices
    edges: Set[Tuple[int, int]] = field(default_factory=set)
    vertex_map: Dict[Tuple[int, int], int] = field(default_factory=dict)
    
    # Riemann sphere specific
    complex_coords: Optional[np.ndarray] = None
    north_pole_index: Optional[int] = None
    south_pole_index: Optional[int] = None


class IcosphereGenerator:
    """
    Generate icosahedral sphere mesh with uniform vertex distribution.
    
    This generator creates a sphere mesh starting from a regular icosahedron
    and recursively subdividing triangular faces to achieve desired resolution.
    
    Parameters
    ----------
    radius : float, default=1.0
        Radius of the sphere.
    subdivisions : int, default=4
        Number of subdivision levels. Each level quadruples the number of faces.
        Level 0: 20 faces, Level n: 20 * 4^n faces.
    """
    
    def __init__(self, radius: float = 1.0, subdivisions: int = 4):
        self.radius = radius
        self.subdivisions = max(0, min(subdivisions, 8))  # Limit to reasonable range

        self._data = None
        
    def generate(self) -> pv.PolyData:
        """
        Generate the icosahedral sphere mesh.
        
        Returns
        -------
        mesh : pv.PolyData
            PyVista mesh object representing the sphere.
        """
        self._create_base_icosahedron()
        
        # Perform subdivisions
        for _ in range(self.subdivisions):
            self._subdivide()
        
        # Create PyVista mesh
        vertices = self._data.vertices * self.radius
        
        # Convert faces to PyVista format
        n_faces = len(self._data.faces)
        faces_pv = np.hstack([
            np.full((n_faces, 1), 3),  # 3 vertices per face
            self._data.faces
        ]).ravel()
        
        mesh = pv.PolyData(vertices, faces_pv)
        return mesh
    
    def get_data(self) -> IcosphereData:
        """Get the raw mesh data structure."""
        if self._data is None:
            self.generate()
        return self._data
    
    def _create_base_icosahedron(self):
        """Create the base icosahedron with 12 vertices and 20 faces."""
        # Golden ratio
        phi = (1 + np.sqrt(5)) / 2
        
        # Create vertices
        vertices = []
        
        # (±1, ±φ, 0)
        for x in [-1, 1]:
            for y in [-phi, phi]:
                vertices.append([x, y, 0])
        
        # (0, ±1, ±φ)
        for y in [-1, 1]:
            for z in [-phi, phi]:
                vertices.append([0, y, z])
        
        # (±φ, 0, ±1)
        for x in [-phi, phi]:
            for z in [-1, 1]:
                vertices.append([x, 0, z])
        
        vertices = np.array(vertices, dtype=np.float64)
        
        # Normalize to unit sphere
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        
        # Define faces (triangles) - correct winding order for outward normals
        faces = np.array([
            # 5 faces around vertex 0
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            # 5 adjacent faces
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            # 5 faces around vertex 3
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            # 5 adjacent faces
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ], dtype=np.int32)
        
        # Initialize data structure
        self._data = IcosphereData(vertices=vertices, faces=faces)
        
        # Build edge set
        for face in faces:
            for i in range(3):
                v1, v2 = face[i], face[(i + 1) % 3]
                edge = (min(v1, v2), max(v1, v2))
                self._data.edges.add(edge)
    
    def _subdivide(self):
        """Perform one level of subdivision."""
        new_vertices = list(self._data.vertices)
        new_faces = []
        
        # Process each face
        for face in self._data.faces:
            v0, v1, v2 = face
            
            # Get or create midpoint vertices
            m01 = self._get_or_create_midpoint(v0, v1, new_vertices)
            m12 = self._get_or_create_midpoint(v1, v2, new_vertices)
            m20 = self._get_or_create_midpoint(v2, v0, new_vertices)
            
            # Create 4 new triangles
            new_faces.extend([
                [v0, m01, m20],
                [v1, m12, m01],
                [v2, m20, m12],
                [m01, m12, m20]
            ])
        
        # Update data
        self._data.vertices = np.array(new_vertices)
        self._data.faces = np.array(new_faces)
        
        # Rebuild edge set
        self._data.edges.clear()
        for face in self._data.faces:
            for i in range(3):
                v1, v2 = face[i], face[(i + 1) % 3]
                edge = (min(v1, v2), max(v1, v2))
                self._data.edges.add(edge)
    
    def _get_or_create_midpoint(self, v1: int, v2: int, vertices: List) -> int:
        """Get or create midpoint between two vertices."""
        edge = (min(v1, v2), max(v1, v2))
        
        if edge in self._data.vertex_map:
            return self._data.vertex_map[edge]
        
        # Create new midpoint
        mid = (self._data.vertices[v1] + self._data.vertices[v2]) / 2
        # Project to unit sphere
        mid = mid / np.linalg.norm(mid)
        
        # Add to vertices
        mid_idx = len(vertices)
        vertices.append(mid)
        self._data.vertex_map[edge] = mid_idx
        
        return mid_idx


class RectangularSphereGenerator:
    """
    Generate rectangular (latitude-longitude) sphere mesh.
    
    This generator creates a sphere mesh using a regular latitude-longitude grid,
    similar to UV sphere mapping. While this approach has slight distortion near
    poles, it often produces better visual results in PyVista due to the structured
    grid format.
    
    Parameters
    ----------
    radius : float, default=1.0
        Radius of the sphere.
    n_theta : int, default=100
        Number of latitude divisions (from pole to pole).
    n_phi : int, default=100
        Number of longitude divisions (around equator).
    avoid_poles : bool, default=True
        If True, slightly offset from exact poles to avoid singularities.
    domain : Domain, optional
        If provided, only generate mesh points whose stereographic projections
        fall within this domain. Helps avoid numerical instability at extreme values.
    """
    
    def __init__(self, radius: float = 1.0, n_theta: int = 100, n_phi: int = 100,
                 avoid_poles: bool = True, domain: Optional['Domain'] = None):
        self.radius = radius
        self.n_theta = n_theta
        self.n_phi = n_phi
        self.avoid_poles = avoid_poles
        self.domain = domain
        
    def generate(self) -> pv.StructuredGrid:
        """
        Generate the rectangular sphere mesh, optionally constrained by domain.
        
        Returns
        -------
        mesh : pv.StructuredGrid
            PyVista structured grid representing the sphere.
        """
        # Create angle arrays
        if self.avoid_poles:
            # Avoid exact poles to prevent singularities
            theta = np.linspace(0.01, np.pi - 0.01, self.n_theta)
        else:
            theta = np.linspace(0, np.pi, self.n_theta)
            
        phi = np.linspace(0, 2 * np.pi, self.n_phi)
        
        # Create meshgrid
        THETA, PHI = np.meshgrid(theta, phi)
        
        # Convert to Cartesian coordinates
        X = self.radius * np.sin(THETA) * np.cos(PHI)
        Y = self.radius * np.sin(THETA) * np.sin(PHI)
        Z = self.radius * np.cos(THETA)
        
        # If domain is specified, filter points
        if self.domain is not None:
            # Project points to complex plane
            w = stereographic_projection(X.ravel(), Y.ravel(), Z.ravel())
            
            # Check which points are in domain
            in_domain = self.domain.infunc(w)
            in_domain = in_domain.reshape(X.shape)
            
            # Mark out-of-domain points as NaN
            X = np.where(in_domain, X, np.nan)
            Y = np.where(in_domain, Y, np.nan)
            Z = np.where(in_domain, Z, np.nan)
        
        # Create structured grid
        grid = pv.StructuredGrid(X, Y, Z)
        
        return grid
    
    def generate_uv_sphere(self) -> pv.PolyData:
        """
        Generate using PyVista's built-in UV sphere.
        
        This uses PyVista's optimized sphere generation which may have
        better performance characteristics.
        
        Returns
        -------
        mesh : pv.PolyData
            PyVista sphere mesh.
        """
        # Adjust parameters if avoiding poles
        kwargs = {
            'radius': self.radius,
            'theta_resolution': self.n_theta,
            'phi_resolution': self.n_phi,
        }
        
        if self.avoid_poles:
            kwargs['start_theta'] = 0.1
            kwargs['end_theta'] = 359.9
            
        return pv.Sphere(**kwargs)


def stereographic_projection(x: np.ndarray, y: np.ndarray, z: np.ndarray, 
                           from_north: bool = True) -> np.ndarray:
    """
    Project points from sphere to complex plane via stereographic projection.
    
    Parameters
    ----------
    x, y, z : np.ndarray
        Coordinates on the sphere.
    from_north : bool, default=True
        If True, project from north pole (0,0,1) with north pole mapping to infinity.
        If False, project from south pole (0,0,-1).
    
    Returns
    -------
    w : np.ndarray
        Complex coordinates on the plane.
    """
    if from_north:
        # Avoid division by zero at north pole
        mask = z < 0.999
        w = np.zeros(z.shape, dtype=complex)
        w[mask] = (x[mask] + 1j * y[mask]) / (1 - z[mask])
        w[~mask] = np.inf + 1j * np.inf  # North pole maps to infinity
    else:
        # Avoid division by zero at south pole
        mask = z > -0.999
        w = np.zeros(z.shape, dtype=complex)
        w[mask] = (x[mask] + 1j * y[mask]) / (1 + z[mask])
        w[~mask] = np.inf + 1j * np.inf  # South pole maps to infinity
    
    return w


def inverse_stereographic(w: np.ndarray, to_north: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project points from complex plane to sphere via inverse stereographic projection.
    
    Parameters
    ----------
    w : np.ndarray
        Complex coordinates on the plane.
    to_north : bool, default=True
        If True, use projection with north pole at infinity.
        If False, use projection with south pole at infinity.
    
    Returns
    -------
    x, y, z : np.ndarray
        Coordinates on the sphere.
    """
    u = np.real(w)
    v = np.imag(w)
    u2v2 = u**2 + v**2
    
    # Handle infinity
    finite_mask = np.isfinite(w)
    x = np.zeros_like(u)
    y = np.zeros_like(u)
    z = np.zeros_like(u)
    
    if to_north:
        # Finite points
        x[finite_mask] = 2 * u[finite_mask] / (1 + u2v2[finite_mask])
        y[finite_mask] = 2 * v[finite_mask] / (1 + u2v2[finite_mask])
        z[finite_mask] = (u2v2[finite_mask] - 1) / (1 + u2v2[finite_mask])
        # Infinity maps to north pole
        x[~finite_mask] = 0
        y[~finite_mask] = 0
        z[~finite_mask] = 1
    else:
        # Finite points
        x[finite_mask] = 2 * u[finite_mask] / (1 + u2v2[finite_mask])
        y[finite_mask] = 2 * v[finite_mask] / (1 + u2v2[finite_mask])
        z[finite_mask] = (1 - u2v2[finite_mask]) / (1 + u2v2[finite_mask])
        # Infinity maps to south pole
        x[~finite_mask] = 0
        y[~finite_mask] = 0
        z[~finite_mask] = -1
    
    return x, y, z


class ModulusScaling:
    """Scaling functions for mapping complex modulus to sphere radius."""
    
    @staticmethod
    def constant(modulus: np.ndarray, radius: float = 1.0) -> np.ndarray:
        """
        Constant radius - standard Riemann sphere visualization.
        
        This is the default for traditional Riemann sphere where all points
        lie on a sphere of fixed radius, with only color encoding the function.
        """
        return np.full_like(modulus, radius, dtype=float)
    
    @staticmethod
    def arctan(modulus: np.ndarray, r_min: float = 0.2, r_max: float = 1.0) -> np.ndarray:
        """
        Arctan scaling to map [0, ∞) to [r_min, r_max].
        
        This provides smooth compression of large values while preserving
        detail for small to moderate moduli. Good for visualizing functions
        with poles and zeros.
        """
        return r_min + (r_max - r_min) * (2 / np.pi * np.arctan(modulus))
    
    @staticmethod
    def logarithmic(modulus: np.ndarray, base: float = np.e, 
                   r_min: float = 0.2, r_max: float = 1.0) -> np.ndarray:
        """
        Logarithmic scaling for large dynamic range.
        
        Useful for functions with exponential growth or when you need to
        see both very small and very large values clearly.
        """
        with np.errstate(divide='ignore'):
            log_mod = np.log(modulus) / np.log(base)
        # Map log(modulus) to [r_min, r_max] using sigmoid
        scaled = 1 / (1 + np.exp(-log_mod))
        return r_min + (r_max - r_min) * scaled
    
    @staticmethod
    def linear_clamp(modulus: np.ndarray, m_max: float = 10, 
                    r_min: float = 0.2, r_max: float = 1.0) -> np.ndarray:
        """
        Linear scaling with saturation at m_max.
        
        Simple linear mapping that clips values above m_max. Good for
        focusing on a specific range of modulus values.
        """
        scaled = np.clip(modulus / m_max, 0, 1)
        return r_min + (r_max - r_min) * scaled
    
    @staticmethod
    def custom(modulus: np.ndarray, scaling_func: callable, 
              r_min: float = 0.2, r_max: float = 1.0) -> np.ndarray:
        """
        Apply custom scaling function.
        
        The scaling_func should map modulus values to [0, 1] range.
        """
        scaled = scaling_func(modulus)
        # Ensure output is in [0, 1] range
        scaled = np.clip(scaled, 0, 1)
        return r_min + (r_max - r_min) * scaled