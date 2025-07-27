"""Complex domain classes for visualization.

This module provides classes for defining and manipulating complex domains
in the mathematical sense. A domain is represented by a function that
determines which points in the complex plane belong to the domain.

The module includes:
- Base Domain class with mesh generation and set operations
- Rectangle domain for rectangular regions
- Disk domain for circular regions  
- Annulus domain for ring-shaped regions
"""

from abc import ABC, abstractmethod
from math import ceil
from typing import Optional, Tuple, Callable, Union
import numpy as np
from ..utils.validation import ValidationError, validate_resolution


class Domain(ABC):
    """Abstract base class for complex domains.
    
    A domain represents a region in the complex plane defined by a
    membership function. The domain can generate meshes for visualization
    and supports set operations (union, intersection).
    
    Attributes
    ----------
    window_real : tuple[float, float]
        Left and right bounds of the viewing window.
    window_imag : tuple[float, float]
        Bottom and top bounds of the viewing window.
    square : bool
        Whether to constrain the viewing window to be square.
    """
    
    def __init__(self,
                 real: Tuple[float, float],
                 imag: Tuple[float, float],
                 square: bool = True):
        """Initialize domain with viewing window.
        
        Parameters
        ----------
        real : tuple[float, float]
            Left and right edges of the viewing window.
        imag : tuple[float, float]
            Bottom and top edges of the viewing window.
        square : bool, optional
            If True, constrain viewing window to be square.
        """
        # Validate inputs
        if real[0] == real[1]:
            raise ValidationError("Real bounds cannot be equal")
        if imag[0] == imag[1]:
            raise ValidationError("Imaginary bounds cannot be equal")
        
        self.square = square
        
        # Set up viewing window
        if square:
            real_d = abs(real[0] - real[1])
            imag_d = abs(imag[0] - imag[1])
            delta = (real_d - imag_d) / 2
            
            if delta >= 0:
                self.window_real = (min(real), max(real))
                self.window_imag = (min(imag) - delta, max(imag) + delta)
            else:
                delta *= -1
                self.window_real = (min(real) - delta, max(real) + delta)
                self.window_imag = (min(imag), max(imag))
        else:
            self.window_real = (min(real), max(real))
            self.window_imag = (min(imag), max(imag))
    
    @abstractmethod
    def contains(self, z: np.ndarray) -> np.ndarray:
        """Check if points belong to the domain.
        
        Parameters
        ----------
        z : np.ndarray
            Complex values to test.
            
        Returns
        -------
        np.ndarray
            Boolean array indicating membership.
        """
        pass
    
    def spacing(self, n: int) -> float:
        """Calculate mesh point spacing.
        
        Parameters
        ----------
        n : int
            Number of points along the longer axis.
            
        Returns
        -------
        float
            Distance between adjacent mesh points.
        """
        n = validate_resolution(n, min_val=2, max_val=10000)
        real_length = self.window_real[1] - self.window_real[0]
        imag_length = self.window_imag[1] - self.window_imag[0]
        return max(real_length, imag_length) / n
    
    def mesh(self, n: int = 500) -> np.ndarray:
        """Generate complex mesh for the viewing window.
        
        Parameters
        ----------
        n : int, optional
            Number of points along the longer axis.
            
        Returns
        -------
        np.ndarray
            2D array of complex values.
        """
        n = validate_resolution(n, min_val=2, max_val=10000)
        real_length = self.window_real[1] - self.window_real[0]
        imag_length = self.window_imag[1] - self.window_imag[0]
        spacing = self.spacing(n)
        
        real_axis = np.linspace(
            self.window_real[0], 
            self.window_real[1], 
            ceil(real_length / spacing)
        )
        imag_axis = np.linspace(
            self.window_imag[0], 
            self.window_imag[1], 
            ceil(imag_length / spacing)
        )
        
        x, y = np.meshgrid(real_axis, imag_axis)
        return x + 1j * y
    
    def inmask(self, n: int = 500) -> np.ndarray:
        """Generate boolean mask for points inside the domain.
        
        Parameters
        ----------
        n : int, optional
            Number of points along the longer axis.
            
        Returns
        -------
        np.ndarray
            Boolean mask (True for points inside).
        """
        z = self.mesh(n)
        return self.contains(z)
    
    def outmask(self, n: int = 500) -> np.ndarray:
        """Generate boolean mask for points outside the domain.
        
        Parameters
        ----------
        n : int, optional
            Number of points along the longer axis.
            
        Returns
        -------
        np.ndarray
            Boolean mask (True for points outside).
        """
        return ~self.inmask(n)
    
    def domain(self, n: int = 500) -> np.ndarray:
        """Generate mesh with NaN outside the domain.
        
        Parameters
        ----------
        n : int, optional
            Number of points along the longer axis.
            
        Returns
        -------
        np.ndarray
            Complex mesh with NaN for points outside.
        """
        mesh = self.mesh(n)
        mesh[self.outmask(n)] = np.nan
        return mesh
    
    def union(self, other: 'Domain') -> 'CompositeDomain':
        """Create union with another domain.
        
        Parameters
        ----------
        other : Domain
            Domain to union with.
            
        Returns
        -------
        CompositeDomain
            Union of the two domains.
        """
        return CompositeDomain(self, other, 'union')
    
    def intersection(self, other: 'Domain') -> 'CompositeDomain':
        """Create intersection with another domain.
        
        Parameters
        ----------
        other : Domain
            Domain to intersect with.
            
        Returns
        -------
        CompositeDomain
            Intersection of the two domains.
        """
        return CompositeDomain(self, other, 'intersection')
    
    def difference(self, other: 'Domain') -> 'CompositeDomain':
        """Create set difference with another domain.
        
        Parameters
        ----------
        other : Domain
            Domain to subtract.
            
        Returns
        -------
        CompositeDomain
            Points in this domain but not in other.
        """
        return CompositeDomain(self, other, 'difference')
    
    def __or__(self, other: 'Domain') -> 'CompositeDomain':
        """Union operator (|)."""
        return self.union(other)
    
    def __and__(self, other: 'Domain') -> 'CompositeDomain':
        """Intersection operator (&)."""
        return self.intersection(other)
    
    def __sub__(self, other: 'Domain') -> 'CompositeDomain':
        """Set difference operator (-)."""
        return self.difference(other)


class Rectangle(Domain):
    """Rectangular domain in the complex plane.
    
    Parameters
    ----------
    re_length : float
        Length along the real axis.
    im_length : float
        Length along the imaginary axis.
    center : complex, optional
        Center of the rectangle.
    square : bool, optional
        Whether to use square viewing window.
    """
    
    def __init__(self,
                 re_length: float,
                 im_length: float,
                 center: complex = 0+0j,
                 square: bool = True):
        """Initialize rectangular domain."""
        if re_length <= 0 or im_length <= 0:
            raise ValidationError("Rectangle dimensions must be positive")
        
        self.re_length = re_length
        self.im_length = im_length
        self.center = center
        
        # Calculate bounds
        half_re = re_length / 2
        half_im = im_length / 2
        
        real_bounds = (center.real - half_re, center.real + half_re)
        imag_bounds = (center.imag - half_im, center.imag + half_im)
        
        super().__init__(real_bounds, imag_bounds, square)
    
    def contains(self, z: np.ndarray) -> np.ndarray:
        """Check if points are inside the rectangle."""
        re_min, re_max = self.window_real
        im_min, im_max = self.window_imag
        
        return (
            (np.real(z) >= re_min) & 
            (np.real(z) <= re_max) &
            (np.imag(z) >= im_min) & 
            (np.imag(z) <= im_max)
        )


class Disk(Domain):
    """Circular domain in the complex plane.
    
    Parameters
    ----------
    radius : float
        Radius of the disk.
    center : complex, optional
        Center of the disk.
    """
    
    def __init__(self,
                 radius: float,
                 center: complex = 0+0j):
        """Initialize disk domain."""
        if radius <= 0:
            raise ValidationError("Radius must be positive")
        
        self.radius = radius
        self.center = center
        
        # Calculate bounds (always square for disks)
        real_bounds = (center.real - radius, center.real + radius)
        imag_bounds = (center.imag - radius, center.imag + radius)
        
        super().__init__(real_bounds, imag_bounds, square=True)
    
    def contains(self, z: np.ndarray) -> np.ndarray:
        """Check if points are inside the disk."""
        return np.abs(z - self.center) <= self.radius


class Annulus(Domain):
    """Annular (ring-shaped) domain in the complex plane.
    
    Parameters
    ----------
    inner_radius : float
        Inner radius of the annulus.
    outer_radius : float
        Outer radius of the annulus.
    center : complex, optional
        Center of the annulus.
    """
    
    def __init__(self,
                 inner_radius: float,
                 outer_radius: float,
                 center: complex = 0+0j):
        """Initialize annulus domain."""
        if inner_radius <= 0:
            raise ValidationError("Inner radius must be positive")
        if outer_radius <= inner_radius:
            raise ValidationError("Outer radius must be greater than inner radius")
        
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.center = center
        
        # Calculate bounds (always square for annuli)
        real_bounds = (center.real - outer_radius, center.real + outer_radius)
        imag_bounds = (center.imag - outer_radius, center.imag + outer_radius)
        
        super().__init__(real_bounds, imag_bounds, square=True)
    
    def contains(self, z: np.ndarray) -> np.ndarray:
        """Check if points are inside the annulus."""
        dist = np.abs(z - self.center)
        return (dist >= self.inner_radius) & (dist <= self.outer_radius)


class CompositeDomain(Domain):
    """Domain formed by set operations on other domains.
    
    This class represents domains created by union, intersection,
    or difference of two existing domains.
    
    Parameters
    ----------
    domain1 : Domain
        First domain.
    domain2 : Domain
        Second domain.
    operation : {'union', 'intersection', 'difference'}
        Set operation to apply.
    """
    
    def __init__(self,
                 domain1: Domain,
                 domain2: Domain,
                 operation: str):
        """Initialize composite domain."""
        if operation not in ['union', 'intersection', 'difference']:
            raise ValidationError("Operation must be 'union', 'intersection', or 'difference'")
        
        self.domain1 = domain1
        self.domain2 = domain2
        self.operation = operation
        
        # Calculate combined bounds based on operation
        if operation == 'intersection':
            # Intersection can only be within overlap
            real_bounds = (
                max(domain1.window_real[0], domain2.window_real[0]),
                min(domain1.window_real[1], domain2.window_real[1])
            )
            imag_bounds = (
                max(domain1.window_imag[0], domain2.window_imag[0]),
                min(domain1.window_imag[1], domain2.window_imag[1])
            )
            # Handle non-overlapping domains
            if real_bounds[0] > real_bounds[1]:
                real_bounds = (0, 0)
            if imag_bounds[0] > imag_bounds[1]:
                imag_bounds = (0, 0)
        elif operation == 'difference':
            # Difference is contained within first domain
            real_bounds = domain1.window_real
            imag_bounds = domain1.window_imag
        else:  # union
            # Union needs full extent of both
            all_reals = [
                domain1.window_real[0], domain1.window_real[1],
                domain2.window_real[0], domain2.window_real[1]
            ]
            all_imags = [
                domain1.window_imag[0], domain1.window_imag[1],
                domain2.window_imag[0], domain2.window_imag[1]
            ]
            real_bounds = (min(all_reals), max(all_reals))
            imag_bounds = (min(all_imags), max(all_imags))
        
        # Use square if either parent uses square
        square = domain1.square or domain2.square
        
        super().__init__(real_bounds, imag_bounds, square)
    
    def contains(self, z: np.ndarray) -> np.ndarray:
        """Check if points are in the composite domain."""
        mask1 = self.domain1.contains(z)
        mask2 = self.domain2.contains(z)
        
        if self.operation == 'union':
            return mask1 | mask2
        elif self.operation == 'intersection':
            return mask1 & mask2
        else:  # difference
            return mask1 & ~mask2
    
    def calculate_tight_bounds(self, sample_density: int = 100) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Calculate tight bounds by sampling the actual domain.
        
        Parameters
        ----------
        sample_density : int
            Number of sample points along each axis.
            
        Returns
        -------
        real_bounds, imag_bounds : tuple of tuple
            Tight bounds for the composite domain.
        """
        # Get loose bounds from current window
        real_min, real_max = self.window_real
        imag_min, imag_max = self.window_imag
        
        # Create sample grid
        real_samples = np.linspace(real_min, real_max, sample_density)
        imag_samples = np.linspace(imag_min, imag_max, sample_density)
        real_grid, imag_grid = np.meshgrid(real_samples, imag_samples)
        z_samples = real_grid + 1j * imag_grid
        
        # Find points actually in the domain
        mask = self.contains(z_samples.ravel())
        if not np.any(mask):
            # Empty domain, return small bounds at origin
            return (-0.1, 0.1), (-0.1, 0.1)
        
        valid_points = z_samples.ravel()[mask]
        
        # Calculate tight bounds with small margin
        margin = 0.05  # 5% margin
        real_extent = np.real(valid_points)
        imag_extent = np.imag(valid_points)
        
        real_range = real_extent.max() - real_extent.min()
        imag_range = imag_extent.max() - imag_extent.min()
        
        # Ensure minimum size
        if real_range < 0.1:
            real_range = 0.1
        if imag_range < 0.1:
            imag_range = 0.1
        
        real_bounds = (
            real_extent.min() - margin * real_range,
            real_extent.max() + margin * real_range
        )
        imag_bounds = (
            imag_extent.min() - margin * imag_range,
            imag_extent.max() + margin * imag_range
        )
        
        return real_bounds, imag_bounds
    
    @property
    def tight_bounds(self):
        """Get tight bounds, calculating if necessary."""
        if not hasattr(self, '_tight_bounds'):
            self._tight_bounds = self.calculate_tight_bounds()
        return self._tight_bounds
    
    def mesh(self, n: int = 500, use_tight_bounds: bool = True) -> np.ndarray:
        """Generate mesh grid of complex numbers.
        
        For composite domains, can use tight bounds for better fit.
        
        Parameters
        ----------
        n : int, optional
            Number of points along the longer axis.
        use_tight_bounds : bool, optional
            If True, use tight bounds for composite domains.
            
        Returns
        -------
        np.ndarray
            2D array of complex values.
        """
        n = validate_resolution(n, min_val=2, max_val=10000)
        
        if use_tight_bounds:
            real_bounds, imag_bounds = self.tight_bounds
            real_length = real_bounds[1] - real_bounds[0]
            imag_length = imag_bounds[1] - imag_bounds[0]
            spacing = max(real_length, imag_length) / n
            
            real_axis = np.linspace(
                real_bounds[0], 
                real_bounds[1], 
                ceil(real_length / spacing)
            )
            imag_axis = np.linspace(
                imag_bounds[0], 
                imag_bounds[1], 
                ceil(imag_length / spacing)
            )
        else:
            # Fall back to parent implementation
            real_length = self.window_real[1] - self.window_real[0]
            imag_length = self.window_imag[1] - self.window_imag[0]
            spacing = self.spacing(n)
            
            real_axis = np.linspace(
                self.window_real[0], 
                self.window_real[1], 
                ceil(real_length / spacing)
            )
            imag_axis = np.linspace(
                self.window_imag[0], 
                self.window_imag[1], 
                ceil(imag_length / spacing)
            )
        
        x, y = np.meshgrid(real_axis, imag_axis)
        return x + 1j * y