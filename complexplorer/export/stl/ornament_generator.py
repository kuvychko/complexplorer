"""STL ornament generator for complex functions.

This module generates 3D-printable ornaments from complex functions
by using the modulus-scaled Riemann sphere mesh with optional simple repairs.
"""

import os
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from ...core.colormap import Colormap, Phase
from ...core.domain import Domain
from ...core.field import ComplexField, sample_sphere
from ...core.scaling import sampled_normalization_constant
from ...exceptions import ValidationError
from ...mesh import build_relief
from ...utils.mesh_distortion import get_default_scaling_params
from .mesh_repair import repair_mesh_simple
from .utils import center_mesh, scale_to_size, validate_printability

if TYPE_CHECKING:
    import pyvista as pv

# The ornament defaults live here and nowhere else: the CLI and ``create_ornament`` both pass
# ``None`` through rather than repeating a literal.
DEFAULT_SCALING = "logarithmic"
DEFAULT_NORMALIZE = "geometric"

# How pointy the features are. With ``r = r_min + delta * logistic(log|f| / k)`` the surface
# approaches a feature of order ``mu`` like ``distance ** (mu / k)``, so the tip exponent -- not
# ``k`` -- is what the eye reads:
#
#     mu / k  >  1   rounded dome (a blob)
#     mu / k  =  1   an exact cone
#     mu / k  <  1   a cusp, sharper than a cone
#
# ``k = pointiness * pole_order`` fixes that exponent at ``1 / pointiness`` for every piece, so a
# double pole prints as sharp as a simple one rather than twice as blunt. 2.0 was chosen by building,
# rendering and inspecting a ten-piece collection, not by optimizing a bulk-contrast measure -- that
# measure rewards blunting every tip, because a well-contoured blob spreads more area across the
# radial range than a sculpted star does.
DEFAULT_POINTINESS = 2.0
DEFAULT_POLE_ORDER = 1.0

# Ceiling on the DERIVED scale, because the rule above assumes the mesh can deliver the dynamic range
# a given feature order demands, and past roughly order 3 it cannot: a feature of order ``mu`` only
# drives ``log|f|`` as far as the nearest sample gets to it. At order 5 an uncapped ``k = 10`` wants
# ``log|f|`` to reach +-46 to saturate, while a resolution-400 grid delivers about +-26, so the body
# flattens instead of the tip sharpening. Measured on an order-5 icosahedral relief: the radial range
# spans 67% at ``k = 10`` against 83% at ``k = 6``. The cap is a no-op at order 3 and below.
#
# It deliberately does NOT apply to a scale the caller sets through ``sharpness``: that is an
# instruction rather than a derivation, and the gain-calibrated ``ln(10)`` case must stay exact.
SHARPNESS_CAP = 6.0


class OrnamentGenerator:
    """Generate 3D-printable ornaments from complex functions.

    This version directly uses the modulus-scaled mesh from the
    Riemann sphere visualization without complex healing steps.

    Parameters
    ----------
    func : callable
        Complex function f(z) to visualize.
    resolution : int, default=150
        Mesh resolution (n_theta = n_phi).
    scaling : str, optional
        Modulus scaling method. Defaults to ``'logarithmic'``: a logistic in the log modulus, which
        is self-dual about ``|f| = 1`` and, unlike ``'arctan'``, admits a scale, so ``pointiness``
        has something to set. ``'arctan'`` reaches 90% of its height at ``|f| = 6.3`` whatever the
        function, which is a tip exponent well above 1 on a typical rational function -- a rounded
        pebble on every piece, regardless of the mathematics behind it.
    scaling_params : dict, optional
        Parameters for the scaling method. Anything given here wins over the values derived from
        ``pointiness``, ``sharpness`` and ``contrast``.
    cmap : Colormap, optional
        Colormap for visualization. Default is Phase colormap.
    domain : Domain, optional
        Domain to restrict evaluation. Helps avoid numerical issues.
    normalize : {'geometric', 'median'}, float or None, default='geometric'
        How to rescale ``|f|`` before the transfer, so the arbitrary constant in front of ``f`` stops
        changing the ornament's shape. ``'geometric'`` puts the area-weighted geometric mean of
        ``|f|`` at sea level; ``'median'`` uses the area-weighted median instead, which holds up
        better when a high-order feature at infinity covers enough of the sphere to drag sea level
        away from the structure worth seeing. A float is used as the constant directly, and ``None``
        disables normalization, reproducing the pre-3.1 mapping. The constant is estimated from the
        samples the relief is built from, so ``f`` is never evaluated twice.
    pointiness : float, default=2.0
        Tip sharpness, as the reciprocal of the tip exponent: the surface approaches a feature like
        ``distance ** (1 / pointiness)``. Larger is sharper. Sets the transfer's log-modulus scale to
        ``pointiness * pole_order``. Ignored by transfers that have no scale, such as ``'arctan'``.
    pole_order : float, default=1.0
        The order of the features being shaped. Scaling the transfer by it makes a double pole print
        as sharp as a simple one instead of twice as blunt. The resulting scale is capped at
        ``SHARPNESS_CAP`` (6.0), so orders above 3 do not ask for a dynamic range the mesh cannot
        deliver -- see that constant. It is not inferred from the function: fitting the scale to a
        piece's log-modulus spread is a known dead end that produces tip exponents from 1 to 8 and
        blunts most pieces, so the order is yours to state.
    sharpness : float, optional
        The log-modulus scale, set directly, bypassing ``pointiness * pole_order`` and its cap.
        ``ln(10)`` makes
        one unit of relief exactly one decade of gain -- 20 dB -- which turns a transfer function's
        relief into a calibrated scale readable like a Bode magnitude curve along a meridian. That is
        not expressible as a tip exponent, which is why it is settable on its own.
    contrast : tuple of (float, float), optional
        ``(boost, weight)`` for the two-scale transfer, which mixes a narrow logistic at
        ``scale / boost`` for bulk contrast with the wide one for the tips. Off by default: applied
        globally it degrades the pieces whose tips are the point of the piece, so it is a per-piece
        choice. Use it where a few widely separated features leave the body a near-perfect sphere.
        See :meth:`~complexplorer.core.scaling.ModulusScaling.log_mixture`.
    """

    def __init__(
        self,
        func: Callable,
        resolution: int = 150,
        scaling: str | None = None,
        scaling_params: dict[str, Any] | None = None,
        cmap: Colormap | None = None,
        domain: Domain | None = None,
        *,
        normalize: str | float | None = DEFAULT_NORMALIZE,
        pointiness: float = DEFAULT_POINTINESS,
        pole_order: float = DEFAULT_POLE_ORDER,
        sharpness: float | None = None,
        contrast: tuple[float, float] | None = None,
    ):
        """Initialize ornament generator."""
        self.func = func
        self.resolution = resolution
        self.cmap = cmap or Phase(phase_sectors=6, auto_scale_r=True)
        self.domain = domain
        self.normalize = normalize
        self.pointiness = pointiness
        self.pole_order = pole_order
        self.contrast = contrast

        # The log-modulus scale: set directly, or derived from the tip exponent and capped. See
        # SHARPNESS_CAP -- a derived scale beyond what the mesh can resolve flattens the body instead
        # of sharpening the tip, so it is clamped; an explicit scale is taken as given.
        if sharpness is not None:
            self.sharpness = float(sharpness)
        else:
            self.sharpness = min(float(pointiness) * float(pole_order), SHARPNESS_CAP)
        if self.sharpness <= 0:
            raise ValidationError(
                f"The log-modulus scale must be positive; got {self.sharpness} from "
                f"pointiness={pointiness}, pole_order={pole_order}, sharpness={sharpness}"
            )

        scaling = scaling or DEFAULT_SCALING
        # Contrast is a property of the transfer, so asking for it on the default transfer selects
        # the two-scale one. Asked for alongside an explicitly chosen mode, it is that mode's
        # business and is left alone.
        if contrast is not None and scaling == DEFAULT_SCALING:
            scaling = "log_mixture"
        self.scaling = scaling

        params = dict(get_default_scaling_params(scaling, for_stl=True))
        if scaling == "logarithmic":
            # logarithmic computes logistic(log|f| / ln(base)), so ln(base) IS the scale.
            params["base"] = float(np.exp(self.sharpness))
        elif scaling == "log_mixture":
            params["scale"] = self.sharpness
            if contrast is not None:
                boost, weight = contrast
                params["boost"] = float(boost)
                params["weight"] = float(weight)
        # Explicit parameters always win, so a caller can pin anything derived above.
        params.update(scaling_params or {})
        self.scaling_params = params

        self.sphere_mesh = None
        self.applied_normalization: float | None = None

    def _resolve_normalization(self, field: ComplexField) -> float | None:
        """The constant to multiply ``|f|`` by, from the field already sampled."""
        if self.normalize is None:
            return None
        if isinstance(self.normalize, str):
            return sampled_normalization_constant(
                np.asarray(field.modulus, dtype=float),
                np.asarray(field.sphere_xyz, dtype=float)[..., 2],
                statistic=self.normalize,
            )
        constant = float(self.normalize)
        if constant <= 0:
            raise ValidationError(
                f"An explicit normalization constant must be positive; got {constant}"
            )
        return constant

    def generate_ornament(self, verbose: bool = False) -> "pv.PolyData":
        """Generate the ornament mesh.

        Parameters
        ----------
        verbose : bool, optional
            Print progress information.

        Returns
        -------
        pv.PolyData
            Generated ornament mesh with color information.
        """
        if verbose:
            print("Generating Riemann sphere ornament:")
            print(f"  Resolution: {self.resolution}")
            print(f"  Scaling: {self.scaling}")
            print(f"  Parameters: {self.scaling_params}")

        # Sample on the sphere (canonical projection) and build the relief via the kernel. The
        # normalization constant comes from this same field -- f is evaluated once.
        field = sample_sphere(self.func, resolution=self.resolution, domain=self.domain)
        constant = self._resolve_normalization(field)
        self.applied_normalization = constant
        sm = build_relief(
            field,
            cmap=self.cmap,
            scaling=self.scaling,
            scaling_params=self.scaling_params,
            normalize=constant,
        )
        sphere = sm.to_pyvista()

        self.sphere_mesh = sphere

        if verbose:
            if constant is None:
                print("  Normalization: off")
            else:
                print(f"  Normalization: {constant:.6g} ({self.normalize})")
            print(f"  Generated mesh: {sphere.n_points} vertices, {sphere.n_cells} faces")
            actual_radii = np.linalg.norm(sphere.points, axis=1)
            print(f"  Radius range: [{actual_radii.min():.3f}, {actual_radii.max():.3f}]")

        return sphere

    def validate_mesh(self, size_mm: float = 50, verbose: bool = True) -> dict[str, Any]:
        """Validate mesh for 3D printing.

        Parameters
        ----------
        size_mm : float, default=50
            Target size in millimeters.
        verbose : bool, default=True
            Print validation results.

        Returns
        -------
        dict
            Validation results.
        """
        if self.sphere_mesh is None:
            raise ValidationError("No mesh generated yet. Call generate_ornament() first.")

        return validate_printability(self.sphere_mesh, size_mm, verbose)

    def save_stl(
        self,
        filename: str,
        size_mm: float = 50,
        center: bool = True,
        repair: bool = True,
        binary: bool = True,
        validate: bool = True,
        verbose: bool = True,
        size_measure: str = "extent",
    ) -> str:
        """Save the ornament as STL file.

        Parameters
        ----------
        filename : str
            Output filename (should end with .stl).
        size_mm : float, default=50
            Scale mesh to this size in millimeters, measured per ``size_measure``.
        size_measure : {'extent', 'max'}, default='extent'
            What ``size_mm`` measures. ``'extent'`` is the object's true tip-to-tip width, which is a
            property of the shape; ``'max'`` is the largest axis-aligned bounding-box dimension, which
            depends on how the piece sits in the coordinate frame and is what versions before 3.1
            used. A relief with spikes on the cube diagonals is understated by ``sqrt(3)`` under
            ``'max'``. Use ``'max'`` when the box is what matters, such as fitting a build plate.
        center : bool, default=True
            Center the mesh at origin.
        repair : bool, default=True
            Apply simple mesh repair (fill holes).
        binary : bool, default=True
            Save as binary STL (smaller file size).
        validate : bool, default=True
            Validate before saving.
        verbose : bool, default=True
            Print progress information.

        Returns
        -------
        str
            Path to saved file.
        """
        if self.sphere_mesh is None:
            raise ValidationError("No mesh generated yet. Call generate_ornament() first.")

        mesh = self.sphere_mesh.copy()

        # Repair if requested
        if repair:
            if verbose:
                print("\nRepairing mesh...")
            mesh = repair_mesh_simple(mesh, fill_holes=True, verbose=verbose)

        # Orient the surface consistently outward, so the facet normals written into the STL mean
        # something to a consumer that reads them rather than recomputing. Repair leaves winding
        # alone, and a translation or a positive scale cannot disturb this, so it is done once here.
        mesh = mesh.compute_normals(
            consistent_normals=True, auto_orient_normals=True, inplace=False
        )

        # Scale to target size, about the origin, which leaves the relief's star centre there.
        mesh = scale_to_size(mesh, size_mm, axis=size_measure)

        # Validate BEFORE centring: the radii reported are measured from the origin, and centring
        # moves the bounding-box centre there instead, which understates the range on a lopsided
        # piece whose star centre and box centre are far apart.
        if validate:
            results = validate_printability(mesh, size_mm, verbose=verbose)
            if not results["is_watertight"] and not repair:
                warnings.warn(
                    "Mesh is not watertight. Consider enabling repair=True.",
                    stacklevel=2,
                )

        # Center if requested
        if center:
            mesh = center_mesh(mesh)

        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

        # Save
        mesh.save(filename, binary=binary)

        if verbose:
            file_size_mb = os.path.getsize(filename) / (1024 * 1024)
            print(f"\nSaved STL file: {filename}")
            print(f"File size: {file_size_mb:.2f} MB")

        return filename

    def generate_and_save(
        self,
        filename: str,
        size_mm: float = 50,
        center: bool = True,
        repair: bool = True,
        binary: bool = True,
        validate: bool = True,
        verbose: bool = True,
        size_measure: str = "extent",
    ) -> str:
        """Generate ornament and save as STL in one step.

        Parameters
        ----------
        filename : str
            Output filename.
        size_mm : float, default=50
            Target size in millimeters, measured per ``size_measure``.
        size_measure : {'extent', 'max'}, default='extent'
            What ``size_mm`` measures -- see :meth:`save_stl`.
        center : bool, default=True
            Center the mesh.
        repair : bool, default=True
            Apply simple mesh repair.
        binary : bool, default=True
            Use binary STL format.
        validate : bool, default=True
            Validate before saving.
        verbose : bool, default=True
            Print progress.

        Returns
        -------
        str
            Path to saved file.
        """
        self.generate_ornament(verbose=verbose)
        return self.save_stl(
            filename, size_mm, center, repair, binary, validate, verbose, size_measure
        )


def create_ornament(
    func: Callable,
    filename: str,
    size_mm: float = 50,
    resolution: int = 150,
    scaling: str | None = None,
    scaling_params: dict[str, Any] | None = None,
    cmap: Colormap | None = None,
    domain: Domain | None = None,
    verbose: bool = True,
    *,
    size_measure: str = "extent",
    normalize: str | float | None = DEFAULT_NORMALIZE,
    pointiness: float = DEFAULT_POINTINESS,
    pole_order: float = DEFAULT_POLE_ORDER,
    sharpness: float | None = None,
    contrast: tuple[float, float] | None = None,
) -> str:
    """Create a 3D-printable ornament from a complex function.

    Convenience function for creating STL files from complex functions.

    Parameters
    ----------
    func : callable
        Complex function to visualize.
    filename : str
        Output STL filename.
    size_mm : float, default=50
        Size in millimeters, measured per ``size_measure``.
    size_measure : {'extent', 'max'}, default='extent'
        What ``size_mm`` measures -- see :meth:`OrnamentGenerator.save_stl`.
    resolution : int, default=150
        Mesh resolution.
    scaling : str, optional
        Modulus scaling method. Defaults to the ornament transfer, a logistic in the log modulus.
    scaling_params : dict, optional
        Scaling parameters.
    cmap : Colormap, optional
        Color mapping.
    domain : Domain, optional
        Domain restriction.
    verbose : bool, default=True
        Print progress.
    normalize, pointiness, pole_order, sharpness, contrast
        Relief shaping, passed through to :class:`OrnamentGenerator` -- see it for the details.

    Returns
    -------
    str
        Path to saved STL file.
    """
    gen = OrnamentGenerator(
        func,
        resolution,
        scaling,
        scaling_params,
        cmap,
        domain,
        normalize=normalize,
        pointiness=pointiness,
        pole_order=pole_order,
        sharpness=sharpness,
        contrast=contrast,
    )
    return gen.generate_and_save(filename, size_mm, verbose=verbose, size_measure=size_measure)


__all__ = ["OrnamentGenerator", "create_ornament"]
