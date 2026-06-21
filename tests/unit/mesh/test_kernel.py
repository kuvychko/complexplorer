"""Unit tests for the PyVista surface kernel (SurfaceMesh + builders)."""

import numpy as np
import pytest

pytest.importorskip("pyvista")

import complexplorer as cp
from complexplorer.core.domain import Annulus, Rectangle
from complexplorer.core.field import sample, sample_sphere
from complexplorer.export.stl.ornament_generator import OrnamentGenerator
from complexplorer.mesh import build_landscape, build_relief
from complexplorer.plotting.pyvista.plot_3d import create_complex_surface
from complexplorer.utils.validation import ValidationError

CMAP = cp.Phase(n_phi=6, v_base=0.6)


class TestBuildLandscape:
    def test_arrays_and_shape(self):
        sm = build_landscape(
            sample(lambda z: np.sin(z), Rectangle(4, 4), resolution=20),
            cmap=CMAP,
            modulus_mode="arctan",
        )
        mesh = sm.to_pyvista()
        assert mesh.n_points == 400
        for name in ("RGB", "magnitude", "phase"):
            assert name in mesh.array_names
        assert mesh["RGB"].shape == (400, 3)

    def test_matches_create_complex_surface(self):
        f = lambda z: np.sin(z)  # noqa: E731 (pole-free -> deterministic colors)
        old, _ = create_complex_surface(
            Rectangle(4, 4), f, resolution=25, cmap=CMAP, modulus_mode="arctan"
        )
        new = build_landscape(
            sample(f, Rectangle(4, 4), resolution=25), cmap=CMAP, modulus_mode="arctan"
        ).to_pyvista()
        np.testing.assert_allclose(old.points, new.points, equal_nan=True)
        for k in ("RGB", "magnitude", "phase"):
            np.testing.assert_allclose(np.asarray(old[k]), np.asarray(new[k]), equal_nan=True)

    def test_rejects_sphere_field(self):
        with pytest.raises(ValidationError):
            build_landscape(sample_sphere(lambda z: z, resolution=10))


class TestBuildRelief:
    def test_arrays_and_faces(self):
        sm = build_relief(sample_sphere(lambda z: z**2, resolution=20), cmap=CMAP, scaling="arctan")
        mesh = sm.to_pyvista()
        assert mesh.n_cells > 0 and mesh.n_points > 0
        for name in ("RGB", "magnitude", "phase", "radius"):
            assert name in mesh.array_names

    def test_matches_ornament_no_domain(self):
        f = lambda z: z**2  # noqa: E731
        old = OrnamentGenerator(f, resolution=25, scaling="arctan", cmap=CMAP).generate_ornament(
            verbose=False
        )
        new = build_relief(
            sample_sphere(f, resolution=25), cmap=CMAP, scaling="arctan", for_stl=True
        ).to_pyvista()
        assert old.n_points == new.n_points
        for k in ("RGB", "magnitude", "phase", "radius"):
            np.testing.assert_allclose(np.asarray(old[k]), np.asarray(new[k]), equal_nan=True)

    def test_matches_ornament_with_domain(self):
        f = lambda z: (z - 1) / (z + 1)  # noqa: E731
        dom = Annulus(0.3, 3.0)
        old = OrnamentGenerator(
            f, resolution=30, scaling="arctan", cmap=CMAP, domain=dom
        ).generate_ornament(verbose=False)
        new = build_relief(
            sample_sphere(f, resolution=30, domain=dom), cmap=CMAP, scaling="arctan", for_stl=True
        ).to_pyvista()
        assert old.n_points == new.n_points  # cell removal applied identically
        for k in ("magnitude", "phase", "radius"):
            np.testing.assert_allclose(np.asarray(old[k]), np.asarray(new[k]), equal_nan=True)

    def test_rejects_planar_field(self):
        with pytest.raises(ValidationError):
            build_relief(sample(lambda z: z, Rectangle(2, 2), resolution=10))


class TestSurfaceMeshExport:
    def test_save_stl_nonempty_and_finite(self, tmp_path):
        sm = build_relief(
            sample_sphere(lambda z: z / (z**3 - 1), resolution=30),
            cmap=CMAP,
            scaling="arctan",
            for_stl=True,
        )
        out = tmp_path / "ornament.stl"
        sm.save_stl(str(out), size_mm=40, verbose=False)
        assert out.exists() and out.stat().st_size > 0

        import pyvista as pv

        reloaded = pv.read(str(out))
        assert reloaded.n_points > 0 and reloaded.n_cells > 0
        assert np.all(np.isfinite(reloaded.points))

    def test_landscape_stl_drops_nonfinite_vertices(self, tmp_path):
        # A masked landscape has NaN vertices; STL export must drop them.
        from complexplorer.core.domain import Disk

        sm = build_landscape(sample(lambda z: z**2, Disk(1.5), resolution=25), cmap=CMAP)
        out = tmp_path / "landscape.stl"
        sm.save_stl(str(out), size_mm=30, repair=False, validate=False, verbose=False)
        import pyvista as pv

        reloaded = pv.read(str(out))
        assert reloaded.n_points > 0
        assert np.all(np.isfinite(reloaded.points))
