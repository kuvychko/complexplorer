"""Output-pinning regression tests for the PyVista surface kernel refactor.

These lock the *current* mesh output of the five public PyVista/STL entry points
(via their shared internals) BEFORE the surface-kernel refactor, so that faceting them
over `sample()` + `build_*()` + `SurfaceMesh` cannot silently change geometry, colors, or
scalars.

Golden snapshots live in `baselines/*.npz`. On first run a missing baseline is created and
the test skips; thereafter the test compares. The ONE intentional change in the refactor is
the `riemann_pv` projection flip (z=0 -> south pole), at which point its baseline is
regenerated on purpose (delete `baselines/riemann_pv.npz`).
"""

import pathlib

import numpy as np
import pytest

pytest.importorskip("pyvista")

import complexplorer as cp
from complexplorer.core.domain import Rectangle
from complexplorer.export.stl.ornament_generator import OrnamentGenerator
from complexplorer.plotting.pyvista.plot_3d import create_complex_surface
from complexplorer.plotting.pyvista.riemann import riemann_pv

BASELINES = pathlib.Path(__file__).parent / "baselines"
ARRAYS = ("points", "RGB", "magnitude", "phase")


def _signature(mesh) -> dict:
    """Extract the geometry + decoration arrays that must survive the refactor."""
    sig = {"points": np.asarray(mesh.points)}
    for name in ("RGB", "magnitude", "phase"):
        sig[name] = np.asarray(mesh[name])
    return sig


def _check_or_save(name: str, sig: dict) -> None:
    BASELINES.mkdir(exist_ok=True)
    path = BASELINES / f"{name}.npz"
    if not path.exists():
        np.savez_compressed(path, **sig)
        pytest.skip(f"regression baseline created: {path.name}")
    baseline = np.load(path)
    for key in ARRAYS:
        np.testing.assert_allclose(
            sig[key],
            baseline[key],
            rtol=1e-5,
            atol=1e-7,
            equal_nan=True,
            err_msg=f"{name}: '{key}' drifted from the pre-refactor baseline",
        )


def _cmap():
    return cp.Phase(n_phi=6, v_base=0.6)


# --- 1.1 landscape (create_complex_surface backs plot_landscape_pv + pair_plot) ---


def test_landscape_with_modulus():
    grid, _ = create_complex_surface(
        Rectangle(4, 4),
        lambda z: (z**2 - 1) / (z**2 + 1),
        resolution=30,
        cmap=_cmap(),
        modulus_mode="arctan",
    )
    _check_or_save("landscape_modulus", _signature(grid))


def test_landscape_with_domain_mask():
    # Disk masks the corners of the bounding grid -> locks the out-of-domain NaN-blanking
    # path. Uses a pole-free function so in-domain values are finite and deterministic
    # (an exact in-domain pole hit yields non-deterministic RGB from the colormap's
    # nan->int cast, which would make this baseline flaky).
    from complexplorer.core.domain import Disk

    grid, _ = create_complex_surface(
        Disk(1.5),
        lambda z: z**2,
        resolution=25,
        cmap=_cmap(),
        modulus_mode="none",
    )
    _check_or_save("landscape_mask", _signature(grid))


# --- 1.2 riemann_pv (faithful: extract the real mesh via return_plotter) ---


def test_riemann_pv():
    plotter = riemann_pv(
        lambda z: (z - 1) / (z + 1),
        resolution=30,
        cmap=_cmap(),
        modulus_mode="arctan",
        interactive=False,
        return_plotter=True,
        show_orientation=False,
    )
    try:
        _check_or_save("riemann_pv", _signature(plotter.meshes[0]))
    finally:
        plotter.close()


# --- 1.3 ornament (mesh arrays; unchanged by the projection flip) ---


def test_ornament_generate():
    mesh = OrnamentGenerator(
        lambda z: z / (z**3 - 1), resolution=30, scaling="arctan"
    ).generate_ornament(verbose=False)
    _check_or_save("ornament", _signature(mesh))
