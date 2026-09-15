"""Regenerate the Phase reference fixture used by test_phase_golden.py.

The fixture was first recorded from the pre-rebase implementation (which took `n_phi`); the
configs are written in the canonical `phase_sectors` naming, so re-running this against the
current code must reproduce byte-identical arrays.
"""

import json

import numpy as np

from complexplorer import Phase

CONFIGS = [
    {},
    {"phase_sectors": 6},
    {"r_linear_step": 0.5},
    {"r_log_base": 2.0},
    {"phase_sectors": 6, "r_linear_step": 0.5},
    {"phase_sectors": 6, "r_linear_step": 0.4, "r_log_base": 3.0},
    {"phase_sectors": 6, "auto_scale_r": True},
    {"phase_sectors": 8, "auto_scale_r": True, "scale_radius": 0.8},
    {"phase_sectors": 12, "v_base": 0.3},
    {"v_base": 0.0},
]


def grid():
    x = np.linspace(-2.0, 2.0, 64)
    z = x[None, :] + 1j * x[:, None]
    # Values the contract cares about: exact zero, poles, NaN, +/-inf, tiny and huge moduli.
    specials = np.array(
        [
            0,
            np.nan,
            np.inf,
            -np.inf,
            1j * np.inf,
            complex(np.nan, np.nan),
            1e-12,
            1e12,
            1 + 0j,
            -1 + 0j,
            1j,
            -1j,
        ],
        dtype=complex,
    )
    z = z.copy()
    z.flat[: specials.size] = specials
    return z


def to_current(cfg):
    cfg = dict(cfg)
    if "phase_sectors" in cfg:
        cfg["phase_sectors"] = cfg.pop("phase_sectors")
    return cfg


def main():
    z = grid()
    arrays = {"z": z}
    for i, cfg in enumerate(CONFIGS):
        arrays[f"rgb_{i}"] = Phase(**cfg).rgb(z)
    np.savez_compressed("tests/data/phase_golden.npz", **arrays)
    with open("tests/data/phase_golden_configs.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(CONFIGS, f, indent=2)
        f.write("\n")
    print("wrote fixture:", len(CONFIGS), "configs, grid", z.shape)


main()
