"""Lightweight geometry, symmetry, and interface checks using synthetic input.

Passing these checks validates only the tested code invariants, not a material
model or comparison with an experiment.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from pycdm.geometry import helix_nanorods, single_particle
from pycdm.materials import load_tabulated_epsilon
from pycdm.polarizability import build_alpha_blocks
from pycdm.spectrum import circular_spectrum, solve_one


def load_synthetic_epsilon():
    epsilon_csv = PROJECT_ROOT / "data" / "synthetic_epsilon.csv"
    return load_tabulated_epsilon(epsilon_csv, clamp=False)


def check_single_particle_cd_small() -> float:
    epsilon = load_synthetic_epsilon()
    particle = single_particle((4.0, 4.0, 4.0), material="SyntheticMaterial")
    df = circular_spectrum(
        particle,
        {"SyntheticMaterial": epsilon},
        [300.0, 430.0, 680.0],
        medium_n=1.33,
        alpha_method="kuwata",
    )
    max_cd = float(np.max(np.abs(df["cd_ext_nm2"].to_numpy())))
    assert max_cd < 1e-10, f"single isotropic particle CD should be near zero, got {max_cd}"
    return max_cd


def check_enantiomer_cd_reversal() -> float:
    epsilon = load_synthetic_epsilon()
    common = dict(
        rod_radius_nm=4.0,
        rod_length_nm=40.0,
        helix_radius_nm=40.0,
        pitch_nm=80.0,
        rods_per_turn=8,
        turns=2.0,
        array_shape=(1, 1),
        material="SyntheticMaterial",
    )
    left = helix_nanorods(**common, handedness="left")
    right = helix_nanorods(**common, handedness="right")
    wavelengths = [290.0, 330.0, 430.0, 530.0]
    dl = circular_spectrum(left, {"SyntheticMaterial": epsilon}, wavelengths, medium_n=1.33, alpha_method="kuwata")
    dr = circular_spectrum(right, {"SyntheticMaterial": epsilon}, wavelengths, medium_n=1.33, alpha_method="kuwata")
    mismatch = float(np.max(np.abs(dl["cd_ext_nm2"].to_numpy() + dr["cd_ext_nm2"].to_numpy())))
    scale = float(np.max(np.abs(dl["cd_ext_nm2"].to_numpy())) + 1e-30)
    rel = mismatch / scale
    assert rel < 1e-9, f"enantiomer CD should reverse sign; relative mismatch {rel}"
    return rel


def check_matrix_shapes() -> tuple[int, tuple[int, int], tuple[int, int, int]]:
    epsilon = load_synthetic_epsilon()
    geom = helix_nanorods(
        rod_radius_nm=4.0,
        rod_length_nm=40.0,
        helix_radius_nm=35.0,
        pitch_nm=70.0,
        rods_per_turn=6,
        turns=1.0,
        material="SyntheticMaterial",
    )
    result = solve_one(geom, {"SyntheticMaterial": epsilon}, 430.0, medium_n=1.33, alpha_method="kuwata")
    alpha_blocks = build_alpha_blocks(geom, {"SyntheticMaterial": epsilon}, 430.0, eps_medium=1.33**2)
    assert result["F"].shape == (3 * len(geom), 3 * len(geom))
    assert alpha_blocks.shape == (len(geom), 3, 3)
    return len(geom), result["F"].shape, alpha_blocks.shape


def main() -> None:
    print("single particle max CD:", check_single_particle_cd_small())
    print("enantiomer CD reversal relative mismatch:", check_enantiomer_cd_reversal())
    print("matrix shapes:", check_matrix_shapes())
    print("sanity checks passed")


if __name__ == "__main__":
    main()
