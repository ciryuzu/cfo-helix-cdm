"""Synthetic-input helix demo; the bundled epsilon is not CFO material data."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from pycdm.geometry import helix_nanorods
from pycdm.materials import load_tabulated_epsilon
from pycdm.plotting import plot_circular_spectrum, plot_geometry_xy
from pycdm.spectrum import circular_spectrum


def main() -> None:
    print("Synthetic example only; these results are not experimental CFO evidence.")
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)

    epsilon_csv = PROJECT_ROOT / "data" / "synthetic_epsilon.csv"
    epsilon = load_tabulated_epsilon(epsilon_csv, clamp=False)

    geom = helix_nanorods(
        rod_radius_nm=4.0,
        rod_length_nm=40.0,
        helix_radius_nm=40.0,
        pitch_nm=80.0,
        rods_per_turn=8,
        turns=2.0,
        handedness="left",
        array_shape=(1, 1),
        material="SyntheticMaterial",
    )

    wavelengths = np.arange(280.0, 801.0, 10.0)
    df = circular_spectrum(
        geom,
        {"SyntheticMaterial": epsilon},
        wavelengths,
        medium_n=1.33,
        alpha_method="kuwata",
        per_particle=True,
    )

    csv_path = out / "synthetic_helix_demo_spectrum.csv"
    plot_path = out / "synthetic_helix_demo_spectrum.png"
    geom_path = out / "synthetic_helix_demo_geometry_xy.png"
    df.to_csv(csv_path, index=False)
    plot_circular_spectrum(df, plot_path)
    plot_geometry_xy(geom, geom_path)

    print(f"particles: {len(geom)}")
    print(f"spectrum: {csv_path}")
    print(f"plot: {plot_path}")
    print(f"geometry: {geom_path}")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
