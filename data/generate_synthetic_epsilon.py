"""Generate the explicitly synthetic passive dielectric table used in demos.

This single-oscillator Lorentz model is pedagogical input, not measured CFO
data and not a fitted representation of any particular material.
"""

from __future__ import annotations

import cmath
import csv
import hashlib
import json
from pathlib import Path


EV_NM = 1239.8419843320026
EPSILON_INFINITY = 4.0
OSCILLATOR_STRENGTH_EV2 = 14.0
RESONANCE_EV = 4.2
DAMPING_EV = 0.8
OUTPUT = Path(__file__).resolve().with_name("synthetic_epsilon.csv")
MANIFEST = OUTPUT.with_name("manifest.json")


def epsilon_at(wavelength_nm: float) -> complex:
    """Return a passive Lorentz epsilon using exp(-i omega t) convention."""
    energy_ev = EV_NM / wavelength_nm
    denominator = RESONANCE_EV**2 - energy_ev**2 - 1j * DAMPING_EV * energy_ev
    return EPSILON_INFINITY + OSCILLATOR_STRENGTH_EV2 / denominator


def main() -> None:
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("wavelength_nm", "eps1", "eps2", "n", "k"))
        for wavelength_nm in range(200, 801, 10):
            epsilon = epsilon_at(float(wavelength_nm))
            index = cmath.sqrt(epsilon)
            if index.real < 0 or index.imag < 0 or epsilon.imag < 0:
                raise ValueError("expected the passive square-root branch")
            writer.writerow(
                (
                    wavelength_nm,
                    f"{epsilon.real:.12g}",
                    f"{epsilon.imag:.12g}",
                    f"{index.real:.12g}",
                    f"{index.imag:.12g}",
                )
            )
    manifest = {
        "file": OUTPUT.name,
        "role": "synthetic teaching input for code examples and numerical checks",
        "source_class": "synthetic",
        "license": "MIT",
        "generated_by": Path(__file__).name,
        "generation_command": "python data/generate_synthetic_epsilon.py",
        "model": {
            "type": "single damped Lorentz oscillator",
            "epsilon_infinity": EPSILON_INFINITY,
            "oscillator_strength_ev2": OSCILLATOR_STRENGTH_EV2,
            "resonance_ev": RESONANCE_EV,
            "damping_ev": DAMPING_EV,
            "energy_conversion_constant_ev_nm": EV_NM,
        },
        "conventions": {
            "time_dependence": "exp(-i omega t)",
            "dielectric_function": "epsilon1 + i epsilon2",
            "passive_input": "epsilon2 >= 0",
            "wavelength_unit": "nm",
            "epsilon_unit": "dimensionless",
            "wavelength_range_nm": [200, 800],
            "wavelength_step_nm": 10,
        },
        "sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
    }
    with MANIFEST.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote synthetic passive epsilon table: {OUTPUT.name} ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
