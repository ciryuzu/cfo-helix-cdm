"""High-level spectrum helpers."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .core import interaction_matrix, polarization, solve_fields
from .geometry import ParticleSet
from .incident import plane_wave_field
from .observables import absorption, extinction
from .polarizability import build_alpha_blocks
from .rotations import euler_zyz_active


def solve_one(
    particles: ParticleSet,
    epsilon_functions: Mapping[str, object],
    wavelength_nm: float,
    *,
    medium_n: float = 1.33,
    alpha_method: str = "kuwata",
    k_hat: Sequence[float] = (0.0, 0.0, 1.0),
    handedness: str = "left",
) -> dict[str, object]:
    eps_medium = medium_n**2
    k = 2.0 * np.pi * medium_n / wavelength_nm
    alpha_blocks = build_alpha_blocks(
        particles,
        epsilon_functions,
        wavelength_nm,
        eps_medium=eps_medium,
        method=alpha_method,
    )
    F = interaction_matrix(k, particles.positions_nm, alpha_blocks)
    Ein = plane_wave_field(particles.positions_nm, k, k_hat=np.asarray(k_hat, dtype=float), handedness=handedness)
    E = solve_fields(F, Ein)
    P = polarization(E, alpha_blocks)
    return {
        "wavelength_nm": wavelength_nm,
        "k": k,
        "alpha_blocks": alpha_blocks,
        "F": F,
        "Ein": Ein,
        "E": E,
        "P": P,
        "extinction": extinction(k, P, Ein),
        "absorption": absorption(k, P, E),
    }


def circular_spectrum(
    particles: ParticleSet,
    epsilon_functions: Mapping[str, object],
    wavelengths_nm: Sequence[float],
    *,
    medium_n: float = 1.33,
    alpha_method: str = "kuwata",
    k_hat: Sequence[float] = (0.0, 0.0, 1.0),
    per_particle: bool = True,
) -> pd.DataFrame:
    rows = []
    n_particles = len(particles)
    scale = n_particles if per_particle else 1.0
    for wl in wavelengths_nm:
        left = solve_one(
            particles,
            epsilon_functions,
            float(wl),
            medium_n=medium_n,
            alpha_method=alpha_method,
            k_hat=k_hat,
            handedness="left",
        )
        right = solve_one(
            particles,
            epsilon_functions,
            float(wl),
            medium_n=medium_n,
            alpha_method=alpha_method,
            k_hat=k_hat,
            handedness="right",
        )
        cext_l = float(left["extinction"]) / scale
        cext_r = float(right["extinction"]) / scale
        cabs_l = float(left["absorption"]) / scale
        cabs_r = float(right["absorption"]) / scale
        rows.append(
            {
                "wavelength_nm": float(wl),
                "cext_left_nm2": cext_l,
                "cext_right_nm2": cext_r,
                "cext_avg_nm2": 0.5 * (cext_l + cext_r),
                "cd_ext_nm2": cext_l - cext_r,
                "cabs_left_nm2": cabs_l,
                "cabs_right_nm2": cabs_r,
                "cabs_avg_nm2": 0.5 * (cabs_l + cabs_r),
                "cd_abs_nm2": cabs_l - cabs_r,
            }
        )
    return pd.DataFrame(rows)


def sphere_cubature_gl(n_requested: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre cubature on the sphere, matching CoupledDipole.jl.

    Returns direction vectors and weights normalized so sum(weights) = 1.
    """
    rnd = int(np.ceil(np.sqrt(n_requested / 2.0)))
    x_phi, w_phi = np.polynomial.legendre.leggauss(2 * rnd)
    x_mu, w_mu = np.polynomial.legendre.leggauss(rnd)
    phi = np.pi * (x_phi + 1.0)
    w_phi = np.pi * w_phi
    mu01 = 0.5 * (x_mu + 1.0)
    w_mu01 = 0.5 * w_mu
    directions = []
    weights = []
    for mu, wm in zip(mu01, w_mu01):
        theta = np.arccos(2.0 * mu - 1.0)
        for ph, wp in zip(phi, w_phi):
            R = euler_zyz_active(ph, theta, 0.0)
            directions.append(R[:, 2])
            weights.append((1.0 / (2.0 * np.pi)) * wm * wp)
    return np.asarray(directions, dtype=float), np.asarray(weights, dtype=float)


def orientation_averaged_circular_spectrum(
    particles: ParticleSet,
    epsilon_functions: Mapping[str, object],
    wavelengths_nm: Sequence[float],
    *,
    medium_n: float = 1.33,
    alpha_method: str = "kuwata",
    n_inc: int = 36,
    per_particle: bool = True,
) -> pd.DataFrame:
    """Orientation-averaged circular spectrum in Julia oa_df-like wide form."""
    directions, weights = sphere_cubature_gl(n_inc)
    rows = []
    scale = len(particles) if per_particle else 1.0
    eps_medium = medium_n**2
    for wl in wavelengths_nm:
        k = 2.0 * np.pi * medium_n / float(wl)
        alpha_blocks = build_alpha_blocks(
            particles,
            epsilon_functions,
            float(wl),
            eps_medium=eps_medium,
            method=alpha_method,
        )
        F = interaction_matrix(k, particles.positions_nm, alpha_blocks)

        # Match the Julia strategy: one system matrix, many incident RHS columns.
        rhs = []
        for direction in directions:
            rhs.append(
                plane_wave_field(
                    particles.positions_nm,
                    k,
                    k_hat=direction,
                    handedness="left",
                )
            )
        for direction in directions:
            rhs.append(
                plane_wave_field(
                    particles.positions_nm,
                    k,
                    k_hat=direction,
                    handedness="right",
                )
            )
        Ein = np.column_stack(rhs)
        E = solve_fields(F, Ein)
        P = polarization(E, alpha_blocks)

        dot_inc = np.sum(np.conj(Ein) * P, axis=0)
        dot_tot = np.sum(np.conj(E) * P, axis=0)
        pdotp = np.sum(np.conj(P) * P, axis=0)
        cext_all = 4.0 * np.pi * k * np.imag(dot_inc)
        cabs_all = 4.0 * np.pi * k * (np.imag(dot_tot) - (2.0 / 3.0) * k**3 * np.real(pdotp))

        n_dir = len(directions)
        cext_left = float(np.dot(weights, cext_all[:n_dir]))
        cext_right = float(np.dot(weights, cext_all[n_dir:]))
        cabs_left = float(np.dot(weights, cabs_all[:n_dir]))
        cabs_right = float(np.dot(weights, cabs_all[n_dir:]))

        cext_left /= scale
        cext_right /= scale
        cabs_left /= scale
        cabs_right /= scale
        rows.append(
            {
                "wavelength_nm": float(wl),
                "cext_average_nm2": 0.5 * (cext_left + cext_right),
                # CoupledDipole.jl's circular Jones ordering reports the
                # opposite L/R label relative to incident.circular_polarization.
                # Use the Julia-compatible convention here for benchmarking.
                "cext_dichroism_nm2": cext_right - cext_left,
                "cabs_average_nm2": 0.5 * (cabs_left + cabs_right),
                "cabs_dichroism_nm2": cabs_right - cabs_left,
                "cext_left_nm2": cext_left,
                "cext_right_nm2": cext_right,
                "cabs_left_nm2": cabs_left,
                "cabs_right_nm2": cabs_right,
            }
        )
    return pd.DataFrame(rows)


def oa_long_dataframe(df: pd.DataFrame, *, cluster: str = "helix", hand: str = "left", source: str = "python") -> pd.DataFrame:
    """Convert orientation-averaged output to CoupledDipole oa_df-like long form."""
    rows = []
    mapping = [
        ("average", "extinction", "cext_average_nm2"),
        ("average", "absorption", "cabs_average_nm2"),
        ("dichroism", "extinction", "cext_dichroism_nm2"),
        ("dichroism", "absorption", "cabs_dichroism_nm2"),
    ]
    for typ, crosstype, col in mapping:
        for _, row in df.iterrows():
            rows.append(
                {
                    "wavelength": row["wavelength_nm"],
                    "value": row[col],
                    "type": typ,
                    "crosstype": crosstype,
                    "cluster": cluster,
                    "hand": hand,
                    "source": source,
                }
            )
    return pd.DataFrame(rows)
