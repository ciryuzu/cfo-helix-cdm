"""Dielectric-function inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


DielectricFn = Callable[[float | np.ndarray], complex | np.ndarray]


def constant_epsilon(epsilon: complex) -> DielectricFn:
    def eps(lambda_nm: float | np.ndarray) -> complex | np.ndarray:
        arr = np.asarray(lambda_nm)
        out = np.full(arr.shape, complex(epsilon), dtype=complex)
        return out.item() if out.shape == () else out

    return eps


def constant_refractive_index(n: complex) -> DielectricFn:
    return constant_epsilon(complex(n) ** 2)


def epsilon_au_drude(lambda_nm: float | np.ndarray) -> complex | np.ndarray:
    """Gold dielectric function copied from CoupledDipole.jl's epsilon_Au.

    Extended Drude model for the visible region. The Julia reference gives
    epsilon_Au(632.8 nm) = -11.40185 + 1.18679im.
    """
    wl = np.asarray(lambda_nm, dtype=float)
    eps_inf = 1.54
    lambda_p = 177.5
    mu_p = 14500.0
    a1 = 1.27
    lambda_1 = 470.0
    mu_p1 = 1900.0
    a2 = 1.1
    lambda_2 = 325.0
    mu_p2 = 1060.0
    phi = -np.pi / 4.0

    eps = eps_inf * (
        1.0 - 1.0 / (lambda_p**2 * ((1.0 / wl) ** 2 + 1j / (mu_p * wl)))
    )
    eps += a1 / lambda_1 * (
        np.exp(1j * phi) / (1.0 / lambda_1 - 1.0 / wl - 1j / mu_p1)
        + np.exp(-1j * phi) / (1.0 / lambda_1 + 1.0 / wl + 1j / mu_p1)
    )
    eps += a2 / lambda_2 * (
        np.exp(1j * phi) / (1.0 / lambda_2 - 1.0 / wl - 1j / mu_p2)
        + np.exp(-1j * phi) / (1.0 / lambda_2 + 1.0 / wl + 1j / mu_p2)
    )
    return eps.item() if eps.shape == () else eps


def load_tabulated_epsilon(
    path: str | Path,
    *,
    wavelength_col: str = "wavelength_nm",
    eps1_col: str = "eps1",
    eps2_col: str = "eps2",
    clamp: bool = True,
) -> DielectricFn:
    """Load a CSV table and return a linear-interpolation epsilon function."""
    df = pd.read_csv(path)
    wl = df[wavelength_col].to_numpy(float)
    eps = df[eps1_col].to_numpy(float) + 1j * df[eps2_col].to_numpy(float)
    order = np.argsort(wl)
    wl = wl[order]
    eps = eps[order]

    def epsilon(lambda_nm: float | np.ndarray) -> complex | np.ndarray:
        x = np.asarray(lambda_nm, dtype=float)
        if not clamp and (np.any(x < wl[0]) or np.any(x > wl[-1])):
            raise ValueError(f"wavelength outside tabulated range {wl[0]}-{wl[-1]} nm")
        xc = np.clip(x, wl[0], wl[-1]) if clamp else x
        real = np.interp(xc, wl, eps.real)
        imag = np.interp(xc, wl, eps.imag)
        out = real + 1j * imag
        return out.item() if out.shape == () else out

    epsilon.wavelength_range_nm = (float(wl[0]), float(wl[-1]))  # type: ignore[attr-defined]
    epsilon.source_path = str(path)  # type: ignore[attr-defined]
    return epsilon


def medium_n_from_epsilon(eps_medium: complex) -> complex:
    n = np.sqrt(complex(eps_medium))
    return n if n.real >= 0 else -n
