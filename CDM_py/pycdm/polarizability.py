"""Particle polarizability prescriptions."""

from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy.integrate import quad
from scipy.special import spherical_jn, spherical_yn

from .geometry import ParticleSet


def depolarization_spheroid(a: float, c: float) -> np.ndarray:
    if np.isclose(a, c):
        lz = 1.0 / 3.0
    elif c > a:
        e = np.sqrt(1.0 - (a / c) ** 2)
        lz = (1.0 - e**2) / e**3 * (-e + np.arctanh(e))
    else:
        e = np.sqrt(1.0 - (c / a) ** 2)
        lz = 1.0 / e**2 * (1.0 - np.sqrt(1.0 - e**2) / e * np.arcsin(e))
    lx = (1.0 - lz) / 2.0
    return np.array([lx, lx, lz], dtype=float)


def depolarization_ellipsoid(a: float, b: float, c: float) -> np.ndarray:
    if np.isclose(a, b) and np.isclose(b, c):
        return np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    if np.isclose(a, b):
        return depolarization_spheroid(a, c)
    if np.isclose(b, c):
        lz, ly, lx = depolarization_spheroid(b, a)
        return np.array([lx, ly, lz])
    if np.isclose(a, c):
        lx, lz, ly = depolarization_spheroid(a, b)
        return np.array([lx, ly, lz])

    volume_factor = a * b * c / 2.0

    def integral(ai: float, aj: float, ak: float) -> float:
        def f(q: float) -> float:
            return 1.0 / ((q + ai * ai) * np.sqrt((q + a * a) * (q + b * b) * (q + c * c)))

        return volume_factor * quad(f, 0.0, np.inf, epsabs=1e-10, epsrel=1e-8, limit=200)[0]

    lx = integral(a, b, c)
    ly = integral(b, a, c)
    lz = integral(c, a, b)
    return np.array([lx, ly, lz], dtype=float)


def alpha_rayleigh(lambda_nm: float, epsilon: complex, eps_medium: complex, size_nm: np.ndarray) -> np.ndarray:
    a, b, c = np.asarray(size_nm, dtype=float)
    volume = 4.0 * np.pi / 3.0 * a * b * c
    lfac = depolarization_ellipsoid(a, b, c)
    return volume / (4.0 * np.pi) * (epsilon - eps_medium) / (eps_medium + lfac * (epsilon - eps_medium))


def alpha_kuwata(lambda_nm: float, epsilon: complex, eps_medium: complex, size_nm: np.ndarray) -> np.ndarray:
    a, b, c = np.asarray(size_nm, dtype=float)
    volume = 4.0 * np.pi / 3.0 * a * b * c
    x = 2.0 * np.pi / lambda_nm * np.array([a, b, c], dtype=float)
    lfac = depolarization_ellipsoid(a, b, c)
    A = -0.4865 * lfac - 1.046 * lfac**2 + 0.8481 * lfac**3
    B = 0.01909 * lfac + 0.19999 * lfac**2 + 0.6077 * lfac**3
    denom = (
        lfac
        + eps_medium / (epsilon - eps_medium)
        + A * eps_medium * x**2
        + B * eps_medium**2 * x**4
        - 1j / 3.0 * 4.0 * np.pi**2 * eps_medium ** (3.0 / 2.0) * volume / lambda_nm**3
    )
    return volume / (4.0 * np.pi) / denom


def _riccati_bessel(n: int, z: complex) -> tuple[complex, complex, complex, complex]:
    jn = spherical_jn(n, z)
    yn = spherical_yn(n, z)
    hn = jn + 1j * yn
    psi = z * jn
    xi = z * hn
    dpsi = z * spherical_jn(n - 1, z) - n * spherical_jn(n, z) if n > 0 else np.sin(z)
    hprev = spherical_jn(n - 1, z) + 1j * spherical_yn(n - 1, z)
    dxi = z * hprev - n * hn if n > 0 else np.cos(z) + 1j * np.sin(z)
    return psi, xi, dpsi, dxi


def alpha_mie(lambda_nm: float, epsilon: complex, eps_medium: complex, size_nm: np.ndarray) -> np.ndarray:
    radius = float(np.mean(size_nm))
    n_medium = np.sqrt(eps_medium)
    s = np.sqrt(epsilon) / n_medium
    k = n_medium * 2.0 * np.pi / lambda_nm
    x = k * radius
    z = s * x
    psi_x, xi_x, dpsi_x, dxi_x = _riccati_bessel(1, x)
    psi_z, _xi_z, dpsi_z, _dxi_z = _riccati_bessel(1, z)
    pp1 = psi_z * dpsi_x
    pp2 = psi_x * dpsi_z
    pp3 = psi_z * dxi_x
    pp4 = xi_x * dpsi_z
    delta = (pp2 - s * pp1) / (-pp4 + s * pp3)
    conversion = (2.0 / 3.0) * 1j * k**3
    alpha = delta / conversion
    return np.array([alpha, alpha, alpha], dtype=complex)


def principal_alpha(
    lambda_nm: float,
    epsilon: complex,
    eps_medium: complex,
    size_nm: np.ndarray,
    *,
    method: str = "kuwata",
) -> np.ndarray:
    method = method.lower()
    if method in {"rayleigh", "cm", "static"}:
        return alpha_rayleigh(lambda_nm, epsilon, eps_medium, size_nm)
    if method == "kuwata":
        return alpha_kuwata(lambda_nm, epsilon, eps_medium, size_nm)
    if method == "mie":
        return alpha_mie(lambda_nm, epsilon, eps_medium, size_nm)
    raise ValueError(f"unknown polarizability method: {method}")


def build_alpha_blocks(
    particles: ParticleSet,
    epsilon_functions: Mapping[str, object],
    lambda_nm: float,
    *,
    eps_medium: complex,
    method: str = "kuwata",
) -> np.ndarray:
    """Build lab-frame polarizability tensors with shape `(N, 3, 3)`."""
    blocks = np.empty((len(particles), 3, 3), dtype=complex)
    for i, material in enumerate(particles.materials):
        eps_fn = epsilon_functions[material]
        epsilon = complex(eps_fn(lambda_nm))  # type: ignore[operator]
        alpha_p = principal_alpha(lambda_nm, epsilon, eps_medium, particles.sizes_nm[i], method=method)
        R = particles.rotations[i]
        blocks[i] = R @ np.diag(alpha_p) @ R.T
    return blocks
