"""Incident fields."""

from __future__ import annotations

import numpy as np

from .rotations import unit_vector


def transverse_basis(k_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k_hat = unit_vector(k_hat)
    if abs(k_hat[2]) > 0.9:
        e1 = np.array([1.0, 0.0, 0.0])
    else:
        e1 = unit_vector(np.cross(np.array([0.0, 0.0, 1.0]), k_hat))
    e2 = unit_vector(np.cross(k_hat, e1))
    return e1, e2


def circular_polarization(k_hat: np.ndarray, handedness: str) -> np.ndarray:
    e1, e2 = transverse_basis(k_hat)
    if handedness.lower().startswith("l"):
        return (e1 + 1j * e2) / np.sqrt(2.0)
    if handedness.lower().startswith("r"):
        return (e1 - 1j * e2) / np.sqrt(2.0)
    raise ValueError("handedness must start with 'left' or 'right'")


def plane_wave_field(
    positions_nm: np.ndarray,
    k: complex,
    *,
    k_hat: np.ndarray = np.array([0.0, 0.0, 1.0]),
    polarization: np.ndarray | None = None,
    handedness: str | None = None,
) -> np.ndarray:
    k_hat = unit_vector(k_hat)
    if polarization is None:
        if handedness is None:
            polarization = transverse_basis(k_hat)[0].astype(complex)
        else:
            polarization = circular_polarization(k_hat, handedness)
    polarization = np.asarray(polarization, dtype=complex)
    phases = np.exp(1j * k * positions_nm @ k_hat)
    field = phases[:, None] * polarization[None, :]
    return field.reshape(-1)
