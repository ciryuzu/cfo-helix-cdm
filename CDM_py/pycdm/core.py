"""Core coupled-dipole matrix and solve."""

from __future__ import annotations

import numpy as np


def green_interaction_block(k: complex, rij_nm: np.ndarray) -> np.ndarray:
    """Return the off-diagonal interaction block used by CoupledDipole.jl.

    This is the block A_ij in F_ij = A_ij alpha_j. It equals the negative of
    the more common electric dyadic Green tensor convention.
    """
    r = float(np.linalg.norm(rij_nm))
    if r == 0:
        raise ValueError("overlapping dipoles produce singular interaction")
    n = np.asarray(rij_nm, dtype=float) / r
    nn = np.outer(n, n)
    eye = np.eye(3)
    kr = k * r
    return k**2 * np.exp(1j * kr) / r * ((nn - eye) + (1j / kr - 1.0 / kr**2) * (3.0 * nn - eye))


def interaction_matrix(k: complex, positions_nm: np.ndarray, alpha_blocks: np.ndarray) -> np.ndarray:
    n = positions_nm.shape[0]
    F = np.eye(3 * n, dtype=complex)
    for i in range(n):
        ii = slice(3 * i, 3 * i + 3)
        for j in range(i + 1, n):
            jj = slice(3 * j, 3 * j + 3)
            Aij = green_interaction_block(k, positions_nm[i] - positions_nm[j])
            F[ii, jj] = Aij @ alpha_blocks[j]
            F[jj, ii] = Aij.T @ alpha_blocks[i]
    return F


def solve_fields(F: np.ndarray, incident_field: np.ndarray) -> np.ndarray:
    return np.linalg.solve(F, incident_field)


def polarization(total_field: np.ndarray, alpha_blocks: np.ndarray) -> np.ndarray:
    p = np.empty_like(total_field, dtype=complex)
    for i, alpha in enumerate(alpha_blocks):
        ii = slice(3 * i, 3 * i + 3)
        p[ii] = alpha @ total_field[ii]
    return p
