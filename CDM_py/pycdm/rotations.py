"""Rotation helpers.

The package uses local-to-lab rotation matrices: the columns of R are the
particle local x, y, z axes expressed in the lab frame.
"""

from __future__ import annotations

import numpy as np


def unit_vector(v: np.ndarray, *, fallback: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    nrm = np.linalg.norm(arr)
    if nrm > 0:
        return arr / nrm
    if fallback is None:
        raise ValueError("zero vector cannot be normalized")
    return unit_vector(fallback)


def rotation_about_axis(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = unit_vector(axis)
    x, y, z = axis
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=float,
    )


def frame_from_z_axis(axis: np.ndarray, twist_rad: float = 0.0) -> np.ndarray:
    """Return local-to-lab frame whose local z axis points along `axis`."""
    z_axis = unit_vector(axis, fallback=np.array([0.0, 0.0, 1.0]))
    if abs(np.dot(z_axis, np.array([0.0, 0.0, 1.0]))) < 0.9:
        reference = np.array([0.0, 0.0, 1.0])
    else:
        reference = np.array([1.0, 0.0, 0.0])
    x_axis = unit_vector(np.cross(reference, z_axis))
    y_axis = np.cross(z_axis, x_axis)
    frame = np.column_stack([x_axis, y_axis, z_axis])
    if twist_rad:
        frame = rotation_about_axis(z_axis, twist_rad) @ frame
    return frame


def identity_rotations(n: int) -> np.ndarray:
    return np.repeat(np.eye(3, dtype=float)[None, :, :], n, axis=0)


def euler_zyz_active(phi: float, theta: float, psi: float = 0.0) -> np.ndarray:
    """Active ZYZ rotation matching the convention used by CoupledDipole.jl."""
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    ctheta = np.cos(theta)
    stheta = np.sin(theta)
    cpsi = np.cos(psi)
    spsi = np.sin(psi)
    return np.array(
        [
            [
                cphi * ctheta * cpsi - sphi * spsi,
                -cphi * ctheta * spsi - sphi * cpsi,
                cphi * stheta,
            ],
            [
                sphi * ctheta * cpsi + cphi * spsi,
                -sphi * ctheta * spsi + cphi * cpsi,
                sphi * stheta,
            ],
            [-stheta * cpsi, stheta * spsi, ctheta],
        ],
        dtype=float,
    )
