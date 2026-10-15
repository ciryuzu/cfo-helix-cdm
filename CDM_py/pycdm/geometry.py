"""Particle geometry builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .rotations import frame_from_z_axis, identity_rotations


@dataclass(frozen=True)
class ParticleSet:
    """Geometry and labels for a coupled-dipole calculation.

    `sizes_nm` stores ellipsoid semi-axes `(a, b, c)`. For a nanorod represented
    as an equivalent prolate spheroid, use `(radius, radius, length/2)`.
    """

    positions_nm: np.ndarray
    sizes_nm: np.ndarray
    rotations: np.ndarray
    materials: tuple[str, ...]
    types: tuple[str, ...]

    def __post_init__(self) -> None:
        n = len(self.positions_nm)
        object.__setattr__(self, "positions_nm", np.asarray(self.positions_nm, dtype=float).reshape(n, 3))
        object.__setattr__(self, "sizes_nm", np.asarray(self.sizes_nm, dtype=float).reshape(n, 3))
        object.__setattr__(self, "rotations", np.asarray(self.rotations, dtype=float).reshape(n, 3, 3))
        if len(self.materials) != n or len(self.types) != n:
            raise ValueError("materials and types must have one entry per particle")

    def __len__(self) -> int:
        return self.positions_nm.shape[0]

    def translated(self, shift_nm: Sequence[float]) -> "ParticleSet":
        shift = np.asarray(shift_nm, dtype=float)
        return ParticleSet(
            self.positions_nm + shift,
            self.sizes_nm.copy(),
            self.rotations.copy(),
            self.materials,
            self.types,
        )

    def with_material(self, material: str) -> "ParticleSet":
        return ParticleSet(
            self.positions_nm.copy(),
            self.sizes_nm.copy(),
            self.rotations.copy(),
            tuple(material for _ in range(len(self))),
            self.types,
        )


def combine_particle_sets(parts: Iterable[ParticleSet]) -> ParticleSet:
    parts = tuple(parts)
    if not parts:
        raise ValueError("cannot combine an empty list of ParticleSet objects")
    return ParticleSet(
        np.vstack([p.positions_nm for p in parts]),
        np.vstack([p.sizes_nm for p in parts]),
        np.vstack([p.rotations for p in parts]),
        tuple(m for p in parts for m in p.materials),
        tuple(t for p in parts for t in p.types),
    )


def single_particle(
    semiaxes_nm: Sequence[float],
    *,
    position_nm: Sequence[float] = (0.0, 0.0, 0.0),
    axis: Sequence[float] = (0.0, 0.0, 1.0),
    twist_rad: float = 0.0,
    material: str = "CFO",
    particle_type: str = "particle",
) -> ParticleSet:
    return ParticleSet(
        np.asarray(position_nm, dtype=float)[None, :],
        np.asarray(semiaxes_nm, dtype=float)[None, :],
        frame_from_z_axis(np.asarray(axis, dtype=float), twist_rad)[None, :, :],
        (material,),
        (particle_type,),
    )


def helix_particles(
    n_particles: int,
    semiaxes_nm: Sequence[float],
    helix_radius_nm: float,
    pitch_nm: float,
    angle_step_rad: float,
    *,
    start_angle_rad: float = 0.0,
    handedness: str = "left",
    orientation: str = "tangent",
    material: str = "CFO",
    particle_type: str = "particle",
) -> ParticleSet:
    """Build a helical cluster with the helix axis along lab z."""
    if n_particles < 1:
        raise ValueError("n_particles must be positive")
    sign = -1.0 if handedness.lower().startswith("l") else 1.0
    idx = np.arange(1, n_particles + 1, dtype=float)
    theta = start_angle_rad + idx * angle_step_rad
    x = helix_radius_nm * np.cos(theta)
    y = helix_radius_nm * np.sin(theta)
    z = sign * theta * pitch_nm / (2.0 * np.pi)
    z = z - sign * np.max(sign * z) / 2.0
    positions = np.column_stack([x, y, z])

    rotations = []
    for th in theta:
        radial = np.array([np.cos(th), np.sin(th), 0.0])
        tangent = np.array(
            [
                -helix_radius_nm * np.sin(th),
                helix_radius_nm * np.cos(th),
                sign * pitch_nm / (2.0 * np.pi),
            ]
        )
        if orientation == "tangent":
            axis = tangent
        elif orientation == "radial":
            axis = radial
        elif orientation == "z":
            axis = np.array([0.0, 0.0, 1.0])
        else:
            raise ValueError("orientation must be 'tangent', 'radial', or 'z'")
        rotations.append(frame_from_z_axis(axis))

    return ParticleSet(
        positions,
        np.repeat(np.asarray(semiaxes_nm, dtype=float)[None, :], n_particles, axis=0),
        np.stack(rotations),
        tuple(material for _ in range(n_particles)),
        tuple(particle_type for _ in range(n_particles)),
    )


def array_of_clusters(
    base: ParticleSet,
    shape: tuple[int, int] = (1, 1),
    pitch_nm: tuple[float, float] = (250.0, 250.0),
    *,
    center: bool = True,
) -> ParticleSet:
    nx, ny = shape
    px, py = pitch_nm
    parts = []
    x0 = -(nx - 1) * px / 2.0 if center else 0.0
    y0 = -(ny - 1) * py / 2.0 if center else 0.0
    for ix in range(nx):
        for iy in range(ny):
            parts.append(base.translated((x0 + ix * px, y0 + iy * py, 0.0)))
    return combine_particle_sets(parts)


def helix_nanorods(
    *,
    rod_radius_nm: float,
    rod_length_nm: float,
    helix_radius_nm: float,
    pitch_nm: float,
    rods_per_turn: int,
    turns: float,
    handedness: str = "left",
    array_shape: tuple[int, int] = (1, 1),
    array_pitch_nm: tuple[float, float] = (250.0, 250.0),
    material: str = "CFO",
) -> ParticleSet:
    n_rods = max(1, int(round(rods_per_turn * turns)))
    angle_step = 2.0 * np.pi / rods_per_turn
    base = helix_particles(
        n_rods,
        (rod_radius_nm, rod_radius_nm, rod_length_nm / 2.0),
        helix_radius_nm,
        pitch_nm,
        angle_step,
        handedness=handedness,
        orientation="tangent",
        material=material,
    )
    return array_of_clusters(base, array_shape, array_pitch_nm)


def isotropic_rotations(n: int) -> np.ndarray:
    return identity_rotations(n)
