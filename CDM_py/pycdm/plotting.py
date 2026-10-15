"""Plotting helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_circular_spectrum(df: pd.DataFrame, path: str | Path | None = None):
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True)
    axes[0].plot(df["wavelength_nm"], df["cext_avg_nm2"], label="avg extinction", color="#2563eb")
    axes[0].plot(df["wavelength_nm"], df["cabs_avg_nm2"], label="avg absorption", color="#059669")
    axes[0].set_ylabel("cross section / nm$^2$")
    axes[0].legend()
    axes[1].plot(df["wavelength_nm"], df["cd_ext_nm2"], label="extinction CD", color="#dc2626")
    axes[1].plot(df["wavelength_nm"], df["cd_abs_nm2"], label="absorption CD", color="#7c3aed")
    axes[1].axhline(0.0, color="0.6", lw=0.8)
    axes[1].set_xlabel("wavelength / nm")
    axes[1].set_ylabel("L - R / nm$^2$")
    axes[1].legend()
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=220)
    return fig


def plot_geometry_xy(particles, path: str | Path | None = None):
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.scatter(particles.positions_nm[:, 0], particles.positions_nm[:, 1], s=24)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x / nm")
    ax.set_ylabel("y / nm")
    ax.set_title(f"{len(particles)} particles, xy projection")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=220)
    return fig


def plot_geometry_3d(particles, path: str | Path | None = None, *, color_by_z: bool = True):
    fig = plt.figure(figsize=(6.2, 5.4))
    ax = fig.add_subplot(111, projection="3d")
    c = particles.positions_nm[:, 2] if color_by_z else "#d97706"
    ax.scatter(
        particles.positions_nm[:, 0],
        particles.positions_nm[:, 1],
        particles.positions_nm[:, 2],
        c=c,
        cmap="viridis" if color_by_z else None,
        s=26,
        depthshade=True,
    )
    ax.set_xlabel("x / nm")
    ax.set_ylabel("y / nm")
    ax.set_zlabel("z / nm")
    ax.set_title(f"{len(particles)} CFO particles")
    spans = particles.positions_nm.max(axis=0) - particles.positions_nm.min(axis=0)
    centers = particles.positions_nm.mean(axis=0)
    radius = max(float(spans.max()) / 2.0, 1.0)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
    ax.view_init(elev=24, azim=-55)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=220)
    return fig
