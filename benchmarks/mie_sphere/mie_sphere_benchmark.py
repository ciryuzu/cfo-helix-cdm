"""Exact Mie benchmark for a homogeneous isotropic sphere.

The bundled default input is synthetic teaching data, not measured CFO data.
The dielectric table uses exp(-i omega t), epsilon = epsilon_1 + i epsilon_2;
miepython uses m = n - i kappa, so conjugation is applied at its API boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import miepython
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
STAGE_DIR = SCRIPT_DIR
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "CDM_py"))

from pycdm.polarizability import principal_alpha


DEFAULT_INPUT = PROJECT_ROOT / "data" / "synthetic_epsilon.csv"
DEFAULT_RADII_NM = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)
PRIMARY_RADIUS_NM = 4.0
PRIMARY_LABELS = ("E1", "M1", "E2", "M2")


def positive_branch_sqrt(epsilon: complex) -> complex:
    """Return n+i*kappa for passive epsilon under exp(-i omega t)."""
    value = complex(np.sqrt(epsilon + 0j))
    if value.real < 0:
        value = -value
    if value.imag < -1e-14:
        raise ValueError(f"passive epsilon produced negative kappa: {epsilon!r}")
    return value


def cdm_cross_sections(
    wavelength_nm: float,
    epsilon: complex,
    radius_nm: float,
    n_medium: float,
    method: str,
) -> tuple[float, float, float]:
    """Evaluate the existing CDM single-sphere cross-section convention."""
    k = 2.0 * np.pi * n_medium / wavelength_nm
    alpha = principal_alpha(
        wavelength_nm,
        epsilon,
        n_medium**2,
        np.array([radius_nm, radius_nm, radius_nm]),
        method=method,
    )[0]
    cext = 4.0 * np.pi * k * np.imag(alpha)
    csca = 4.0 * np.pi * k * (2.0 / 3.0) * k**3 * abs(alpha) ** 2
    cabs = cext - csca
    return float(cext), float(csca), float(cabs)


def mie_one(
    wavelength_nm: float,
    epsilon: complex,
    radius_nm: float,
    n_medium: float,
) -> tuple[dict[str, float], list[dict[str, float | int | str]]]:
    """Return exact totals and electric/magnetic contributions for one point."""
    n_particle = positive_branch_sqrt(epsilon)
    k_medium = 2.0 * np.pi * n_medium / wavelength_nm
    x = k_medium * radius_nm

    # miepython represents passive loss with a negative imaginary index.
    m_relative_mie = np.conj(n_particle / n_medium)
    a_coeff, b_coeff = miepython.coefficients(m_relative_mie, x)
    orders = np.arange(1, len(a_coeff) + 1, dtype=int)
    area_nm2 = np.pi * radius_nm**2
    q_prefactor = 2.0 / x**2

    rows: list[dict[str, float | int | str]] = []
    for kind, coeffs in (("E", a_coeff), ("M", b_coeff)):
        for order, coefficient in zip(orders, coeffs):
            weight = q_prefactor * (2 * order + 1)
            cext = area_nm2 * weight * np.real(coefficient)
            csca = area_nm2 * weight * abs(coefficient) ** 2
            rows.append(
                {
                    "radius_nm": radius_nm,
                    "wavelength_nm": wavelength_nm,
                    "order": int(order),
                    "kind": kind,
                    "label": f"{kind}{order}",
                    "coefficient_re": float(np.real(coefficient)),
                    "coefficient_im": float(np.imag(coefficient)),
                    "cext_nm2": float(cext),
                    "csca_nm2": float(csca),
                    "cabs_nm2": float(cext - csca),
                }
            )

    cext_sum = float(sum(row["cext_nm2"] for row in rows))
    csca_sum = float(sum(row["csca_nm2"] for row in rows))
    cabs_sum = float(sum(row["cabs_nm2"] for row in rows))
    alpha_e1 = 3j * a_coeff[0] / (2.0 * k_medium**3)

    qext_lib, qsca_lib, _qback, _g = miepython.efficiencies(
        np.conj(n_particle),
        2.0 * radius_nm,
        wavelength_nm,
        n_env=n_medium,
    )
    cext_lib = float(area_nm2 * qext_lib)
    csca_lib = float(area_nm2 * qsca_lib)

    total = {
        "radius_nm": radius_nm,
        "wavelength_nm": wavelength_nm,
        "energy_eV": 1239.841984 / wavelength_nm,
        "eps1": float(np.real(epsilon)),
        "eps2": float(np.imag(epsilon)),
        "n_particle": float(np.real(n_particle)),
        "kappa_particle": float(np.imag(n_particle)),
        "n_medium": n_medium,
        "size_parameter_x": x,
        "n_mie_terms": len(a_coeff),
        "alpha_E1_re_nm3": float(np.real(alpha_e1)),
        "alpha_E1_im_nm3": float(np.imag(alpha_e1)),
        "alpha_E1_abs_nm3": float(abs(alpha_e1)),
        "cext_mie_nm2": cext_sum,
        "csca_mie_nm2": csca_sum,
        "cabs_mie_nm2": cabs_sum,
        "cext_library_nm2": cext_lib,
        "csca_library_nm2": csca_lib,
    }
    return total, rows


def add_channel_columns(totals: pd.DataFrame, multipoles: pd.DataFrame) -> pd.DataFrame:
    out = totals.copy()
    keys = ["radius_nm", "wavelength_nm"]
    for label in PRIMARY_LABELS:
        subset = multipoles.loc[multipoles["label"] == label, keys + ["cext_nm2", "csca_nm2", "cabs_nm2"]]
        subset = subset.rename(
            columns={
                "cext_nm2": f"{label}_cext_nm2",
                "csca_nm2": f"{label}_csca_nm2",
                "cabs_nm2": f"{label}_cabs_nm2",
            }
        )
        out = out.merge(subset, on=keys, how="left", validate="one_to_one")

    for cross_type in ("cext", "csca", "cabs"):
        total_col = f"{cross_type}_mie_nm2"
        primary_cols = [f"{label}_{cross_type}_nm2" for label in PRIMARY_LABELS]
        out[f"higher_{cross_type}_nm2"] = out[total_col] - out[primary_cols].sum(axis=1)
        for label in PRIMARY_LABELS:
            out[f"{label}_{cross_type}_fraction"] = np.divide(
                out[f"{label}_{cross_type}_nm2"],
                out[total_col],
                out=np.full(len(out), np.nan),
                where=np.abs(out[total_col]) > 1e-30,
            )

    out["scattering_share_of_extinction"] = np.divide(
        out["csca_mie_nm2"],
        out["cext_mie_nm2"],
        out=np.full(len(out), np.nan),
        where=np.abs(out["cext_mie_nm2"]) > 1e-30,
    )
    out["energy_residual_nm2"] = out["cext_mie_nm2"] - out["csca_mie_nm2"] - out["cabs_mie_nm2"]
    return out


def add_cdm_comparisons(totals: pd.DataFrame) -> pd.DataFrame:
    out = totals.copy()
    methods = ("rayleigh", "kuwata", "mie")
    for method in methods:
        values = [
            cdm_cross_sections(
                float(row.wavelength_nm),
                complex(row.eps1, row.eps2),
                float(row.radius_nm),
                float(row.n_medium),
                method,
            )
            for row in out.itertuples(index=False)
        ]
        out[f"cdm_{method}_cext_nm2"] = [value[0] for value in values]
        out[f"cdm_{method}_csca_nm2"] = [value[1] for value in values]
        out[f"cdm_{method}_cabs_nm2"] = [value[2] for value in values]
    return out


def signal_mask(frame: pd.DataFrame) -> np.ndarray:
    threshold = 0.01 * float(frame["cext_mie_nm2"].max())
    return frame["cext_mie_nm2"].to_numpy(float) >= threshold


def max_relative_error(actual: np.ndarray, reference: np.ndarray, mask: np.ndarray | None = None) -> float:
    actual = np.asarray(actual, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if mask is None:
        mask = np.ones(reference.shape, dtype=bool)
    scale = np.maximum(np.abs(reference), 1e-30)
    return float(np.max(np.abs(actual[mask] - reference[mask]) / scale[mask]))


def build_radius_summary(totals: pd.DataFrame, multipoles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for radius, frame in totals.groupby("radius_nm", sort=True):
        frame = frame.sort_values("wavelength_nm").reset_index(drop=True)
        mask = signal_mask(frame)
        peak_ext = frame.loc[frame["cext_mie_nm2"].idxmax()]
        peak_abs = frame.loc[frame["cabs_mie_nm2"].idxmax()]
        peak_sca = frame.loc[frame["csca_mie_nm2"].idxmax()]

        selected = multipoles.loc[multipoles["radius_nm"] == radius]
        cumulative_l3 = (
            selected.loc[selected["order"] <= 3]
            .groupby("wavelength_nm", as_index=False)[["cext_nm2", "csca_nm2"]]
            .sum()
            .set_index("wavelength_nm")
            .reindex(frame["wavelength_nm"])
        )
        rows.append(
            {
                "radius_nm": radius,
                "max_size_parameter_x": float(frame["size_parameter_x"].max()),
                "peak_extinction_wavelength_nm": float(peak_ext["wavelength_nm"]),
                "peak_extinction_nm2": float(peak_ext["cext_mie_nm2"]),
                "peak_absorption_wavelength_nm": float(peak_abs["wavelength_nm"]),
                "peak_absorption_nm2": float(peak_abs["cabs_mie_nm2"]),
                "peak_scattering_wavelength_nm": float(peak_sca["wavelength_nm"]),
                "peak_scattering_nm2": float(peak_sca["csca_mie_nm2"]),
                "max_scattering_share_signal_region": float(np.nanmax(frame.loc[mask, "scattering_share_of_extinction"])),
                "max_non_E1_ext_fraction_signal_region": float(
                    np.nanmax(1.0 - frame.loc[mask, "E1_cext_fraction"])
                ),
                "max_M1_ext_fraction_signal_region": float(np.nanmax(frame.loc[mask, "M1_cext_fraction"])),
                "max_E2_ext_fraction_signal_region": float(np.nanmax(frame.loc[mask, "E2_cext_fraction"])),
                "max_l3_ext_truncation_error_signal_region": max_relative_error(
                    cumulative_l3["cext_nm2"].to_numpy(float),
                    frame["cext_mie_nm2"].to_numpy(float),
                    mask,
                ),
                "max_l3_sca_truncation_error_signal_region": max_relative_error(
                    cumulative_l3["csca_nm2"].to_numpy(float),
                    frame["csca_mie_nm2"].to_numpy(float),
                    mask,
                ),
                "cdm_rayleigh_max_ext_rel_error_signal_region": max_relative_error(
                    frame["cdm_rayleigh_cext_nm2"], frame["cext_mie_nm2"], mask
                ),
                "cdm_kuwata_max_ext_rel_error_signal_region": max_relative_error(
                    frame["cdm_kuwata_cext_nm2"], frame["cext_mie_nm2"], mask
                ),
                "cdm_electric_mie_max_ext_rel_error_signal_region": max_relative_error(
                    frame["cdm_mie_cext_nm2"], frame["cext_mie_nm2"], mask
                ),
            }
        )
    return pd.DataFrame(rows)


def build_validation(
    source: pd.DataFrame,
    totals: pd.DataFrame,
    radius_summary: pd.DataFrame,
) -> dict[str, object]:
    epsilon_values = source["eps1"].to_numpy(float) + 1j * source["eps2"].to_numpy(float)
    refractive_index = np.sqrt(epsilon_values)
    index_n_error = float(np.max(np.abs(refractive_index.real - source["n"].to_numpy(float))))
    index_k_error = float(np.max(np.abs(refractive_index.imag - source["k"].to_numpy(float))))
    validation_mask = np.zeros(len(totals), dtype=bool)
    for _radius, frame in totals.groupby("radius_nm"):
        validation_mask[frame.index.to_numpy()] = signal_mask(frame)
    lib_ext_error_all = max_relative_error(totals["cext_mie_nm2"], totals["cext_library_nm2"])
    lib_sca_error_all = max_relative_error(totals["csca_mie_nm2"], totals["csca_library_nm2"])
    lib_ext_error_signal = max_relative_error(
        totals["cext_mie_nm2"], totals["cext_library_nm2"], validation_mask
    )
    lib_sca_error_signal = max_relative_error(
        totals["csca_mie_nm2"], totals["csca_library_nm2"], validation_mask
    )
    lib_ext_abs_error = float(np.max(np.abs(totals["cext_mie_nm2"] - totals["cext_library_nm2"])))
    lib_sca_abs_error = float(np.max(np.abs(totals["csca_mie_nm2"] - totals["csca_library_nm2"])))
    energy_scale = np.maximum(np.abs(totals["cext_mie_nm2"].to_numpy(float)), 1e-30)
    energy_error = float(np.max(np.abs(totals["energy_residual_nm2"].to_numpy(float)) / energy_scale))
    min_cross_section = float(
        totals[["cext_mie_nm2", "csca_mie_nm2", "cabs_mie_nm2"]].to_numpy(float).min()
    )
    peak_cross_section = float(totals["cext_mie_nm2"].max())
    min_cross_section_relative_to_peak = min_cross_section / peak_cross_section
    cdm_e1_error_signal = max_relative_error(
        totals["cdm_mie_cext_nm2"], totals["E1_cext_nm2"], validation_mask
    )
    max_l3_ext = float(radius_summary["max_l3_ext_truncation_error_signal_region"].max())
    max_l3_sca = float(radius_summary["max_l3_sca_truncation_error_signal_region"].max())

    checks = {
        "input_refractive_index_consistency": max(index_n_error, index_k_error) < 1e-10,
        "coefficient_sum_matches_library_total_in_signal_region": max(
            lib_ext_error_signal, lib_sca_error_signal
        ) < 1e-5,
        "energy_identity": energy_error < 1e-12,
        "passive_cross_sections_within_numerical_floor": min_cross_section_relative_to_peak >= -1e-10,
        "multipole_lmax_3_converged_for_scan": max(max_l3_ext, max_l3_sca) < 1e-6,
        "existing_cdm_mie_matches_exact_E1_in_signal_region": cdm_e1_error_signal < 1e-6,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "metrics": {
            "input_n_max_abs_error": index_n_error,
            "input_kappa_max_abs_error": index_k_error,
            "coefficient_vs_library_cext_max_rel_error_all_points": lib_ext_error_all,
            "coefficient_vs_library_csca_max_rel_error_all_points": lib_sca_error_all,
            "coefficient_vs_library_cext_max_rel_error_signal_region": lib_ext_error_signal,
            "coefficient_vs_library_csca_max_rel_error_signal_region": lib_sca_error_signal,
            "coefficient_vs_library_cext_max_abs_error_nm2": lib_ext_abs_error,
            "coefficient_vs_library_csca_max_abs_error_nm2": lib_sca_abs_error,
            "energy_max_relative_residual": energy_error,
            "minimum_total_cross_section_nm2": min_cross_section,
            "minimum_cross_section_relative_to_global_peak": min_cross_section_relative_to_peak,
            "lmax3_ext_max_relative_error_signal_region": max_l3_ext,
            "lmax3_sca_max_relative_error_signal_region": max_l3_sca,
            "existing_cdm_mie_vs_exact_E1_max_rel_error_signal_region": cdm_e1_error_signal,
        },
        "conventions": {
            "time_dependence": "exp(-i omega t)",
            "passive_epsilon": "epsilon1 + i epsilon2, epsilon2 >= 0",
            "miepython_index": "n - i kappa",
            "length_unit": "nm",
            "cross_section_unit": "nm^2",
        },
        "software": {
            "python": sys.version.split()[0],
            "miepython": miepython.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }


def plot_primary(frame: pd.DataFrame, output: Path) -> None:
    frame = frame.sort_values("wavelength_nm")
    wavelength = frame["wavelength_nm"]
    positive_floor = max(float(frame["cext_mie_nm2"].max()) * 1e-12, 1e-30)
    reliable = signal_mask(frame)
    reliable_wavelength = wavelength.where(reliable)
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.0), sharex=True)

    axes[0].semilogy(wavelength, np.maximum(frame["cext_mie_nm2"], positive_floor), label="Extinction", lw=1.8)
    axes[0].semilogy(wavelength, np.maximum(frame["cabs_mie_nm2"], positive_floor), label="Absorption", lw=1.5)
    axes[0].semilogy(wavelength, np.maximum(frame["csca_mie_nm2"], positive_floor), label="Scattering", lw=1.5)
    axes[0].set_ylabel("Cross section / nm$^2$")
    axes[0].legend(frameon=False, ncol=3, fontsize=8)

    axes[1].semilogy(
        reliable_wavelength,
        np.maximum(frame["M1_cext_fraction"].where(reliable), 1e-16),
        label="M1 extinction fraction",
        lw=1.3,
    )
    axes[1].semilogy(
        reliable_wavelength,
        np.maximum(frame["E2_cext_fraction"].where(reliable), 1e-16),
        label="E2 extinction fraction",
        lw=1.3,
    )
    axes[1].semilogy(
        reliable_wavelength,
        np.maximum((1.0 - frame["E1_cext_fraction"]).where(reliable), 1e-16),
        label="All non-E1 extinction",
        lw=1.3,
        ls="--",
    )
    axes[1].semilogy(
        reliable_wavelength,
        np.maximum(frame["scattering_share_of_extinction"].where(reliable), 1e-16),
        label="Scattering / extinction",
        color="#8c2d04",
        lw=1.3,
    )
    axes[1].axhline(0.01, color="0.4", lw=0.8, ls=":", label="1% reference")
    axes[1].set_ylabel("Fraction in signal region")
    axes[1].set_xlabel("Vacuum wavelength / nm")
    axes[1].legend(frameon=False, ncol=2, fontsize=8)

    for axis in axes:
        axis.grid(True, color="0.88", lw=0.6)
    fig.suptitle(
        f"Sphere Mie benchmark: radius {frame['radius_nm'].iloc[0]:g} nm, "
        f"medium n={frame['n_medium'].iloc[0]:g}"
    )
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_radius_summary(summary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 7.0), sharex=True)
    axes[0].plot(summary["radius_nm"], summary["max_size_parameter_x"], marker="o", label="max x")
    axes[0].set_ylabel("Maximum size parameter")
    axes[0].grid(True, color="0.88", lw=0.6)

    axes[1].semilogy(
        summary["radius_nm"],
        np.maximum(summary["max_non_E1_ext_fraction_signal_region"], 1e-16),
        marker="o",
        label="non-E1 extinction fraction",
    )
    axes[1].semilogy(
        summary["radius_nm"],
        np.maximum(summary["max_scattering_share_signal_region"], 1e-16),
        marker="s",
        label="scattering / extinction",
    )
    axes[1].set_xlabel("Sphere radius / nm")
    axes[1].set_ylabel("Maximum fraction in signal region")
    axes[1].grid(True, color="0.88", lw=0.6)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Sphere size scan")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def scientific(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}e}"


def write_report(
    source_path: Path,
    source_sha256: str,
    primary: pd.DataFrame,
    radius_summary: pd.DataFrame,
    validation: dict[str, object],
    output: Path,
) -> None:
    primary = primary.sort_values("wavelength_nm").reset_index(drop=True)
    signal = signal_mask(primary)
    peak = primary.loc[primary["cext_mie_nm2"].idxmax()]
    absorption_peak = primary.loc[primary["cabs_mie_nm2"].idxmax()]
    primary_summary = radius_summary.loc[np.isclose(radius_summary["radius_nm"], PRIMARY_RADIUS_NM)].iloc[0]
    scattering_max = float(primary.loc[signal, "scattering_share_of_extinction"].max())
    input_note = (
        "Bundled synthetic teaching input; not experimental or CFO material data."
        if source_path.resolve() == DEFAULT_INPUT.resolve()
        else f"Custom input file: {source_path.name}; review its provenance separately."
    )
    text = f"""# Homogeneous-sphere Mie benchmark

This run compares an exact Mie-series calculation with electric-dipole
polarizability approximations for a homogeneous isotropic sphere. It is a
numerical code benchmark, not evidence about any particular material.

## Input and settings

- Input: {input_note}
- Input SHA-256: `{source_sha256}`
- Wavelength range: {primary['wavelength_nm'].min():g}–{primary['wavelength_nm'].max():g} nm
- Primary radius: {PRIMARY_RADIUS_NM:g} nm; scan radii: {', '.join(f'{x:g}' for x in radius_summary['radius_nm'])} nm
- Medium refractive index: {primary['n_medium'].iloc[0]:g}
- Convention: exp(-i omega t), epsilon = eps1 + i eps2 with eps2 >= 0
- Cross-section unit: nm^2

## Primary-radius summary

- Maximum size parameter: {primary_summary['max_size_parameter_x']:.5g}
- Peak extinction: {scientific(float(peak['cext_mie_nm2']))} nm^2 at {peak['wavelength_nm']:.1f} nm
- Peak absorption: {scientific(float(absorption_peak['cabs_mie_nm2']))} nm^2 at {absorption_peak['wavelength_nm']:.1f} nm
- Maximum scattering/extinction ratio in the signal region: {scattering_max:.5g}
- Maximum relative extinction error versus the exact result: Rayleigh {primary_summary['cdm_rayleigh_max_ext_rel_error_signal_region']:.3e}; Kuwata {primary_summary['cdm_kuwata_max_ext_rel_error_signal_region']:.3e}; electric-dipole Mie {primary_summary['cdm_electric_mie_max_ext_rel_error_signal_region']:.3e}

The signal region is defined as extinction at least 1% of the maximum for each
radius. The synthetic default table has positive loss throughout its tabulated
range; near-zero cross sections can still show floating-point roundoff.

## Numerical checks

- Overall validation: **{validation['status']}**
- Pass/fail checks: {', '.join(f'{name}={value}' for name, value in validation['checks'].items())}
- Maximum relative energy-balance residual: {validation['metrics']['energy_max_relative_residual']:.3e}
- Maximum truncation error at multipole order l <= 3: extinction {validation['metrics']['lmax3_ext_max_relative_error_signal_region']:.3e}; scattering {validation['metrics']['lmax3_sca_max_relative_error_signal_region']:.3e}
- Electric-dipole Mie polarizability versus exact E1 extinction error: {validation['metrics']['existing_cdm_mie_vs_exact_E1_max_rel_error_signal_region']:.3e}

This calculation assumes a homogeneous isotropic sphere in a uniform,
non-absorbing medium. It does not include particle interactions, substrates,
anisotropy, magneto-optical response, or a real material data set.
"""
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exact Mie benchmark for a homogeneous isotropic sphere.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--medium-n", type=float, default=1.33)
    parser.add_argument("--radii", type=float, nargs="+", default=list(DEFAULT_RADII_NM))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = pd.read_csv(args.input)
    required = {"wavelength_nm", "eps1", "eps2", "n", "k"}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"input is missing columns: {sorted(missing)}")
    source = source.sort_values("wavelength_nm").drop_duplicates("wavelength_nm").reset_index(drop=True)
    if (source["eps2"] < -1e-12).any():
        raise ValueError("passive exp(-i omega t) input requires eps2 >= 0")
    if PRIMARY_RADIUS_NM not in args.radii:
        raise ValueError(f"radius scan must contain the primary radius {PRIMARY_RADIUS_NM:g} nm")

    results_dir = STAGE_DIR / "results"
    figures_dir = STAGE_DIR / "figures"
    validation_dir = STAGE_DIR / "validation"
    report_dir = STAGE_DIR / "report"
    for directory in (results_dir, figures_dir, validation_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)

    totals_rows = []
    multipole_rows = []
    for radius_nm in sorted(set(args.radii)):
        for row in source.itertuples(index=False):
            total, channels = mie_one(
                float(row.wavelength_nm),
                complex(float(row.eps1), float(row.eps2)),
                float(radius_nm),
                float(args.medium_n),
            )
            totals_rows.append(total)
            multipole_rows.extend(channels)

    multipoles = pd.DataFrame(multipole_rows)
    totals = add_channel_columns(pd.DataFrame(totals_rows), multipoles)
    totals = add_cdm_comparisons(totals)
    totals = totals.sort_values(["radius_nm", "wavelength_nm"]).reset_index(drop=True)
    multipoles = multipoles.sort_values(["radius_nm", "wavelength_nm", "order", "kind"]).reset_index(drop=True)
    radius_summary = build_radius_summary(totals, multipoles)
    validation = build_validation(source, totals, radius_summary)
    validation["input"] = {
        "path": args.input.name,
        "source_class": "synthetic teaching input" if args.input.resolve() == DEFAULT_INPUT.resolve() else "user-supplied input",
        "sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
    }

    primary = totals.loc[np.isclose(totals["radius_nm"], PRIMARY_RADIUS_NM)].copy()
    primary.to_csv(results_dir / "b0_cross_sections_radius_4nm.csv", index=False, lineterminator="\n")
    totals.to_csv(results_dir / "b0_cross_sections_all_radii.csv", index=False, lineterminator="\n")
    multipoles.to_csv(results_dir / "b0_multipoles_long.csv", index=False, lineterminator="\n")
    radius_summary.to_csv(results_dir / "b0_radius_scan_summary.csv", index=False, lineterminator="\n")
    with (validation_dir / "b0_validation.json").open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(validation, ensure_ascii=False, indent=2))

    plot_primary(primary, figures_dir / "b0_4nm_overview.png")
    plot_radius_summary(radius_summary, figures_dir / "b0_radius_scan.png")
    write_report(
        args.input.resolve(),
        validation["input"]["sha256"],
        primary,
        radius_summary,
        validation,
        report_dir / "mie_sphere_benchmark.md",
    )

    artifact_paths = [
        results_dir / "b0_cross_sections_radius_4nm.csv",
        results_dir / "b0_cross_sections_all_radii.csv",
        results_dir / "b0_multipoles_long.csv",
        results_dir / "b0_radius_scan_summary.csv",
        figures_dir / "b0_4nm_overview.png",
        figures_dir / "b0_radius_scan.png",
        validation_dir / "b0_validation.json",
        report_dir / "mie_sphere_benchmark.md",
    ]
    checked_in_paths = {
        "benchmarks/mie_sphere/results/b0_radius_scan_summary.csv",
        "benchmarks/mie_sphere/figures/b0_4nm_overview.png",
        "benchmarks/mie_sphere/figures/b0_radius_scan.png",
        "benchmarks/mie_sphere/validation/b0_validation.json",
        "benchmarks/mie_sphere/report/mie_sphere_benchmark.md",
    }
    manifest = {
        "benchmark": "homogeneous isotropic sphere Mie-series comparison",
        "source_class": "synthetic teaching input" if args.input.resolve() == DEFAULT_INPUT.resolve() else "user-supplied input",
        "command": "python benchmarks/mie_sphere/mie_sphere_benchmark.py",
        "input": {
            "basename": args.input.name,
            "sha256": validation["input"]["sha256"],
        },
        "configuration": {
            "medium_refractive_index": float(args.medium_n),
            "radii_nm": sorted(set(float(radius) for radius in args.radii)),
            "primary_radius_nm": PRIMARY_RADIUS_NM,
            "wavelength_range_nm": [float(source["wavelength_nm"].min()), float(source["wavelength_nm"].max())],
        },
        "conventions": validation["conventions"],
        "software": validation["software"],
        "validation_status": validation["status"],
        "benchmark_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "artifacts": [
            {
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "checked_in": path.relative_to(PROJECT_ROOT).as_posix() in checked_in_paths,
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in artifact_paths
        ],
    }
    with (validation_dir / "b0_manifest.json").open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    print(f"B0 calculation complete: {len(totals)} wavelength-radius points")
    print(f"Validation: {validation['status']}")
    print(radius_summary.to_string(index=False))
    if validation["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
