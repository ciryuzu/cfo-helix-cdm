"""Cross-section observables."""

from __future__ import annotations

import numpy as np


def extinction(k: complex, dipoles: np.ndarray, incident_field: np.ndarray) -> float:
    return float(4.0 * np.pi * k.real * np.imag(np.vdot(incident_field, dipoles)))


def absorption(k: complex, dipoles: np.ndarray, total_field: np.ndarray) -> float:
    term = np.imag(np.vdot(total_field, dipoles)) - (2.0 / 3.0) * k.real**3 * np.real(np.vdot(dipoles, dipoles))
    return float(4.0 * np.pi * k.real * term)
