"""Entanglement measures for two-qubit pure states.

Three standard quantities are computed from the same Schmidt decomposition:

* von Neumann entropy of one half, ``S = -sum_i s_i^2 log2 s_i^2``
* concurrence, ``C = 2 s_0 s_1`` for a pure state
* negativity, ``(||rho^{T_B}||_1 - 1) / 2``

All three are 0 on a product state and take their textbook values on a Bell
state (1, 1 and 0.5 respectively).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "schmidt_coefficients",
    "von_neumann_entropy",
    "concurrence",
    "negativity",
    "density_matrix",
]


def _normalised(psi) -> np.ndarray:
    psi = np.asarray(psi, dtype=complex).reshape(4)
    norm = np.linalg.norm(psi)
    if norm == 0:
        raise ValueError("state vector is zero")
    return psi / norm


def density_matrix(psi) -> np.ndarray:
    psi = _normalised(psi)
    return np.outer(psi, psi.conj())


def schmidt_coefficients(psi) -> np.ndarray:
    """Singular values of the 2x2 amplitude matrix, largest first."""
    psi = _normalised(psi)
    matrix = psi.reshape(2, 2)  # rows: qubit 0, columns: qubit 1
    return np.linalg.svd(matrix, compute_uv=False)


def von_neumann_entropy(psi, base: float = 2.0) -> float:
    s = schmidt_coefficients(psi)
    p = s ** 2
    p = p[p > 1e-15]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log(p) / np.log(base)))


def concurrence(psi) -> float:
    s = schmidt_coefficients(psi)
    return float(max(0.0, 2.0 * s[0] * s[1]))


def negativity(psi) -> float:
    rho = density_matrix(psi).reshape(2, 2, 2, 2)
    # rho[a, b, c, d] = <a b|rho|c d>; transpose the second qubit (b <-> d).
    rho_pt = rho.transpose(0, 3, 2, 1).reshape(4, 4)
    trace_norm = float(np.sum(np.abs(np.linalg.eigvalsh(rho_pt))))
    return max(0.0, (trace_norm - 1.0) / 2.0)
