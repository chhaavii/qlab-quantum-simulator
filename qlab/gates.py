"""Single-qubit gate matrices.

Every gate is a 2x2 complex unitary.  The simulator only needs the matrices;
multi-qubit gates are built by the register (controlled gates, SWAP).
"""

from __future__ import annotations

import numpy as np

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2.0)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4.0)]], dtype=complex)


def phase(theta: float) -> np.ndarray:
    """diag(1, exp(i*theta)) -- the phase gate used by QFT and phase estimation."""
    return np.array([[1.0, 0.0], [0.0, np.exp(1j * theta)]], dtype=complex)


def rx(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(theta: float) -> np.ndarray:
    return np.array(
        [[np.exp(-0.5j * theta), 0.0], [0.0, np.exp(0.5j * theta)]], dtype=complex
    )
