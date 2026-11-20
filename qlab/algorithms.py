"""Quantum algorithms built on :class:`qlab.register.QuantumRegister`.

Everything here uses gates only -- no algorithm is written as a special-case
matrix, so the circuits are the ones you would draw on paper.
"""

from __future__ import annotations

import numpy as np

from . import gates
from .register import QuantumRegister

__all__ = [
    "bell_state",
    "ghz_state",
    "random_single_qubit_state",
    "teleport",
    "deutsch_jozsa",
    "grover",
    "grover_best_iteration",
    "qft",
    "phase_estimation",
]


# --------------------------------------------------------------- Bell / GHZ
def bell_state() -> QuantumRegister:
    reg = QuantumRegister(2)
    reg.h(0)
    reg.cnot(0, 1)
    return reg


def ghz_state(n: int) -> QuantumRegister:
    if n < 2:
        raise ValueError("GHZ needs at least two qubits")
    reg = QuantumRegister(n)
    reg.h(0)
    for q in range(1, n):
        reg.cnot(0, q)
    return reg


# --------------------------------------------------------------- teleportation
def random_single_qubit_state(rng=None) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    psi = rng.normal(size=2) + 1j * rng.normal(size=2)
    return psi / np.linalg.norm(psi)


def teleport(psi, rng=None) -> float:
    """Teleport ``psi`` and return the fidelity of the received qubit.

    Qubit 0 holds the state, qubits 1 and 2 share a Bell pair.  After the
    Bell-basis measurement on qubits 0 and 1 the corrections X^m1 Z^m0 are
    applied to qubit 2.
    """
    rng = np.random.default_rng() if rng is None else rng
    psi = np.asarray(psi, dtype=complex).reshape(2)
    psi = psi / np.linalg.norm(psi)

    reg = QuantumRegister(3)
    reg.set_vector(np.kron(psi, np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)))
    reg.h(1)
    reg.cnot(1, 2)

    reg.cnot(0, 1)
    reg.h(0)
    m0 = reg.measure(0, rng)
    m1 = reg.measure(1, rng)

    if m1:
        reg.x(2)
    if m0:
        reg.z(2)

    received = reg.state.reshape(2, 2, 2)[m0, m1, :]
    received = received / np.linalg.norm(received)
    return float(abs(np.vdot(psi, received)) ** 2)


# ------------------------------------------------------------- Deutsch-Jozsa
def _apply_function_oracle(reg: QuantumRegister, f, n: int) -> None:
    """Permutation oracle |x>|y> -> |x>|y xor f(x)> on an (n+1)-qubit register."""
    dim = 2 ** (n + 1)
    perm = np.arange(dim)
    for x in range(2 ** n):
        fx = int(f(x)) & 1
        perm[(x << 1) | 0] = (x << 1) | fx
        perm[(x << 1) | 1] = (x << 1) | (1 ^ fx)
    old = reg.vector()
    new = np.empty_like(old)
    new[perm] = old
    reg.set_vector(new)


def deutsch_jozsa(f, n: int, rng=None) -> str:
    """Decide whether ``f`` is constant or balanced with one oracle call."""
    rng = np.random.default_rng() if rng is None else rng
    reg = QuantumRegister(n + 1)
    reg.x(n)  # ancilla in |1>
    for q in range(n + 1):
        reg.h(q)
    _apply_function_oracle(reg, f, n)
    for q in range(n):
        reg.h(q)
    bits = [reg.measure(q, rng) for q in range(n)]
    return "constant" if not any(bits) else "balanced"


# --------------------------------------------------------------------- Grover
def _multi_controlled_z(reg: QuantumRegister) -> None:
    n = reg.n
    reg.controlled(gates.Z, list(range(n - 1)), n - 1)


def _mark(reg: QuantumRegister, marked: int) -> None:
    n = reg.n
    # X gates turn the marked bit string into the all-ones case.
    zeros = [q for q in range(n) if not (marked >> (n - 1 - q)) & 1]
    for q in zeros:
        reg.x(q)
    _multi_controlled_z(reg)
    for q in zeros:
        reg.x(q)


def _diffuse(reg: QuantumRegister) -> None:
    n = reg.n
    for q in range(n):
        reg.h(q)
    for q in range(n):
        reg.x(q)
    _multi_controlled_z(reg)
    for q in range(n):
        reg.x(q)
    for q in range(n):
        reg.h(q)


def grover(n: int, marked: int, iterations=None, rng=None) -> dict:
    """Run Grover search and return the success probability."""
    rng = np.random.default_rng() if rng is None else rng
    marked = int(marked)
    if not 0 <= marked < 2 ** n:
        raise ValueError("marked item is outside the search space")
    if iterations is None:
        iterations = max(1, int(round(np.pi / 4.0 * np.sqrt(2 ** n))))

    reg = QuantumRegister(n)
    for q in range(n):
        reg.h(q)
    for _ in range(int(iterations)):
        _mark(reg, marked)
        _diffuse(reg)

    probs = reg.probabilities()
    return {
        "success_probability": float(probs[marked]),
        "iterations": int(iterations),
        "sampled": reg.sample(1000, rng),
        "probabilities": probs,
    }


def grover_best_iteration(n: int, marked: int = None) -> dict:
    """Scan the number of iterations and report the best one."""
    marked = 0 if marked is None else int(marked)
    # Search only the useful window; much later iterations revive the
    # probability but are not what "the" Grover iteration count means.
    limit = max(1, int(np.ceil(np.pi / 4.0 * np.sqrt(2 ** n))) + 1)
    best = {"iterations": 1, "success_probability": 0.0}
    for steps in range(1, limit + 1):
        prob = grover(n, marked, iterations=steps)["success_probability"]
        if prob > best["success_probability"]:
            best = {"iterations": steps, "success_probability": prob}
    return best


# ------------------------------------------------------------------------ QFT
def qft(reg: QuantumRegister, qubits=None, inverse: bool = False) -> None:
    """Quantum Fourier transform on ``qubits`` (default: the whole register).

    ``F[j, k] = exp(2*pi*i*j*k / N) / sqrt(N)``.  With ``inverse=True`` the
    adjoint circuit is applied, so ``qft(reg, inverse=True)`` undoes ``qft(reg)``.
    """
    qubits = list(range(reg.n)) if qubits is None else [int(q) for q in qubits]
    k = len(qubits)

    if inverse:
        for i in range(k // 2):
            reg.swap(qubits[i], qubits[k - 1 - i])
        for j in reversed(range(k)):
            for m in reversed(range(j + 1, k)):
                reg.controlled(gates.phase(-np.pi / 2.0 ** (m - j)), qubits[m], qubits[j])
            reg.h(qubits[j])
    else:
        for j in range(k):
            reg.h(qubits[j])
            for m in range(j + 1, k):
                reg.controlled(gates.phase(np.pi / 2.0 ** (m - j)), qubits[m], qubits[j])
        for i in range(k // 2):
            reg.swap(qubits[i], qubits[k - 1 - i])


def qft_matrix(n: int) -> np.ndarray:
    """The unitary that :func:`qft` implements, for reference and testing."""
    N = 2 ** n
    j = np.arange(N)[:, None]
    k = np.arange(N)[None, :]
    return np.exp(2j * np.pi * j * k / N) / np.sqrt(N)


# -------------------------------------------------------------- phase estimation
def phase_estimation(theta: float, m: int = 3) -> dict:
    """Estimate ``theta`` in ``U|1> = exp(2*pi*i*theta)|1>`` with ``m`` bits."""
    U = np.diag([1.0, np.exp(2j * np.pi * theta)])
    reg = QuantumRegister(m + 1)
    reg.x(m)  # eigenstate |1> of U
    for q in range(m):
        reg.h(q)
    for j in range(m):
        power = 2 ** (m - 1 - j)
        reg.controlled(np.linalg.matrix_power(U, power), j, m)
    qft(reg, qubits=list(range(m)), inverse=True)

    # Marginal over the counting register: the target qubit is the last bit.
    probs = reg.probabilities().reshape(2 ** m, 2).sum(axis=1)
    index = int(np.argmax(probs))
    return {
        "estimate": index / 2 ** m,
        "index": index,
        "probability": float(probs[index]),
        "register": reg,
    }
