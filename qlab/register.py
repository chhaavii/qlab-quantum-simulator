"""A dense state-vector quantum register.

The state of ``n`` qubits is stored as a NumPy array of shape ``(2, 2, ..., 2)``
with one axis per qubit.  This keeps single-qubit gates cheap: a gate is a
contraction along one axis, so nothing has to be rebuilt.

Convention: **qubit 0 is the most significant bit**.  The flat index of a basis
state ``|b0 b1 ... b(n-1)>`` is ``sum_i b_i * 2**(n-1-i)``, which is exactly
what ``state.reshape(-1)`` produces.
"""

from __future__ import annotations

import numpy as np

from . import gates

__all__ = ["QuantumRegister"]


def _as_gate(matrix) -> np.ndarray:
    U = np.asarray(matrix, dtype=complex)
    if U.shape != (2, 2):
        raise ValueError(f"a single-qubit gate must be 2x2, got {U.shape}")
    return U


def _as_state(n: int, state) -> np.ndarray:
    psi = np.asarray(state, dtype=complex)
    if psi.shape != (2,) * n:
        try:
            psi = psi.reshape((2,) * n)
        except ValueError:
            raise ValueError(
                f"state has {psi.size} amplitudes, expected {2 ** n} for {n} qubits"
            ) from None
    norm = np.linalg.norm(psi.reshape(-1))
    if not np.isclose(norm, 1.0, atol=1e-9):
        raise ValueError(f"state is not normalised (norm = {norm:.6g})")
    return psi.copy()


class QuantumRegister:
    """An ``n``-qubit register with gate application and projective measurement."""

    def __init__(self, n: int, state=None):
        n = int(n)
        if n < 1:
            raise ValueError("a register needs at least one qubit")
        self.n = n
        if state is None:
            psi = np.zeros((2,) * n, dtype=complex)
            psi[(0,) * n] = 1.0
            self.state = psi
        else:
            self.state = _as_state(n, state)

    # ------------------------------------------------------------------ state
    @property
    def dim(self) -> int:
        return 2 ** self.n

    def vector(self) -> np.ndarray:
        """Flat state vector, qubit 0 first (most significant bit)."""
        return self.state.reshape(-1)

    def set_vector(self, vector) -> None:
        self.state = _as_state(self.n, np.asarray(vector, dtype=complex).reshape(-1))

    def copy(self) -> "QuantumRegister":
        return QuantumRegister(self.n, self.state.copy())

    def probabilities(self) -> np.ndarray:
        return np.abs(self.vector()) ** 2

    # ------------------------------------------------------------------ gates
    def _check(self, qubit) -> int:
        q = int(qubit)
        if not 0 <= q < self.n:
            raise IndexError(f"qubit {q} is outside a {self.n}-qubit register")
        return q

    def apply(self, matrix, target) -> None:
        """Apply a single-qubit gate to ``target``."""
        target = self._check(target)
        U = _as_gate(matrix)
        # Contract the gate's column index with the target axis.
        self.state = np.moveaxis(
            np.tensordot(U, self.state, axes=([1], [target])), 0, target
        )

    def controlled(self, matrix, controls, target) -> None:
        """Apply ``matrix`` to ``target`` when every control qubit is 1.

        Implemented as a slice-and-write: move the target axis first, take the
        block where all controls equal 1, contract the gate inside that block
        and write it back.  Any number of controls is supported, which is what
        Grover's multi-controlled Z needs.
        """
        target = self._check(target)
        if isinstance(controls, (int, np.integer)):
            controls = [int(controls)]
        else:
            controls = [int(c) for c in controls]
        controls = [self._check(c) for c in controls]
        if target in controls:
            raise ValueError("a qubit cannot control itself")
        if not controls:
            self.apply(matrix, target)
            return

        U = _as_gate(matrix)
        rest = [q for q in range(self.n) if q != target and q not in controls]
        perm = [target] + controls + rest
        st = np.transpose(self.state, perm)
        block_index = (slice(None),) + (1,) * len(controls)
        block = st[block_index]
        st[block_index] = np.tensordot(U, block, axes=([1], [0]))
        self.state = np.transpose(st, np.argsort(perm))

    # named gates
    def x(self, q):
        self.apply(gates.X, q)

    def y(self, q):
        self.apply(gates.Y, q)

    def z(self, q):
        self.apply(gates.Z, q)

    def h(self, q):
        self.apply(gates.H, q)

    def s(self, q):
        self.apply(gates.S, q)

    def t(self, q):
        self.apply(gates.T, q)

    def phase(self, theta, q):
        self.apply(gates.phase(theta), q)

    def cnot(self, control, target):
        self.controlled(gates.X, control, target)

    def cz(self, control, target):
        self.controlled(gates.Z, control, target)

    def swap(self, a, b):
        a, b = self._check(a), self._check(b)
        if a == b:
            return
        # SWAP = CNOT(a,b) CNOT(b,a) CNOT(a,b)
        self.cnot(a, b)
        self.cnot(b, a)
        self.cnot(a, b)

    # ------------------------------------------------------------ measurement
    def probability_one(self, qubit) -> float:
        qubit = self._check(qubit)
        index = [slice(None)] * self.n
        index[qubit] = 1
        return float(np.sum(np.abs(self.state[tuple(index)]) ** 2))

    def measure(self, qubit, rng=None) -> int:
        """Projective measurement of one qubit (collapse + renormalise)."""
        rng = np.random.default_rng() if rng is None else rng
        qubit = self._check(qubit)
        outcome = 1 if rng.random() < self.probability_one(qubit) else 0
        index = [slice(None)] * self.n
        index[qubit] = 1 - outcome
        self.state[tuple(index)] = 0.0
        norm = float(np.linalg.norm(self.vector()))
        if norm < 1e-12:
            raise RuntimeError("measurement left no probability mass")
        self.state /= norm
        return outcome

    def measure_all(self, rng=None) -> str:
        """Measure every qubit and return the bit string (qubit 0 first)."""
        rng = np.random.default_rng() if rng is None else rng
        bits = [str(self.measure(q, rng)) for q in range(self.n)]
        return "".join(bits)

    def sample(self, shots: int = 1024, rng=None) -> dict:
        """Sample ``shots`` bit strings from the full probability distribution."""
        rng = np.random.default_rng() if rng is None else rng
        probs = self.probabilities()
        probs = probs / probs.sum()
        draws = rng.choice(self.dim, size=int(shots), p=probs)
        counts: dict[str, int] = {}
        for index in draws:
            key = format(int(index), f"0{self.n}b")
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"QuantumRegister({self.n} qubits, dim={self.dim})"
