"""BB84 quantum key distribution, simulated at the qubit level.

Basis 0 is the computational (Z) basis, basis 1 the Hadamard (X) basis.
Without an eavesdropper the sifted keys agree exactly.  An intercept-resend
eavesdropper who measures every qubit in a random basis introduces a 25%
error rate, which is the signature BB84 looks for.
"""

from __future__ import annotations

import numpy as np

from .register import QuantumRegister

__all__ = ["run_bb84"]


def _prepare(bit: int, basis: int) -> QuantumRegister:
    reg = QuantumRegister(1)
    if bit:
        reg.x(0)
    if basis:
        reg.h(0)
    return reg


def _measure(reg: QuantumRegister, basis: int, rng) -> int:
    if basis:
        reg.h(0)
    return reg.measure(0, rng)


def run_bb84(n_bits: int, eavesdropper: bool = False, rng=None) -> dict:
    """Run BB84 and report the sifted key and the quantum bit error rate."""
    rng = np.random.default_rng() if rng is None else rng
    n_bits = int(n_bits)

    alice_bits = rng.integers(0, 2, n_bits)
    alice_bases = rng.integers(0, 2, n_bits)
    bob_bases = rng.integers(0, 2, n_bits)
    bob_bits = np.zeros(n_bits, dtype=int)

    for i in range(n_bits):
        reg = _prepare(int(alice_bits[i]), int(alice_bases[i]))
        if eavesdropper:
            eve_basis = int(rng.integers(0, 2))
            eve_bit = _measure(reg, eve_basis, rng)
            reg = _prepare(eve_bit, eve_basis)
        bob_bits[i] = _measure(reg, int(bob_bases[i]), rng)

    keep = alice_bases == bob_bases
    alice_key = alice_bits[keep]
    bob_key = bob_bits[keep]
    errors = int(np.sum(alice_key != bob_key))
    sifted = int(keep.sum())
    return {
        "alice_key": alice_key,
        "bob_key": bob_key,
        "sifted_length": sifted,
        "errors": errors,
        "qber": (errors / sifted) if sifted else 0.0,
    }
