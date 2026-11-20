"""Run the experiments from the QLab report and print their results.

Usage:  python demo.py
"""

from __future__ import annotations

import numpy as np

from qlab import entanglement, gates, gravity
from qlab.algorithms import (
    bell_state,
    deutsch_jozsa,
    ghz_state,
    grover_best_iteration,
    phase_estimation,
    qft,
    qft_matrix,
    random_single_qubit_state,
    teleport,
)
from qlab.bb84 import run_bb84
from qlab.register import QuantumRegister

rng = np.random.default_rng(2026)
line = "-" * 62


def section(title: str) -> None:
    print(f"\n{title}\n{line}")


def main() -> None:
    section("Bell state, 1000 shots")
    print(bell_state().sample(1000, rng))

    section("GHZ state on four qubits, 1000 shots")
    print(ghz_state(4).sample(1000, rng))

    section("QFT vs the discrete Fourier transform (3 qubits)")
    n = 3
    for j in (0, 1, 5):
        reg = QuantumRegister(n)
        basis = np.zeros(2 ** n, dtype=complex)
        basis[j] = 1.0
        reg.set_vector(basis)
        qft(reg)
        error = np.max(np.abs(reg.vector() - qft_matrix(n)[:, j]))
        print(f"  |{j:03b}> -> max |QFT - DFT| = {error:.2e}")

    section("Phase estimation, theta = 0.25")
    print(f"  estimate = {phase_estimation(0.25, m=3)['estimate']:.4f}")

    section("Teleportation over 200 random states")
    fidelities = [teleport(random_single_qubit_state(rng), rng) for _ in range(200)]
    print(f"  minimum fidelity = {min(fidelities):.9f}")

    section("Deutsch-Jozsa")
    for f, name in (
        (lambda x: 0, "constant 0"),
        (lambda x: 1, "constant 1"),
        (lambda x: x % 2, "parity"),
        (lambda x: 1 if x < 8 else 0, "half and half"),
    ):
        print(f"  {name:<14} -> {deutsch_jozsa(f, 4, rng)}")

    section("Grover search")
    print(f"  {'qubits':>6} {'items':>6} {'iterations':>11} {'success':>8} {'random':>7}")
    for n in (3, 4, 6):
        best = grover_best_iteration(n)
        print(
            f"  {n:>6} {2 ** n:>6} {best['iterations']:>11} "
            f"{best['success_probability']:>7.1%} {1 / 2 ** n:>7.1%}"
        )

    section("BB84")
    for eavesdropper, label in ((False, "no eavesdropper"), (True, "intercept-resend")):
        result = run_bb84(2000, eavesdropper=eavesdropper, rng=rng)
        print(
            f"  {label:<18} sifted {result['sifted_length']:>4} bits, "
            f"QBER {result['qber']:>6.1%}"
        )

    section("Bose-Marletto-Vedral phase vs branch separation")
    mass, distance = 1e-13, 1e-3
    previous = None
    for separation in (50e-6, 100e-6, 200e-6, 400e-6):
        phi = gravity.phase_difference(mass, distance, separation, 1.0)
        ratio = "" if previous is None else f"  (x{phi / previous:.2f})"
        print(f"  s = {separation * 1e6:>5.0f} um  ->  dphi after 1 s = {phi:8.4f} rad{ratio}")
        previous = phi

    section("Optimal interaction time and BMV concurrence")
    separation = 250e-6
    heavy = 1e-12
    print(f"  m = 1e-12 kg, d = 1 mm, s = 250 um")
    print(f"  {'T2 (s)':>8} {'t_opt (s)':>10} {'dphi':>7} {'concurrence':>12}")
    for t2 in (0.05, 0.2, 1.0):
        best = gravity.best_concurrence(heavy, distance, separation, t2)
        print(
            f"  {t2:>8.1f} {best['time']:>10.4f} {best['phase']:>7.4f} "
            f"{best['concurrence']:>12.4f}"
        )

    section("Entanglement measures on a Bell state")
    bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    print(f"  entropy     = {entanglement.von_neumann_entropy(bell):.6f}")
    print(f"  concurrence = {entanglement.concurrence(bell):.6f}")
    print(f"  negativity  = {entanglement.negativity(bell):.6f}")


if __name__ == "__main__":
    main()
