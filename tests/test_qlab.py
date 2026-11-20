"""The fourteen checks that back the results in the project report."""

import numpy as np
import pytest

from qlab import entanglement, gates
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
from qlab.gravity import bmv_state
from qlab.register import QuantumRegister

BELL = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
PRODUCT = np.array([1, 0, 0, 0], dtype=complex)


# 1 ---------------------------------------------------------------------------
def test_bell_state_probabilities():
    counts = bell_state().sample(1000, rng=np.random.default_rng(7))
    assert set(counts) == {"00", "11"}
    assert abs(counts["00"] - counts["11"]) < 100


# 2 ---------------------------------------------------------------------------
def test_ghz_state_probabilities():
    reg = ghz_state(4)
    probs = reg.probabilities()
    assert np.isclose(probs[0b0000], 0.5)
    assert np.isclose(probs[0b1111], 0.5)
    assert np.isclose(probs.sum(), 1.0)
    assert np.count_nonzero(probs > 1e-12) == 2


# 3 ---------------------------------------------------------------------------
def test_qft_matches_discrete_fourier_transform():
    n = 3
    expected_matrix = qft_matrix(n)
    for j in range(2 ** n):
        reg = QuantumRegister(n)
        basis = np.zeros(2 ** n, dtype=complex)
        basis[j] = 1.0
        reg.set_vector(basis)
        qft(reg)
        assert np.allclose(reg.vector(), expected_matrix[:, j], atol=1e-10)


# 4 ---------------------------------------------------------------------------
def test_inverse_qft_restores_the_input():
    rng = np.random.default_rng(11)
    for n in (1, 2, 4):
        psi = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
        psi /= np.linalg.norm(psi)
        reg = QuantumRegister(n)
        reg.set_vector(psi)
        qft(reg)
        qft(reg, inverse=True)
        assert np.allclose(reg.vector(), psi, atol=1e-10)


# 5 ---------------------------------------------------------------------------
def test_phase_estimation_recovers_one_quarter():
    result = phase_estimation(0.25, m=3)
    assert result["estimate"] == pytest.approx(0.25, abs=1e-12)
    assert result["probability"] > 0.999


# 6 ---------------------------------------------------------------------------
def test_teleportation_fidelity():
    rng = np.random.default_rng(3)
    states = [np.array([1, 0], complex), np.array([0, 1], complex),
              np.array([1, 1], complex) / np.sqrt(2)]
    states += [random_single_qubit_state(rng) for _ in range(20)]
    for psi in states:
        assert teleport(psi, rng) > 0.999999


# 7 ---------------------------------------------------------------------------
def test_deutsch_jozsa_classifies_four_functions():
    n = 3
    functions = [
        (lambda x: 0, "constant"),
        (lambda x: 1, "constant"),
        (lambda x: x % 2, "balanced"),
        (lambda x: 1 if x < 2 ** (n - 1) else 0, "balanced"),
    ]
    rng = np.random.default_rng(5)
    for f, expected in functions:
        assert deutsch_jozsa(f, n, rng) == expected


# 8 ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [3, 4, 6])
def test_grover_finds_the_marked_item(n):
    best = grover_best_iteration(n)
    expected_iterations = int(round(np.pi / 4 * np.sqrt(2 ** n)))
    assert best["iterations"] == pytest.approx(expected_iterations, abs=1)
    assert best["success_probability"] > 0.90


# 9 ---------------------------------------------------------------------------
def test_bb84_without_eavesdropper_has_no_errors():
    result = run_bb84(400, eavesdropper=False, rng=np.random.default_rng(21))
    assert result["sifted_length"] > 120
    assert result["qber"] == 0.0


# 10 --------------------------------------------------------------------------
def test_bb84_with_eavesdropper_shows_a_quarter_error_rate():
    result = run_bb84(600, eavesdropper=True, rng=np.random.default_rng(22))
    assert 0.18 < result["qber"] < 0.32


# 11 --------------------------------------------------------------------------
def test_entropy_of_bell_and_product_states():
    assert entanglement.von_neumann_entropy(BELL) == pytest.approx(1.0)
    assert entanglement.von_neumann_entropy(PRODUCT) == pytest.approx(0.0)


# 12 --------------------------------------------------------------------------
def test_concurrence_of_bell_and_product_states():
    assert entanglement.concurrence(BELL) == pytest.approx(1.0)
    assert entanglement.concurrence(PRODUCT) == pytest.approx(0.0)


# 13 --------------------------------------------------------------------------
def test_negativity_of_bell_and_product_states():
    assert entanglement.negativity(BELL) == pytest.approx(0.5)
    assert entanglement.negativity(PRODUCT) == pytest.approx(0.0)


# 14 --------------------------------------------------------------------------
@pytest.mark.parametrize("phase", [0.1, 0.5, 1.0, np.pi / 2, 2.0])
def test_bmv_concurrence_is_sin_of_half_phase(phase):
    expected = abs(np.sin(phase / 2))
    assert entanglement.concurrence(bmv_state(phase)) == pytest.approx(expected, abs=1e-12)
