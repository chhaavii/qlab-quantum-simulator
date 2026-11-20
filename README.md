# QLab -- a quantum computer simulator built from scratch

A state-vector quantum computer simulator written in Python with **only NumPy**
(no Qiskit, Cirq or other SDK), plus the algorithms and experiments from the
project report: Deutsch-Jozsa, Grover, the quantum Fourier transform, phase
estimation, teleportation, three entanglement measures, BB84 with an
eavesdropper, and the Bose-Marletto-Vedral (BMV) gravity-induced entanglement
model.

## Install and run

```bash
python3 -m pip install -r requirements.txt
python3 -m pytest -q      # the fourteen automated tests
python3 demo.py           # reproduce the numbers in the report
```

## Layout

```
qlab/
  register.py       state tensor, gates, controlled gates, measurement
  gates.py          2x2 gate matrices
  algorithms.py     Bell/GHZ, teleportation, Deutsch-Jozsa, Grover, QFT,
                    phase estimation
  entanglement.py   Schmidt coefficients, entropy, concurrence, negativity
  bb84.py           BB84 key exchange with optional intercept-resend Eve
  gravity.py        BMV phase, dephasing, optimal interaction time
tests/test_qlab.py  the fourteen checks from the report
demo.py             runs every experiment and prints the results
```

## Conventions

* The state of `n` qubits is an array of shape `(2, ..., 2)`, one axis per
  qubit, so a gate is a contraction along a single axis.
* **Qubit 0 is the most significant bit.** The flat index of `|b0...bn-1>` is
  `sum b_i 2^(n-1-i)`.
* A controlled gate takes the block of the state where every control is 1,
  applies the gate to the target inside that block, and writes it back. This
  scales with the block size instead of building a full `2^n x 2^n` matrix and
  supports any number of controls (Grover's multi-controlled Z).
* Measurement computes the probability of 1, draws an outcome, zeroes the
  amplitudes that disagree and renormalises.

## What the tests check

The fourteen tests in `tests/test_qlab.py` cover:

1. Bell-state sampling only produces `00` and `11`, in equal numbers.
2. The four-qubit GHZ state has exactly two nonzero amplitudes.
3. The QFT matches the discrete Fourier transform on every 3-qubit basis state.
4. The inverse QFT restores a random state.
5. Phase estimation returns exactly 0.25 for `theta = 0.25`.
6. Teleportation reaches fidelity 1 for 23 test states.
7. Deutsch-Jozsa classifies two constant and two balanced functions.
8. Grover succeeds above 90% for 3, 4 and 6 qubits at the predicted iteration.
9. BB84 without an eavesdropper has a 0% error rate.
10. BB84 with a full intercept-resend eavesdropper has about a 25% error rate.
11. Entropy is 1 on a Bell state and 0 on a product state.
12. Concurrence is 1 on a Bell state and 0 on a product state.
13. Negativity is 0.5 on a Bell state and 0 on a product state.
14. The BMV concurrence equals `|sin(dphi/2)|`.

## The gravity model

Two masses `m`, each in a superposition of positions separated by `s`, with the
interferometers a distance `d` apart. The relative phase is

```
dphi = (G m^2 t / hbar) * (1/d + 1/(d+s) - 2/(d+s/2))
```

which for `s << d` is `G m^2 t s^2 / (2 hbar d^3)` -- quadratic in the branch
separation. The two masses start in `(|00>+|01>+|10>+|11>)/2`; gravity attaches
phases `0, dphi/2, dphi/2, 0`, giving concurrence `|sin(dphi/2)|`.

A single coherence time `T2` multiplies the concurrence by `exp(-t/T2)`, so
signal and decoherence compete. The optimal interaction time is

```
t_opt = (2/k) arctan(k T2 / 2),   k = dphi per second
```

## Limitations

* Memory doubles with every qubit: 25 qubits need about 0.5 GB, 30 about 16 GB.
* Gates are ideal; there are no gate, readout or connectivity errors.
* The only noise is the phenomenological dephasing factor in the gravity model.
* BB84 is simulated at the bit level and ignores photon loss, detector noise
  and finite-key statistics, so the 11% security threshold is only a guide.
* The gravity study is a two-qubit idealisation with point-like branches.
* Results have not been cross-checked against Qiskit or Cirq.
