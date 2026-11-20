"""Gravity-induced entanglement between two masses (Bose-Marletto-Vedral).

Model
-----
Two masses ``m`` are each held in a superposition of two positions separated
by ``s``, with the two interferometers a distance ``d`` apart.  Treating each
branch as a point mass and using the Newtonian potential, the phase picked up
by a branch pair is ``G m^2 t / (hbar * r)``.  The relative phase between the
four branches is therefore

    dphi = (G m^2 t / hbar) * (1/d + 1/(d+s) - 2/(d+s/2))

which for ``s << d`` grows like ``G m^2 t s^2 / (2 hbar d^3)`` -- quadratic in
the branch separation, not linear.  The two masses start in the product state

    (|00> + |01> + |10> + |11>) / 2

and gravity attaches phases ``0, dphi/2, dphi/2, 0`` to those four branches.
The resulting concurrence is ``|sin(dphi/2)|``, which is the BMV witness.

A single phenomenological coherence time ``T2`` multiplies the concurrence by
``exp(-t/T2)``, so waiting longer builds signal and loses coherence at the
same time.  Maximising ``exp(-t/T2) |sin(k t / 2)|`` gives
``t_opt = (2/k) arctan(k T2 / 2)``.
"""

from __future__ import annotations

import numpy as np

from . import entanglement

__all__ = [
    "G",
    "HBAR",
    "phase_difference",
    "phase_rate",
    "bmv_state",
    "concurrence",
    "optimal_interaction_time",
    "best_concurrence",
]

G = 6.67430e-11  # m^3 kg^-1 s^-2
HBAR = 1.054571817e-34  # J s


def phase_difference(mass: float, distance: float, separation: float, time: float) -> float:
    """Relative gravitational phase ``dphi`` accumulated in ``time`` seconds."""
    if mass <= 0 or distance <= 0 or separation <= 0 or time < 0:
        raise ValueError("mass, distance and separation must be positive")
    geometry = (
        1.0 / distance
        + 1.0 / (distance + separation)
        - 2.0 / (distance + separation / 2.0)
    )
    return float(G * mass ** 2 * time / HBAR * geometry)


def phase_rate(mass: float, distance: float, separation: float) -> float:
    """``dphi`` per second."""
    return phase_difference(mass, distance, separation, 1.0)


def bmv_state(phase: float) -> np.ndarray:
    """Two-mass state after the gravitational phase ``phase`` is applied."""
    phases = np.array([0.0, phase / 2.0, phase / 2.0, 0.0], dtype=complex)
    return np.exp(1j * phases) / 2.0


def concurrence(
    mass: float, distance: float, separation: float, time: float, coherence_time: float = np.inf
) -> float:
    """Dephased BMV concurrence ``exp(-t/T2) |sin(dphi/2)|``."""
    dphi = phase_difference(mass, distance, separation, time)
    decay = 0.0 if np.isinf(coherence_time) else float(np.exp(-time / coherence_time))
    return float(decay * abs(np.sin(dphi / 2.0)))


def optimal_interaction_time(
    mass: float, distance: float, separation: float, coherence_time: float
) -> float:
    """Interaction time that maximises the dephased concurrence."""
    k = phase_rate(mass, distance, separation)
    if k == 0:
        return 0.0
    return float(2.0 / k * np.arctan(k * coherence_time / 2.0))


def best_concurrence(
    mass: float, distance: float, separation: float, coherence_time: float
) -> dict:
    time = optimal_interaction_time(mass, distance, separation, coherence_time)
    value = concurrence(mass, distance, separation, time, coherence_time)
    return {
        "time": time,
        "phase": phase_difference(mass, distance, separation, time),
        "concurrence": value,
        "entanglement": entanglement.concurrence(bmv_state(phase_difference(mass, distance, separation, time))) * np.exp(-time / coherence_time),
    }
