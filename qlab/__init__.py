"""QLab -- a quantum computer simulator written with NumPy only.

Typical use::

    from qlab import QuantumRegister
    from qlab.algorithms import bell_state

    reg = bell_state()
    print(reg.sample(1000))       # {'00': ~500, '11': ~500}
"""

from .register import QuantumRegister
from . import algorithms, bb84, entanglement, gates, gravity

__all__ = ["QuantumRegister", "algorithms", "bb84", "entanglement", "gates", "gravity"]
__version__ = "1.0.0"
