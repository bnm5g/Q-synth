"""
qsynth.ir
=========

Intermediate Representation (IR) layer.

Converts a :class:`QUBOProblem` into a Pauli Hamiltonian expressed as a
weighted sum of Pauli tensor products (Ising model):

    H = Σᵢ hᵢ Zᵢ  +  Σᵢ<ⱼ Jᵢⱼ ZᵢZⱼ  +  const·I

This is the standard QUBO → Ising mapping via the substitution:
    xᵢ = (1 − σᵢᶻ) / 2

The resulting Pauli Hamiltonian is the direct input to the Synthesizer.

Public surface
--------------
- :class:`PauliTerm`        – a single weighted Pauli string.
- :class:`PauliHamiltonian` – collection of :class:`PauliTerm` objects.
- :func:`qubo_to_ising`     – performs the QUBO → Ising mapping.
- :func:`build_hamiltonian` – wraps qubo_to_ising into a PauliHamiltonian.
"""

from qsynth.ir.pauli_hamiltonian import PauliHamiltonian, PauliTerm
from qsynth.ir.ising_mapper import qubo_to_ising, build_hamiltonian

__all__ = [
    "PauliTerm",
    "PauliHamiltonian",
    "qubo_to_ising",
    "build_hamiltonian",
]
