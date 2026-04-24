"""
qsynth.synthesizer
==================

The core synthesis engine.

Takes a :class:`PauliHamiltonian` and constructs a parameterized logical
quantum circuit that implements the Trotterized Hamiltonian evolution
operator  e^{-i H t}.

Architecture
------------
All gate types are OO representations backed by unitary matrices.
The :class:`NaiveSynthesizer` builds the QAOA-style Ansatz:

  1. Apply Hadamard layer to put all qubits in |+⟩.
  2. For each QAOA layer p:
       a. Phase-separation unitary U_C(γ):
            - For each Z term hᵢ: apply Rz(2hᵢγ) on qubit i.
            - For each ZZ term Jᵢⱼ: apply CNOT(i,j)·Rz(2Jᵢⱼγ)·CNOT(i,j).
       b. Mixing unitary U_B(β):
            - Apply Rx(2β) on every qubit.

Public surface
--------------
- :mod:`gate_defs`       – OO gate representations and unitary validation.
- :class:`LogicalCircuit`– hardware-agnostic ordered gate sequence.
- :class:`NaiveSynthesizer` – reference synthesis engine.
"""

from qsynth.synthesizer.gate_defs import (
    Gate,
    HGate,
    RxGate,
    RyGate,
    RzGate,
    CnotGate,
    CrzGate,
    IdentityGate,
    GateType,
)
from qsynth.synthesizer.logical_circuit import LogicalCircuit, GateApplication
from qsynth.synthesizer.naive_synthesizer import NaiveSynthesizer

__all__ = [
    "Gate",
    "HGate",
    "RxGate",
    "RyGate",
    "RzGate",
    "CnotGate",
    "CrzGate",
    "IdentityGate",
    "GateType",
    "LogicalCircuit",
    "GateApplication",
    "NaiveSynthesizer",
]
