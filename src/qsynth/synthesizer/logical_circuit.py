"""
qsynth.synthesizer.logical_circuit
====================================

Hardware-agnostic logical quantum circuit representation.

A :class:`LogicalCircuit` is an ordered sequence of :class:`GateApplication`
objects.  Each application associates a :class:`Gate` with an ordered tuple
of *logical* qubit indices.

The circuit is hardware-agnostic: qubit indices are abstract integers.
The Compiler layer is responsible for mapping them to physical qubits.

Key operations
--------------
- :meth:`append`    – add a gate application.
- :meth:`depth`     – critical-path depth (counting all gates in parallel).
- :meth:`to_qiskit` – convert to a :class:`qiskit.circuit.QuantumCircuit`.
- :meth:`unitary`   – compute the full unitary of the circuit via matrix product.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister, Parameter
from qiskit.circuit.library import (
    HGate as QiskitH,
    RXGate as QiskitRX,
    RYGate as QiskitRY,
    RZGate as QiskitRZ,
    CXGate as QiskitCX,
    CRZGate as QiskitCRZ,
)

from qsynth.exceptions import DimensionMismatchError
from qsynth.synthesizer.gate_defs import (
    Gate,
    GateType,
    HGate,
    RxGate,
    RyGate,
    RzGate,
    CnotGate,
    CrzGate,
    IdentityGate,
)


@dataclass
class GateApplication:
    """
    Associates a :class:`Gate` with a tuple of target qubit indices.

    Attributes
    ----------
    gate    : Gate        – the gate to apply.
    qubits  : tuple[int, ...]  – logical qubit indices (ordered by gate convention).
    label   : str         – optional annotation.
    """

    gate: Gate
    qubits: tuple[int, ...]
    label: str = ""

    def __post_init__(self) -> None:
        if len(self.qubits) != self.gate.n_qubits:
            raise DimensionMismatchError(
                expected_dim=self.gate.n_qubits,
                actual_dim=len(self.qubits),
                context=f"GateApplication({self.gate.label()})",
            )
        if len(set(self.qubits)) != len(self.qubits):
            raise ValueError(
                f"Duplicate qubit indices in GateApplication: {self.qubits}"
            )

    def __repr__(self) -> str:
        qstr = ",".join(map(str, self.qubits))
        tag = f" [{self.label}]" if self.label else ""
        return f"{self.gate.label()} q[{qstr}]{tag}"


class LogicalCircuit:
    """
    Hardware-agnostic ordered sequence of gate applications.

    Attributes
    ----------
    n_qubits   : int                   – total number of logical qubits.
    operations : list[GateApplication] – ordered gate sequence.
    name       : str                   – optional circuit name.
    """

    def __init__(self, n_qubits: int, name: str = "circuit") -> None:
        if n_qubits <= 0:
            raise ValueError(f"n_qubits must be > 0, got {n_qubits}.")
        self.n_qubits = n_qubits
        self.name = name
        self.operations: list[GateApplication] = []

    # ── mutation ───────────────────────────────────────────────────────────

    def append(
        self,
        gate: Gate,
        qubits: tuple[int, ...] | list[int],
        label: str = "",
    ) -> None:
        """Append a gate application, validating qubit indices."""
        qubits_t = tuple(qubits)
        for q in qubits_t:
            if not (0 <= q < self.n_qubits):
                raise ValueError(
                    f"Qubit index {q} out of range [0, {self.n_qubits})."
                )
        self.operations.append(GateApplication(gate=gate, qubits=qubits_t, label=label))

    def h(self, qubit: int) -> None:
        self.append(HGate(), (qubit,))

    def rx(self, theta: float, qubit: int) -> None:
        self.append(RxGate(theta), (qubit,))

    def ry(self, theta: float, qubit: int) -> None:
        self.append(RyGate(theta), (qubit,))

    def rz(self, theta: float, qubit: int) -> None:
        self.append(RzGate(theta), (qubit,))

    def cnot(self, control: int, target: int) -> None:
        self.append(CnotGate(), (control, target))

    def crz(self, theta: float, control: int, target: int) -> None:
        self.append(CrzGate(theta), (control, target))

    # ── analysis ───────────────────────────────────────────────────────────

    def gate_count(self) -> int:
        """Total number of gate applications."""
        return len(self.operations)

    def two_qubit_gate_count(self) -> int:
        """Count of two-qubit gates (CNOT, CRZ, etc.)."""
        return sum(1 for op in self.operations if op.gate.n_qubits == 2)

    def depth(self) -> int:
        """
        Compute the circuit depth as the critical-path length.

        Uses a greedy layer-assignment algorithm:
        Each gate is placed in the earliest layer where none of its
        target qubits are occupied by a later-placed gate.
        """
        qubit_layer: list[int] = [0] * self.n_qubits
        for op in self.operations:
            earliest = max(qubit_layer[q] for q in op.qubits)
            for q in op.qubits:
                qubit_layer[q] = earliest + 1
        return max(qubit_layer) if qubit_layer else 0

    def unitary(self) -> np.ndarray:
        """
        Compute the full 2ⁿ × 2ⁿ unitary of the circuit.

        Uses Qiskit's :class:`~qiskit.quantum_info.Operator` (C++ backed)
        for fast and correct unitary computation, including proper qubit
        ordering conventions.

        Warning: exponential memory — only use for n ≤ 12.
        """
        from qiskit.quantum_info import Operator
        qc = self.to_qiskit()
        op = Operator(qc)
        return op.data

    def _embed_gate(self, op: GateApplication) -> np.ndarray:
        """(Legacy) Embed a gate into the full n-qubit Hilbert space."""
        if op.gate.n_qubits == 1:
            return self._kron_single(op.gate.matrix(), op.qubits[0])
        elif op.gate.n_qubits == 2:
            return self._embed_two_qubit(op.gate.matrix(), op.qubits[0], op.qubits[1])
        else:
            raise NotImplementedError(
                f"Embedding for {op.gate.n_qubits}-qubit gates not implemented."
            )

    def _kron_single(self, gate_matrix: np.ndarray, target: int) -> np.ndarray:
        """
        Embed a 2×2 single-qubit gate on *target* into the full space.

        Tensor product (Qiskit convention: qubit 0 = least significant bit):
          I_{n-1} ⊗ ... ⊗ G_{target} ⊗ ... ⊗ I_0
        """
        n = self.n_qubits
        # Build kron from qubit n-1 down to 0; G goes at position `target`.
        ops_list = [
            gate_matrix if q == target else np.eye(2, dtype=complex)
            for q in range(n - 1, -1, -1)
        ]
        result = ops_list[0]
        for m in ops_list[1:]:
            result = np.kron(result, m)
        return result

    def _embed_two_qubit(
        self,
        gate_matrix: np.ndarray,
        control: int,
        target: int,
    ) -> np.ndarray:
        """
        Embed a 4×4 two-qubit gate (control, target) into the full n-qubit space.

        Uses fully vectorized numpy operations (no Python loops):
        1. Extract control/target bit values for all 2ⁿ basis states in parallel.
        2. Apply the gate matrix column by column via fancy indexing.
        """
        n = self.n_qubits
        dim = 2 ** n
        states = np.arange(dim, dtype=np.int64)

        # Extract control and target bit values (vectorized).
        c_vals = (states >> control) & 1   # shape (dim,)
        t_vals = (states >> target) & 1    # shape (dim,)
        local_cols = (c_vals * 2 + t_vals).astype(np.int64)  # 0..3

        U_full = np.zeros((dim, dim), dtype=complex)

        for local_row in range(4):
            new_c = (local_row >> 1) & 1
            new_t = local_row & 1
            amps = gate_matrix[local_row, local_cols]   # shape (dim,)
            # Build row states: clear control/target bits, set new values.
            row_states = (
                (states & ~(1 << control) & ~(1 << target))
                | (new_c << control)
                | (new_t << target)
            )
            # Accumulate: U_full[row_states[s], s] += amps[s]
            np.add.at(U_full, (row_states, states), amps)

        return U_full

    # ── Qiskit conversion ──────────────────────────────────────────────────

    def to_qiskit(self) -> QuantumCircuit:
        """
        Convert to a :class:`qiskit.circuit.QuantumCircuit`.

        Gate parameter values are embedded as concrete floats (not symbolic
        Parameters) so the circuit can be immediately simulated.
        """
        qr = QuantumRegister(self.n_qubits, "q")
        qc = QuantumCircuit(qr, name=self.name)
        for op in self.operations:
            g = op.gate
            qs = [qr[q] for q in op.qubits]
            match g.gate_type:
                case GateType.H:
                    qc.h(qs[0])
                case GateType.RX:
                    qc.rx(g.params()[0], qs[0])
                case GateType.RY:
                    qc.ry(g.params()[0], qs[0])
                case GateType.RZ:
                    qc.rz(g.params()[0], qs[0])
                case GateType.CNOT:
                    qc.cx(qs[0], qs[1])
                case GateType.CRZ:
                    qc.crz(g.params()[0], qs[0], qs[1])
                case GateType.IDENTITY:
                    qc.id(qs[0])
                case _:
                    qc.unitary(g.matrix(), qs, label=g.label())
        return qc

    # ── display ────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"LogicalCircuit(n_qubits={self.n_qubits}, "
            f"gates={self.gate_count()}, depth={self.depth()})"
        )

    def describe(self, max_ops: int = 40) -> str:
        lines = [
            f"LogicalCircuit '{self.name}'",
            f"  Qubits   : {self.n_qubits}",
            f"  Gates    : {self.gate_count()}",
            f"  2Q gates : {self.two_qubit_gate_count()}",
            f"  Depth    : {self.depth()}",
            "  Operations:",
        ]
        for i, op in enumerate(self.operations[:max_ops]):
            lines.append(f"    [{i:3d}] {op}")
        if len(self.operations) > max_ops:
            lines.append(f"    ... ({len(self.operations) - max_ops} more)")
        return "\n".join(lines)
