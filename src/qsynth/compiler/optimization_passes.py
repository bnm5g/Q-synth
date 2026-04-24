"""
qsynth.compiler.optimization_passes
=====================================

Algebraic optimization passes for depth reduction.

Each pass implements the :class:`OptimizationPass` Strategy interface and
is composed via a :class:`PassManager`.

Passes implemented
------------------

1. **DeadCodeEliminationPass**
   Removes gates whose matrix is (approximately) the identity.

2. **CancellationPass**
   Detects adjacent gate pairs (G, G†) on the *same* qubits.
   Uses matrix multiplication:  G · G† ≈ I  → cancel both.
   
   Examples:
   - CNOT · CNOT = I
   - H · H = I
   - Rz(θ) · Rz(−θ) = I

3. **MergeRotationsPass**
   Merges consecutive same-axis rotation gates on the same qubit:
   - Rz(a) · Rz(b)  →  Rz(a+b)
   - Rx(a) · Rx(b)  →  Rx(a+b)
   - Ry(a) · Ry(b)  →  Ry(a+b)
   Drops merged gates when |a+b| < tol.

4. **CommutativityPass**
   Performs one pass of adjacent-gate swapping.  Two adjacent gates G₁ G₂
   acting on *disjoint* qubits trivially commute and can be reordered
   without any matrix check.  For overlapping qubits, checks G₁G₂ = G₂G₁
   via matrix product.  Swapping re-orders the gate list to expose more
   cancellation opportunities.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

from qsynth.exceptions import OptimizationError
from qsynth.synthesizer.gate_defs import (
    Gate,
    GateType,
    RxGate,
    RyGate,
    RzGate,
    IdentityGate,
)
from qsynth.synthesizer.logical_circuit import GateApplication, LogicalCircuit


# ── Abstract base ──────────────────────────────────────────────────────────────


class OptimizationPass(ABC):
    """
    Strategy interface for a single compiler optimization pass.

    Each pass transforms a :class:`LogicalCircuit` and returns a new
    (possibly shorter/shallower) :class:`LogicalCircuit`.
    Passes must be *semantics-preserving* (unitary-equivalent output).
    """

    @abstractmethod
    def run(self, circuit: LogicalCircuit) -> LogicalCircuit:
        """Apply this pass to *circuit* and return the optimized circuit."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable pass name."""

    def __repr__(self) -> str:
        return f"Pass({self.name})"


# ── Pass Manager ───────────────────────────────────────────────────────────────


class PassManager:
    """
    Sequences multiple :class:`OptimizationPass` instances.

    Parameters
    ----------
    passes    : list of OptimizationPass
    max_iter  : int   – maximum number of full-pipeline repetitions (convergence).
    """

    def __init__(
        self,
        passes: list[OptimizationPass],
        max_iter: int = 10,
    ) -> None:
        self.passes = passes
        self.max_iter = max_iter

    def run(self, circuit: LogicalCircuit) -> LogicalCircuit:
        """
        Repeatedly apply all passes until convergence (gate count unchanged)
        or *max_iter* is reached.
        """
        current = circuit
        for iteration in range(self.max_iter):
            prev_count = current.gate_count()
            for pass_ in self.passes:
                current = pass_.run(current)
            if current.gate_count() == prev_count:
                break  # Converged
        return current

    @classmethod
    def default(cls) -> "PassManager":
        """
        Return a default pass manager suitable for QAOA circuits:
        DeadCode → Merge → Commute → Cancel (×10 iterations).
        """
        return cls(
            passes=[
                DeadCodeEliminationPass(),
                MergeRotationsPass(),
                CommutativityPass(),
                CancellationPass(),
            ],
            max_iter=10,
        )


# ── Helper utilities ───────────────────────────────────────────────────────────

_TOL = 1e-9


def _ops_share_qubits(a: GateApplication, b: GateApplication) -> bool:
    """Return True if gate applications a and b share at least one qubit."""
    return bool(set(a.qubits) & set(b.qubits))


def _matrices_equal(m1: np.ndarray, m2: np.ndarray, tol: float = _TOL) -> bool:
    return bool(np.allclose(m1, m2, atol=tol))


def _is_identity_matrix(m: np.ndarray, tol: float = _TOL) -> bool:
    dim = m.shape[0]
    return _matrices_equal(m, np.eye(dim, dtype=complex), tol)


def _rebuild(original: LogicalCircuit, ops: list[GateApplication]) -> LogicalCircuit:
    """Create a new LogicalCircuit with the same metadata but new operations."""
    new_circuit = LogicalCircuit(n_qubits=original.n_qubits, name=original.name)
    new_circuit.operations = list(ops)
    return new_circuit


# ── Pass 1: Dead Code Elimination ─────────────────────────────────────────────


class DeadCodeEliminationPass(OptimizationPass):
    """
    Drop gate applications whose matrix is the identity (up to tolerance).

    Catches zero-angle rotations and explicit IdentityGate instances.
    """

    def __init__(self, tol: float = _TOL) -> None:
        self._tol = tol

    @property
    def name(self) -> str:
        return "DeadCodeElimination"

    def run(self, circuit: LogicalCircuit) -> LogicalCircuit:
        kept: list[GateApplication] = []
        for op in circuit.operations:
            gate_mat = op.gate.matrix()
            dim = gate_mat.shape[0]
            if not _is_identity_matrix(gate_mat, self._tol):
                kept.append(op)
        return _rebuild(circuit, kept)


# ── Pass 2: Rotation Merging ───────────────────────────────────────────────────


_ROTATION_TYPES = {GateType.RX, GateType.RY, GateType.RZ}
_ROTATION_CONSTRUCTORS: dict[GateType, type] = {
    GateType.RX: RxGate,
    GateType.RY: RyGate,
    GateType.RZ: RzGate,
}


class MergeRotationsPass(OptimizationPass):
    """
    Merge consecutive same-axis rotations on the same qubit.

    Rz(a) · Rz(b) → Rz(a+b).   (Analogously for Rx, Ry.)
    Zero-angle merged gates are dropped.

    The pass makes a single left-to-right sweep.
    Run the PassManager multiple times for full convergence.
    """

    def __init__(self, tol: float = _TOL) -> None:
        self._tol = tol

    @property
    def name(self) -> str:
        return "MergeRotations"

    def run(self, circuit: LogicalCircuit) -> LogicalCircuit:
        ops = list(circuit.operations)
        changed = True
        while changed:
            changed = False
            new_ops: list[GateApplication] = []
            i = 0
            while i < len(ops):
                if i + 1 < len(ops):
                    a, b = ops[i], ops[i + 1]
                    merged = self._try_merge(a, b)
                    if merged is not None:
                        if merged:  # Non-identity merged gate
                            new_ops.append(merged)
                        i += 2
                        changed = True
                        continue
                new_ops.append(ops[i])
                i += 1
            ops = new_ops
        return _rebuild(circuit, ops)

    def _try_merge(
        self,
        a: GateApplication,
        b: GateApplication,
    ) -> GateApplication | None | bool:
        """
        Attempt to merge gate applications a, b.

        Returns:
        - GateApplication   if merged into a non-trivial gate
        - False (falsy)     if merged into identity (both to be dropped)
        - None              if not mergeable
        """
        if a.gate.gate_type not in _ROTATION_TYPES:
            return None
        if b.gate.gate_type not in _ROTATION_TYPES:
            return None
        if a.gate.gate_type != b.gate.gate_type:
            return None
        if a.qubits != b.qubits:
            return None

        pa, pb = a.gate.params()[0], b.gate.params()[0]
        total = pa + pb
        if abs(total) < self._tol:
            return False  # Identity — drop both
        ctor = _ROTATION_CONSTRUCTORS[a.gate.gate_type]
        new_gate = ctor(total)  # type: ignore[call-arg]
        return GateApplication(gate=new_gate, qubits=a.qubits, label="merged")


# ── Pass 3: Commutativity-Based Reordering ────────────────────────────────────


class CommutativityPass(OptimizationPass):
    """
    Bubble-sort adjacent commuting gate pairs to expose cancellation.

    Two adjacent gate applications G₁ G₂ can be swapped (without changing
    the circuit semantics) if:
    a) They act on *disjoint* qubits (trivially commute), or
    b) They share qubits AND G₁·G₂ = G₂·G₁ (checked via matrix product).

    After one sweep, re-running CancellationPass may find more pairs.
    This pass does NOT change semantics — it only reorders.

    To prevent infinite oscillation, swaps are only performed left-to-right
    in a single pass when they bring gates of the *same type* together.
    """

    def __init__(self, tol: float = _TOL) -> None:
        self._tol = tol

    @property
    def name(self) -> str:
        return "Commutativity"

    def run(self, circuit: LogicalCircuit) -> LogicalCircuit:
        """
        Single left-to-right sweep (no infinite loop).
        Gates are only swapped if they commute AND have the same gate_type
        (to expose merge/cancel opportunities without oscillating).
        """
        ops = list(circuit.operations)
        n = len(ops)
        for i in range(n - 1):
            a, b = ops[i], ops[i + 1]
            if self._should_swap(a, b):
                ops[i], ops[i + 1] = b, a
        return _rebuild(circuit, ops)

    def _should_swap(self, a: GateApplication, b: GateApplication) -> bool:
        """
        Return True if swapping a, b is:
        1. Semantically safe (commutes), AND
        2. Productive (brings same-type gates together for merge/cancel).
        """
        # Only swap gates of same type (to expose merge/cancel).
        if a.gate.gate_type != b.gate.gate_type:
            return False
        if not _ops_share_qubits(a, b):
            # Disjoint qubits commute trivially — only swap if same type.
            return True
        # Shared qubits: must verify commutativity via matrix multiplication.
        if a.gate.n_qubits != b.gate.n_qubits or set(a.qubits) != set(b.qubits):
            return False
        AB = a.gate.matrix() @ b.gate.matrix()
        BA = b.gate.matrix() @ a.gate.matrix()
        if not _matrices_equal(AB, BA, self._tol):
            return False
        # Only swap if the swap would bring the same type together.
        return a.gate.gate_type == b.gate.gate_type


# ── Pass 4: Cancellation ──────────────────────────────────────────────────────


class CancellationPass(OptimizationPass):
    """
    Cancel adjacent inverse gate pairs: G · G† ≈ I → remove both.

    Cancellation is checked via matrix product:
        G₁ · G₂ ≈ I  (Frobenius norm < tol)

    Examples:
    - H · H = I
    - CNOT · CNOT = I
    - Rz(θ) · Rz(−θ) = I

    Makes repeated sweeps until no more cancellations are found.
    """

    def __init__(self, tol: float = _TOL) -> None:
        self._tol = tol

    @property
    def name(self) -> str:
        return "Cancellation"

    def run(self, circuit: LogicalCircuit) -> LogicalCircuit:
        ops = list(circuit.operations)
        changed = True
        while changed:
            changed = False
            new_ops: list[GateApplication] = []
            i = 0
            while i < len(ops):
                if i + 1 < len(ops):
                    a, b = ops[i], ops[i + 1]
                    if self._cancels(a, b):
                        i += 2  # Drop both
                        changed = True
                        continue
                new_ops.append(ops[i])
                i += 1
            ops = new_ops
        return _rebuild(circuit, ops)

    def _cancels(self, a: GateApplication, b: GateApplication) -> bool:
        """
        Return True if  a · b = I  (up to tolerance) on the same qubits.
        """
        if a.qubits != b.qubits:
            return False
        if a.gate.n_qubits != b.gate.n_qubits:
            return False
        product = b.gate.matrix() @ a.gate.matrix()
        return _is_identity_matrix(product, self._tol)
