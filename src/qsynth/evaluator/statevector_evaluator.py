"""
qsynth.evaluator.statevector_evaluator
========================================

Simulates a :class:`LogicalCircuit` using Qiskit's
:class:`~qiskit.primitives.StatevectorEstimator` and computes the
expectation value  ⟨ψ|H|ψ⟩  with respect to a :class:`PauliHamiltonian`.

For small circuits (n ≤ 12), the dense statevector simulation is exact
and runs efficiently on a laptop.

EvaluationResult
----------------
A frozen dataclass holding:
- statevector        : np.ndarray          – final |ψ⟩ vector.
- expectation_value  : float               – ⟨ψ|H|ψ⟩.
- probabilities      : np.ndarray          – |⟨bᵢ|ψ⟩|² for each basis state.
- circuit_depth      : int                 – depth of the evaluated circuit.
- gate_count         : int                 – number of gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp_sparse
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import Statevector

from qsynth.exceptions import EvaluationError
from qsynth.ir.pauli_hamiltonian import PauliHamiltonian
from qsynth.synthesizer.logical_circuit import LogicalCircuit


@dataclass(frozen=True)
class EvaluationResult:
    """
    Result of evaluating a quantum circuit against a Hamiltonian.

    Attributes
    ----------
    statevector       : np.ndarray  – complex amplitude vector |ψ⟩.
    expectation_value : float       – ⟨ψ|H|ψ⟩.
    probabilities     : np.ndarray  – |⟨bᵢ|ψ⟩|², shape (2^n,).
    circuit_depth     : int
    gate_count        : int
    n_qubits          : int
    """

    statevector: np.ndarray
    expectation_value: float
    probabilities: np.ndarray
    circuit_depth: int
    gate_count: int
    n_qubits: int

    def top_k_states(self, k: int = 5) -> list[tuple[str, float]]:
        """
        Return the k most probable computational basis states.

        Returns list of (bit_string, probability).
        """
        n = self.n_qubits
        sorted_indices = np.argsort(self.probabilities)[::-1]
        result = []
        for idx in sorted_indices[:k]:
            bit_str = format(idx, f"0{n}b")
            result.append((bit_str, float(self.probabilities[idx])))
        return result

    def most_likely_state(self) -> str:
        """Return the most probable computational basis state as a bit string."""
        idx = int(np.argmax(self.probabilities))
        return format(idx, f"0{self.n_qubits}b")


class StatevectorEvaluator:
    """
    Evaluates a :class:`LogicalCircuit` via statevector simulation.

    Uses Qiskit's :class:`~qiskit.primitives.StatevectorEstimator` for
    expectation values and :class:`~qiskit.quantum_info.Statevector` for
    amplitude extraction.

    Parameters
    ----------
    use_sparse : bool
        If True, uses sparse Hamiltonian matrix for expectation value
        (faster for large n, same result).
    """

    def __init__(self, use_sparse: bool = True) -> None:
        self._use_sparse = use_sparse

    def evaluate(
        self,
        circuit: LogicalCircuit,
        hamiltonian: PauliHamiltonian,
    ) -> EvaluationResult:
        """
        Simulate *circuit* and compute ⟨H⟩.

        Parameters
        ----------
        circuit     : LogicalCircuit
        hamiltonian : PauliHamiltonian

        Returns
        -------
        EvaluationResult

        Raises
        ------
        EvaluationError
            On dimension mismatch or simulation failure.
        """
        if circuit.n_qubits != hamiltonian.n_qubits:
            raise EvaluationError(
                f"Circuit has {circuit.n_qubits} qubits but Hamiltonian "
                f"has {hamiltonian.n_qubits} qubits."
            )

        n = circuit.n_qubits

        # ── Convert to Qiskit circuit ─────────────────────────────────────
        qc = circuit.to_qiskit()

        # ── Extract statevector ───────────────────────────────────────────
        try:
            sv = Statevector(qc)
            amps = sv.data  # complex np.ndarray, shape (2^n,)
        except Exception as exc:
            raise EvaluationError(f"Statevector simulation failed: {exc}") from exc

        probs = np.abs(amps) ** 2

        # ── Compute expectation value ⟨ψ|H|ψ⟩ ────────────────────────────
        exp_val = self._expectation_value(amps, hamiltonian, n)

        return EvaluationResult(
            statevector=amps,
            expectation_value=float(exp_val),
            probabilities=probs,
            circuit_depth=circuit.depth(),
            gate_count=circuit.gate_count(),
            n_qubits=n,
        )

    def _expectation_value(
        self,
        amps: np.ndarray,
        hamiltonian: PauliHamiltonian,
        n: int,
    ) -> float:
        """
        Compute ⟨ψ|H|ψ⟩ directly from the amplitude vector.

        Uses the sparse Hamiltonian matrix for efficiency.
        """
        if self._use_sparse and n <= 14:
            H_sp = hamiltonian.sparse_matrix()
            Hpsi = H_sp.dot(amps)
        else:
            H = hamiltonian.dense_matrix()
            Hpsi = H @ amps
        ev = float(np.real(np.conj(amps) @ Hpsi))
        return ev

    def evaluate_with_qiskit_estimator(
        self,
        circuit: LogicalCircuit,
        hamiltonian: PauliHamiltonian,
    ) -> float:
        """
        Alternative: use Qiskit's primitive :class:`StatevectorEstimator`.

        Returns only the expectation value (float).
        """
        qc = circuit.to_qiskit()
        op = hamiltonian.to_qiskit()
        estimator = StatevectorEstimator()
        job = estimator.run([(qc, op)])
        result = job.result()
        return float(result[0].data.evs)
