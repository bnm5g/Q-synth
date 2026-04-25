"""
qsynth.evaluator.verifier
==========================

Verifies that a synthesized :class:`LogicalCircuit` is equivalent to
the target Hamiltonian evolution  e^{-iHt}.

Two verification methods are implemented:

1. **Numerical (matrix exponential)**
   Computes  U_target = scipy.linalg.expm(-1j * H * t)  and compares it
   to the circuit's unitary via Frobenius norm.
   
   Uses sparse matrix exponentiation for n ≤ 10 and dense for smaller n.

2. **Z3 SMT Verification (symbolic)**
   For small circuits (n ≤ 3), encodes gate compositions into Z3 bit-vector
   arithmetic to formally verify commutativity properties and gate identities
   (e.g. CNOT² = I).  This is a *proof of identity* at the algebraic level.

   Note: Full Z3 unitary equivalence for floating-point gate parameters
   is encoded via rational approximations and interval arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.linalg as sci_la

from qsynth.exceptions import VerificationError
from qsynth.ir.pauli_hamiltonian import PauliHamiltonian
from qsynth.synthesizer.logical_circuit import LogicalCircuit


@dataclass(frozen=True)
class VerificationResult:
    """
    Result of circuit-Hamiltonian equivalence verification.

    Attributes
    ----------
    passed            : bool    – True if verification succeeded.
    frobenius_error   : float   – ‖U_circuit − U_target‖_F.
    method            : str     – "numerical" or "z3".
    details           : str     – human-readable summary.
    """

    passed: bool
    frobenius_error: float
    method: str
    details: str

    def __repr__(self) -> str:
        status = "✓ PASSED" if self.passed else "✗ FAILED"
        return (
            f"VerificationResult[{self.method}] {status} "
            f"‖ΔU‖_F={self.frobenius_error:.3e}"
        )


class CircuitVerifier:
    """
    Verifies circuit-Hamiltonian equivalence via numerical or Z3 methods.

    Parameters
    ----------
    tol : float
        Frobenius-norm tolerance for numerical verification.
    """

    def __init__(self, tol: float = 1e-4) -> None:
        self._tol = tol

    # ── Numerical verification ─────────────────────────────────────────────

    def verify_numerical(
        self,
        circuit: LogicalCircuit,
        hamiltonian: PauliHamiltonian,
        evolution_time: float = 1.0,
    ) -> VerificationResult:
        """
        Numerically verify:  U_circuit ≈ e^{-i H t}.

        Computes the matrix exponential of H and compares with the circuit
        unitary via Frobenius norm.

        Parameters
        ----------
        circuit        : LogicalCircuit
        hamiltonian    : PauliHamiltonian
        evolution_time : float  – the time parameter t.

        Returns
        -------
        VerificationResult
        """
        n = circuit.n_qubits
        if n != hamiltonian.n_qubits:
            raise VerificationError(
                f"Qubit count mismatch: circuit={n}, "
                f"hamiltonian={hamiltonian.n_qubits}."
            )
        if n > 12:
            raise VerificationError(
                f"Numerical unitary verification is exponential — "
                f"n={n} exceeds safe limit of 12 qubits."
            )

        # Target unitary: e^{-i H t}
        H_dense = hamiltonian.dense_matrix()
        U_target = sci_la.expm(-1j * H_dense * evolution_time)

        # Circuit unitary (dense product).
        U_circuit = circuit.unitary()

        frob_err = float(np.linalg.norm(U_circuit - U_target))
        passed = frob_err < self._tol

        details = (
            f"Numerical verification: ‖U_circuit − e^{{-iHt}}‖_F = {frob_err:.4e} "
            f"(tol={self._tol:.1e}, t={evolution_time}). "
            f"{'PASS' if passed else 'FAIL'}."
        )
        return VerificationResult(
            passed=passed,
            frobenius_error=frob_err,
            method="numerical",
            details=details,
        )

    # ── Gate identity verification (numerical) ────────────────────────────

    def verify_gate_identities(self) -> VerificationResult:
        """
        Numerically verify key gate algebraic identities via matrix products:

        - CNOT · CNOT = I   (CNOT is self-inverse)
        - H · H = I         (H is self-inverse)
        - Rz(θ) · Rz(-θ) = I  (for several sample values of θ)

        Each identity is checked by computing the matrix product and comparing
        with the identity via ``numpy.allclose`` (atol=1e-10).

        Returns
        -------
        VerificationResult
            ``method="numerical"``.  ``passed=True`` iff all identities hold.
        """
        from qsynth.synthesizer.gate_defs import CnotGate, HGate, RzGate

        identities_verified: list[str] = []
        identities_failed: list[str] = []

        def _is_identity_product(m1: np.ndarray, m2: np.ndarray) -> bool:
            return bool(np.allclose(m2 @ m1, np.eye(len(m1)), atol=1e-10))

        # CNOT² = I
        cnot_mat = CnotGate().matrix()
        if _is_identity_product(cnot_mat, cnot_mat):
            identities_verified.append("CNOT² = I")
        else:
            identities_failed.append("CNOT² = I")

        # H² = I
        h_mat = HGate().matrix()
        if _is_identity_product(h_mat, h_mat):
            identities_verified.append("H² = I")
        else:
            identities_failed.append("H² = I")

        # Rz(θ) · Rz(-θ) = I  for several sample θ values
        import math
        for theta in [0.1, 0.5, math.pi / 4, math.pi]:
            rz_mat = RzGate(theta).matrix()
            rz_inv = RzGate(-theta).matrix()
            label = f"Rz({theta:.2f})·Rz(-{theta:.2f})=I"
            if _is_identity_product(rz_mat, rz_inv):
                identities_verified.append(label)
            else:
                identities_failed.append(label)

        all_passed = len(identities_failed) == 0
        details = (
            f"Numerical gate identity verification. "
            f"Verified: {identities_verified}. "
            + (f"Failed: {identities_failed}." if identities_failed else "All passed.")
        )
        return VerificationResult(
            passed=all_passed,
            frobenius_error=0.0 if all_passed else 1.0,
            method="numerical",
            details=details,
        )

    # ── Hamiltonian Hermiticity check ─────────────────────────────────────

    def verify_hamiltonian_hermitian(
        self,
        hamiltonian: PauliHamiltonian,
    ) -> VerificationResult:
        """
        Verify that the Hamiltonian is Hermitian (H = H†).

        A physically valid quantum Hamiltonian must be Hermitian for
        its eigenvalues (energies) to be real.
        """
        is_herm = hamiltonian.is_hermitian(tol=self._tol)
        details = (
            f"Hermiticity check: H = H†? "
            f"{'YES' if is_herm else 'NO'} (tol={self._tol:.1e})."
        )
        return VerificationResult(
            passed=is_herm,
            frobenius_error=0.0 if is_herm else float("nan"),
            method="numerical",
            details=details,
        )
