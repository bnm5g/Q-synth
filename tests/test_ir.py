"""
tests/test_ir.py
================

Unit tests for the Intermediate Representation (IR) layer.

Covers:
- QUBO → Ising mapping correctness (analytical spot checks)
- PauliHamiltonian construction and validation
- Hermiticity of the assembled Hamiltonian
- Sparse vs dense matrix consistency
- Qiskit SparsePauliOp conversion
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp_sparse

from qsynth.ir import (
    PauliHamiltonian,
    PauliTerm,
    build_hamiltonian,
    qubo_to_ising,
)
from qsynth.ir.pauli_hamiltonian import PAULI_MATRICES
from qsynth.parser import build_qubo, parse_markowitz
from qsynth.exceptions import HamiltonianConstructionError


class TestPauliTerm:
    """Tests for PauliTerm."""

    def test_valid_pauli_string(self) -> None:
        t = PauliTerm("IZ", 1.0)
        assert t.n_qubits == 2

    def test_invalid_pauli_character(self) -> None:
        with pytest.raises(HamiltonianConstructionError):
            PauliTerm("AZ", 1.0)

    def test_identity_term(self) -> None:
        t = PauliTerm("II", 0.5)
        assert t.is_identity

    def test_matrix_z_single_qubit(self) -> None:
        t = PauliTerm("Z", 1.0)
        expected = np.array([[1, 0], [0, -1]], dtype=complex)
        np.testing.assert_allclose(t.matrix(), expected)

    def test_matrix_iz_two_qubit(self) -> None:
        """IZ = I ⊗ Z in Qiskit convention (qubit 0 = rightmost)."""
        t = PauliTerm("IZ", 1.0)
        # Qiskit: string[0]='I' → qubit 1, string[1]='Z' → qubit 0
        # Dense: Z ⊗ I in standard ordering → kron(I, Z)
        Z = PAULI_MATRICES["Z"]
        I = PAULI_MATRICES["I"]
        expected = np.kron(I, Z)  # Z acts on qubit 0 (rightmost)
        np.testing.assert_allclose(t.matrix(), expected, atol=1e-10)

    def test_coefficient_scaling(self) -> None:
        t = PauliTerm("Z", 2.5)
        Z = PAULI_MATRICES["Z"]
        np.testing.assert_allclose(t.matrix(), 2.5 * Z, atol=1e-10)

    def test_sparse_matrix_consistent(self) -> None:
        t = PauliTerm("ZZ", 0.75)
        dense = t.matrix()
        sparse = t.sparse_matrix().toarray()
        np.testing.assert_allclose(dense, sparse, atol=1e-10)

    def test_commutes_with_self(self) -> None:
        t = PauliTerm("ZZ", 1.0)
        assert t.commutes_with(t)

    def test_z_and_x_anticommute(self) -> None:
        tz = PauliTerm("Z", 1.0)
        tx = PauliTerm("X", 1.0)
        assert not tz.commutes_with(tx)

    def test_qiskit_conversion(self) -> None:
        from qiskit.quantum_info import SparsePauliOp
        t = PauliTerm("ZZ", 0.5)
        op = t.to_qiskit()
        assert isinstance(op, SparsePauliOp)


class TestPauliHamiltonian:
    """Tests for PauliHamiltonian assembly."""

    def test_empty_hamiltonian_zero_terms(self) -> None:
        h = PauliHamiltonian(n_qubits=2)
        assert h.n_terms() == 0

    def test_add_term_correct_qubit_count(self) -> None:
        h = PauliHamiltonian(n_qubits=2)
        h.add_term("IZ", 1.0)
        assert len(h) == 1

    def test_add_term_wrong_qubit_count_raises(self) -> None:
        h = PauliHamiltonian(n_qubits=2)
        with pytest.raises(HamiltonianConstructionError):
            h.add_term("ZZZ", 1.0)  # 3-qubit term in 2-qubit Hamiltonian

    def test_simplify_combines_duplicates(self) -> None:
        h = PauliHamiltonian(n_qubits=1)
        h.add_term("Z", 1.0)
        h.add_term("Z", 2.0)
        simplified = h.simplify()
        assert len(simplified) == 1
        assert simplified.terms[0].coefficient == pytest.approx(3.0, abs=1e-12)

    def test_simplify_drops_near_zero(self) -> None:
        h = PauliHamiltonian(n_qubits=1)
        h.add_term("Z", 1.0)
        h.add_term("Z", -1.0)
        simplified = h.simplify()
        assert len(simplified) == 0

    def test_dense_matrix_single_z(self) -> None:
        h = PauliHamiltonian(n_qubits=1)
        h.add_term("Z", 1.5)
        H = h.dense_matrix()
        expected = np.array([[1.5, 0], [0, -1.5]], dtype=complex)
        np.testing.assert_allclose(H, expected, atol=1e-10)

    def test_dense_vs_sparse_consistency(self, two_asset_hamiltonian: PauliHamiltonian) -> None:
        dense = two_asset_hamiltonian.dense_matrix()
        sparse = two_asset_hamiltonian.sparse_matrix().toarray()
        np.testing.assert_allclose(dense, sparse, atol=1e-10)

    def test_hermiticity(self, two_asset_hamiltonian: PauliHamiltonian) -> None:
        """Ising Hamiltonians (real Z terms) must be Hermitian."""
        assert two_asset_hamiltonian.is_hermitian()

    def test_hermiticity_four_asset(
        self, four_asset_hamiltonian: PauliHamiltonian
    ) -> None:
        assert four_asset_hamiltonian.is_hermitian()

    def test_qiskit_conversion_round_trip(
        self, two_asset_hamiltonian: PauliHamiltonian
    ) -> None:
        """SparsePauliOp → dense matrix ≈ PauliHamiltonian dense matrix."""
        from qiskit.quantum_info import SparsePauliOp
        op = two_asset_hamiltonian.to_qiskit()
        op_dense = np.array(op.to_matrix())
        ham_dense = two_asset_hamiltonian.dense_matrix()
        np.testing.assert_allclose(op_dense, ham_dense, atol=1e-8)

    def test_constant_included_in_dense(self) -> None:
        h = PauliHamiltonian(n_qubits=1, constant=5.0)
        h.add_term("Z", 1.0)
        H = h.dense_matrix()
        # Diagonal: 5.0 + 1.0 = 6.0 and 5.0 - 1.0 = 4.0
        assert H[0, 0] == pytest.approx(6.0, abs=1e-10)
        assert H[1, 1] == pytest.approx(4.0, abs=1e-10)


class TestIsingMapper:
    """Tests for qubo_to_ising and build_hamiltonian."""

    def test_ising_coefficients_single_asset(self) -> None:
        """
        For 1 asset, 1x1 QUBO Q=[[q]], the Ising model should give:
            h[0] = -Q[0,0]/2  (from linear term Q[0,0]·x₀)
            J = empty
            offset = Q[0,0]/2 + const
        """
        mu = np.array([0.20])
        sigma = np.array([[0.10]])
        obj = parse_markowitz(mu=mu, sigma=sigma, risk_aversion=1.0)
        qubo = build_qubo(obj)
        h, J, offset = qubo_to_ising(qubo)
        Q = qubo.Q
        assert h[0] == pytest.approx(-Q[0, 0] / 2.0, abs=1e-10)
        assert offset == pytest.approx(Q[0, 0] / 2.0 + qubo.constant, abs=1e-10)

    def test_ising_hamiltonian_term_count(
        self, two_asset_hamiltonian: PauliHamiltonian
    ) -> None:
        """2-asset QUBO → at most 2 Z terms + 1 ZZ term."""
        total = len(two_asset_hamiltonian)
        assert total <= 3  # 2 Z + 1 ZZ (some may be zero and dropped)

    def test_ising_hamiltonian_no_x_y_terms(
        self, four_asset_hamiltonian: PauliHamiltonian
    ) -> None:
        """Ising Hamiltonian from QUBO should have only Z-type Pauli operators."""
        for term in four_asset_hamiltonian:
            for c in term.pauli_str:
                assert c in ("I", "Z"), (
                    f"Non-Z operator '{c}' found in Ising Hamiltonian term "
                    f"'{term.pauli_str}'."
                )

    def test_ising_energy_vs_qubo_energy(self) -> None:
        """
        For all 4 binary assignments of 2-asset QUBO, the Ising energy
        (computed from h, J, offset) must exactly match the QUBO energy f(x).

        The Ising energy for state |s⟩ where sᵢ ∈ {0,1} is:
            E(s) = Σᵢ hᵢ·(1-2sᵢ) + Σᵢ<ⱼ Jᵢⱼ·(1-2sᵢ)(1-2sⱼ) + offset
        which should equal xᵀQx + const  for  xᵢ = sᵢ.
        """
        mu = np.array([0.10, 0.20])
        sigma = np.array([[0.05, 0.01], [0.01, 0.10]])
        obj = parse_markowitz(mu=mu, sigma=sigma, risk_aversion=1.0)
        qubo = build_qubo(obj)
        h, J, offset = qubo_to_ising(qubo)
        n = 2

        for bits in range(4):
            x = np.array([(bits >> i) & 1 for i in range(n)], dtype=float)
            sigma_z = 1.0 - 2.0 * x  # σᶻ = +1 for x=0, -1 for x=1

            ising_energy = offset
            for i in range(n):
                ising_energy += h[i] * sigma_z[i]
            for i in range(n):
                for j_idx in range(i + 1, n):
                    ising_energy += J[i, j_idx] * sigma_z[i] * sigma_z[j_idx]

            qubo_energy = qubo.energy(x)
            assert ising_energy == pytest.approx(qubo_energy, abs=1e-6), (
                f"Ising energy {ising_energy:.8f} ≠ QUBO energy {qubo_energy:.8f} "
                f"for x={x} (bits={bits:02b})"
            )
