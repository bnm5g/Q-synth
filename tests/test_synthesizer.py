"""
tests/test_synthesizer.py
==========================

Tests for the gate definitions, LogicalCircuit, and NaiveSynthesizer.

Covers:
- Gate unitary validation (all gates must be unitary)
- Gate inverse correctness (G · G† = I)
- CNOT · CNOT = I (algebraic cancellation proof)
- LogicalCircuit depth computation
- NaiveSynthesizer produces correct structure
- Circuit unitary is unitary
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from qsynth.synthesizer import (
    HGate,
    RxGate,
    RyGate,
    RzGate,
    CnotGate,
    CrzGate,
    IdentityGate,
    LogicalCircuit,
    NaiveSynthesizer,
    GateApplication,
    GateType,
)
from qsynth.ir import PauliHamiltonian
from qsynth.exceptions import (
    NonUnitarySynthesisError,
    DimensionMismatchError,
    UnsynthesizableIRError,
)


_TOL = 1e-9


def _is_unitary(m: np.ndarray, tol: float = _TOL) -> bool:
    n = m.shape[0]
    return bool(np.allclose(m.conj().T @ m, np.eye(n, dtype=complex), atol=tol))


class TestGateDefs:
    """Tests for all gate implementations."""

    @pytest.mark.parametrize("gate", [
        IdentityGate(),
        HGate(),
        RxGate(0.0),
        RxGate(math.pi / 4),
        RxGate(math.pi),
        RyGate(math.pi / 3),
        RzGate(math.pi / 2),
        RzGate(-math.pi),
        CnotGate(),
        CrzGate(math.pi / 4),
    ])
    def test_gate_is_unitary(self, gate) -> None:
        """Every gate matrix must be unitary."""
        assert _is_unitary(gate.matrix()), (
            f"Gate {gate.label()} is not unitary."
        )

    @pytest.mark.parametrize("gate", [
        HGate(),
        RxGate(0.75),
        RyGate(0.5),
        RzGate(1.2),
        CnotGate(),
        CrzGate(0.9),
    ])
    def test_gate_inverse_is_identity(self, gate) -> None:
        """G · G† = I."""
        inv = gate.inverse()
        product = inv.matrix() @ gate.matrix()
        dim = 2 ** gate.n_qubits
        np.testing.assert_allclose(
            product,
            np.eye(dim, dtype=complex),
            atol=_TOL,
            err_msg=f"{gate.label()} · {inv.label()} ≠ I",
        )

    def test_hadamard_self_inverse(self) -> None:
        """H · H = I (special property of Hadamard)."""
        H = HGate().matrix()
        np.testing.assert_allclose(H @ H, np.eye(2, dtype=complex), atol=_TOL)

    def test_cnot_self_inverse(self) -> None:
        """CNOT · CNOT = I."""
        C = CnotGate().matrix()
        np.testing.assert_allclose(C @ C, np.eye(4, dtype=complex), atol=_TOL)

    def test_rz_composition(self) -> None:
        """Rz(a) · Rz(b) = Rz(a + b)."""
        a, b = 0.3, 0.7
        Rza = RzGate(a).matrix()
        Rzb = RzGate(b).matrix()
        Rzab = RzGate(a + b).matrix()
        np.testing.assert_allclose(Rza @ Rzb, Rzab, atol=_TOL)

    def test_rx_pi_is_pauli_x(self) -> None:
        """Rx(π) = −i·X (up to global phase)."""
        Rx = RxGate(math.pi).matrix()
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        # Rx(π) = cos(π/2)·I − i·sin(π/2)·X = −i·X
        expected = -1j * X
        np.testing.assert_allclose(Rx, expected, atol=_TOL)

    def test_rz_pi_is_pauli_z_up_to_phase(self) -> None:
        """Rz(π) = diag(e^{-iπ/2}, e^{iπ/2}) = diag(-i, +i)."""
        Rz = RzGate(math.pi).matrix()
        expected = np.array([[-1j, 0], [0, 1j]], dtype=complex)
        np.testing.assert_allclose(Rz, expected, atol=_TOL)

    def test_identity_gate(self) -> None:
        I = IdentityGate().matrix()
        np.testing.assert_allclose(I, np.eye(2, dtype=complex), atol=_TOL)

    def test_gate_qubit_arity(self) -> None:
        assert IdentityGate().n_qubits == 1
        assert HGate().n_qubits == 1
        assert RxGate(0.5).n_qubits == 1
        assert CnotGate().n_qubits == 2
        assert CrzGate(0.5).n_qubits == 2

    def test_commutativity_rz_rz(self) -> None:
        """Two Rz gates always commute."""
        rz1 = RzGate(0.3)
        rz2 = RzGate(0.9)
        assert rz1.commutes_with(rz2)

    def test_commutativity_rx_rz_anticommute(self) -> None:
        """Rx and Rz do not commute in general."""
        rx = RxGate(math.pi / 4)
        rz = RzGate(math.pi / 4)
        assert not rx.commutes_with(rz)


class TestLogicalCircuit:
    """Tests for LogicalCircuit."""

    def test_empty_circuit_depth_zero(self) -> None:
        c = LogicalCircuit(n_qubits=2)
        assert c.depth() == 0

    def test_single_gate_depth_one(self) -> None:
        c = LogicalCircuit(n_qubits=2)
        c.h(0)
        assert c.depth() == 1

    def test_parallel_gates_depth_one(self) -> None:
        """H on qubit 0 and H on qubit 1 in parallel → depth = 1."""
        c = LogicalCircuit(n_qubits=2)
        c.h(0)
        c.h(1)
        assert c.depth() == 1

    def test_sequential_gates_depth(self) -> None:
        """Two sequential gates on same qubit → depth = 2."""
        c = LogicalCircuit(n_qubits=1)
        c.h(0)
        c.rz(0.5, 0)
        assert c.depth() == 2

    def test_cnot_depth(self) -> None:
        """H + CNOT → depth = 2 (CNOT uses both qubits)."""
        c = LogicalCircuit(n_qubits=2)
        c.h(0)
        c.cnot(0, 1)
        assert c.depth() == 2

    def test_gate_count(self) -> None:
        c = LogicalCircuit(n_qubits=2)
        c.h(0)
        c.h(1)
        c.cnot(0, 1)
        assert c.gate_count() == 3

    def test_two_qubit_gate_count(self) -> None:
        c = LogicalCircuit(n_qubits=2)
        c.h(0)
        c.cnot(0, 1)
        c.cnot(1, 0)
        assert c.two_qubit_gate_count() == 2

    def test_invalid_qubit_index_raises(self) -> None:
        c = LogicalCircuit(n_qubits=2)
        with pytest.raises(ValueError):
            c.h(2)  # Only qubits 0 and 1 exist

    def test_unitary_single_hadamard(self) -> None:
        """Single H gate circuit unitary = H matrix."""
        c = LogicalCircuit(n_qubits=1)
        c.h(0)
        U = c.unitary()
        expected = HGate().matrix()
        np.testing.assert_allclose(U, expected, atol=_TOL)

    def test_unitary_is_unitary(self, four_asset_naive_circuit: LogicalCircuit) -> None:
        """The full QAOA circuit unitary must be unitary."""
        U = four_asset_naive_circuit.unitary()
        dim = 2 ** four_asset_naive_circuit.n_qubits
        np.testing.assert_allclose(
            U.conj().T @ U,
            np.eye(dim, dtype=complex),
            atol=1e-6,
            err_msg="QAOA circuit unitary U†U ≠ I",
        )

    def test_unitary_cnot_twice_is_identity(self) -> None:
        """CNOT · CNOT = I."""
        c = LogicalCircuit(n_qubits=2)
        c.cnot(0, 1)
        c.cnot(0, 1)
        U = c.unitary()
        np.testing.assert_allclose(U, np.eye(4, dtype=complex), atol=_TOL)

    def test_to_qiskit_runs(self, four_asset_naive_circuit: LogicalCircuit) -> None:
        from qiskit.circuit import QuantumCircuit
        qc = four_asset_naive_circuit.to_qiskit()
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == four_asset_naive_circuit.n_qubits

    def test_circuit_repr(self) -> None:
        c = LogicalCircuit(n_qubits=3, name="test")
        assert "LogicalCircuit" in repr(c)


class TestNaiveSynthesizer:
    """Tests for NaiveSynthesizer."""

    def test_synthesize_two_qubits(self, two_asset_hamiltonian: PauliHamiltonian) -> None:
        synth = NaiveSynthesizer(n_layers=1)
        circuit = synth.synthesize(two_asset_hamiltonian)
        assert circuit.n_qubits == 2
        assert circuit.gate_count() > 0

    def test_synthesize_four_qubits(
        self, four_asset_hamiltonian: PauliHamiltonian
    ) -> None:
        synth = NaiveSynthesizer(n_layers=1)
        circuit = synth.synthesize(four_asset_hamiltonian)
        assert circuit.n_qubits == 4

    def test_circuit_starts_with_hadamards(
        self, two_asset_hamiltonian: PauliHamiltonian
    ) -> None:
        synth = NaiveSynthesizer(n_layers=1)
        circuit = synth.synthesize(two_asset_hamiltonian)
        n = circuit.n_qubits
        # First n gates should be H gates
        for i in range(n):
            assert circuit.operations[i].gate.gate_type == GateType.H

    def test_multi_layer_has_more_gates(
        self, two_asset_hamiltonian: PauliHamiltonian
    ) -> None:
        synth1 = NaiveSynthesizer(n_layers=1)
        synth2 = NaiveSynthesizer(n_layers=2)
        c1 = synth1.synthesize(two_asset_hamiltonian)
        c2 = synth2.synthesize(two_asset_hamiltonian)
        assert c2.gate_count() > c1.gate_count()

    def test_invalid_n_layers_raises(self) -> None:
        with pytest.raises(ValueError):
            NaiveSynthesizer(n_layers=0)

    def test_gamma_beta_list_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            NaiveSynthesizer(n_layers=2, gamma=[0.1], beta=0.2)

    def test_non_ising_hamiltonian_raises(self) -> None:
        """NaiveSynthesizer should reject X-type operators."""
        ham = PauliHamiltonian(n_qubits=2)
        ham.add_term("IX", 1.0)  # X term is not Ising
        synth = NaiveSynthesizer()
        with pytest.raises(UnsynthesizableIRError):
            synth.synthesize(ham)

    def test_synthesized_circuit_is_unitary(
        self, two_asset_hamiltonian: PauliHamiltonian
    ) -> None:
        """The synthesized circuit's unitary must satisfy U†U = I."""
        synth = NaiveSynthesizer(n_layers=1)
        circuit = synth.synthesize(two_asset_hamiltonian)
        U = circuit.unitary()
        n = circuit.n_qubits
        np.testing.assert_allclose(
            U.conj().T @ U,
            np.eye(2**n, dtype=complex),
            atol=1e-6,
        )
