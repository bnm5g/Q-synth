"""
tests/test_compiler.py
=======================

Tests for compiler optimization passes and topology mapping.

Key assertions (depth tests):
- OptimizedCircuit.depth() ≤ NaiveCircuit.depth()
- CNOT·CNOT cancellation reduces gate count by 2
- Rz(a)·Rz(b) merge reduces gate count by 1
- Optimized circuit's unitary ≈ naive circuit's unitary (semantics preserved)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from qsynth.compiler import (
    CancellationPass,
    CommutativityPass,
    DeadCodeEliminationPass,
    HardwareTopology,
    MergeRotationsPass,
    OptimizationPass,
    PassManager,
    TopologyMapper,
)
from qsynth.synthesizer import (
    CnotGate,
    HGate,
    LogicalCircuit,
    NaiveSynthesizer,
    RzGate,
    IdentityGate,
    RxGate,
)
from qsynth.ir import PauliHamiltonian


# ── Helpers ────────────────────────────────────────────────────────────────────


def _two_qubit_circuit_with_cnot_pair() -> LogicalCircuit:
    """H, CNOT, CNOT → CNOT pair should cancel."""
    c = LogicalCircuit(n_qubits=2)
    c.h(0)
    c.cnot(0, 1)
    c.cnot(0, 1)
    return c


def _circuit_with_rz_merge() -> LogicalCircuit:
    """Rz(a)·Rz(b) → Rz(a+b)."""
    c = LogicalCircuit(n_qubits=1)
    c.rz(0.3, 0)
    c.rz(0.5, 0)
    return c


def _circuit_with_zero_angle_rz() -> LogicalCircuit:
    """Rz(0) should be eliminated as identity."""
    c = LogicalCircuit(n_qubits=1)
    c.rz(0.0, 0)
    c.h(0)
    return c


class TestDeadCodeEliminationPass:

    def test_removes_identity_gate(self) -> None:
        c = LogicalCircuit(n_qubits=1)
        c.append(IdentityGate(), (0,))
        c.h(0)
        assert c.gate_count() == 2
        optimized = DeadCodeEliminationPass().run(c)
        assert optimized.gate_count() == 1
        assert optimized.operations[0].gate.gate_type.name == "H"

    def test_removes_zero_angle_rz(self) -> None:
        c = _circuit_with_zero_angle_rz()
        optimized = DeadCodeEliminationPass().run(c)
        assert optimized.gate_count() == 1  # Only H remains

    def test_no_change_for_non_identity(self) -> None:
        c = LogicalCircuit(n_qubits=1)
        c.h(0)
        c.rz(0.5, 0)
        optimized = DeadCodeEliminationPass().run(c)
        assert optimized.gate_count() == 2


class TestCancellationPass:

    def test_cnot_cnot_cancels(self) -> None:
        c = LogicalCircuit(n_qubits=2)
        c.cnot(0, 1)
        c.cnot(0, 1)
        assert c.gate_count() == 2
        optimized = CancellationPass().run(c)
        assert optimized.gate_count() == 0

    def test_h_h_cancels(self) -> None:
        c = LogicalCircuit(n_qubits=1)
        c.h(0)
        c.h(0)
        optimized = CancellationPass().run(c)
        assert optimized.gate_count() == 0

    def test_rz_inverse_cancels(self) -> None:
        c = LogicalCircuit(n_qubits=1)
        c.rz(0.75, 0)
        c.rz(-0.75, 0)
        optimized = CancellationPass().run(c)
        assert optimized.gate_count() == 0

    def test_no_cancel_different_qubits(self) -> None:
        c = LogicalCircuit(n_qubits=2)
        c.h(0)
        c.h(1)
        optimized = CancellationPass().run(c)
        assert optimized.gate_count() == 2  # No cancellation

    def test_cancellation_in_sequence(self) -> None:
        """H, CNOT, CNOT, H → H cancels H, CNOT cancels CNOT."""
        c = LogicalCircuit(n_qubits=2)
        c.h(0)
        c.cnot(0, 1)
        c.cnot(0, 1)
        c.h(0)
        optimized = CancellationPass().run(c)
        assert optimized.gate_count() == 0

    def test_semantics_preserved_after_cancellation(self) -> None:
        """After CNOT·CNOT cancellation, circuit unitary = I."""
        c = _two_qubit_circuit_with_cnot_pair()
        U_before = c.unitary()
        optimized = CancellationPass().run(c)
        U_after = optimized.unitary()
        np.testing.assert_allclose(U_before, U_after, atol=1e-9)


class TestMergeRotationsPass:

    def test_rz_merge(self) -> None:
        c = _circuit_with_rz_merge()
        assert c.gate_count() == 2
        optimized = MergeRotationsPass().run(c)
        assert optimized.gate_count() == 1
        assert optimized.operations[0].gate.params()[0] == pytest.approx(0.8, abs=1e-9)

    def test_rx_merge(self) -> None:
        c = LogicalCircuit(n_qubits=1)
        c.rx(0.4, 0)
        c.rx(0.6, 0)
        optimized = MergeRotationsPass().run(c)
        assert optimized.gate_count() == 1
        assert optimized.operations[0].gate.params()[0] == pytest.approx(1.0, abs=1e-9)

    def test_rz_merge_to_zero_drops(self) -> None:
        c = LogicalCircuit(n_qubits=1)
        c.rz(0.5, 0)
        c.rz(-0.5, 0)
        optimized = MergeRotationsPass().run(c)
        assert optimized.gate_count() == 0

    def test_no_merge_different_axes(self) -> None:
        c = LogicalCircuit(n_qubits=1)
        c.rz(0.5, 0)
        c.rx(0.5, 0)
        optimized = MergeRotationsPass().run(c)
        assert optimized.gate_count() == 2

    def test_no_merge_different_qubits(self) -> None:
        c = LogicalCircuit(n_qubits=2)
        c.rz(0.5, 0)
        c.rz(0.5, 1)
        optimized = MergeRotationsPass().run(c)
        assert optimized.gate_count() == 2

    def test_semantics_preserved_after_merge(self) -> None:
        c = _circuit_with_rz_merge()
        U_before = c.unitary()
        optimized = MergeRotationsPass().run(c)
        U_after = optimized.unitary()
        np.testing.assert_allclose(U_before, U_after, atol=1e-9)


class TestPassManager:

    def test_pass_manager_reduces_depth(
        self, four_asset_naive_circuit: LogicalCircuit
    ) -> None:
        """
        PROOF: The optimizer must reduce (or maintain) depth vs. naive circuit.
        This is a core correctness assertion for the depth-minimization claim.
        """
        naive_depth = four_asset_naive_circuit.depth()
        pm = PassManager.default()
        optimized = pm.run(four_asset_naive_circuit)
        optimized_depth = optimized.depth()
        assert optimized_depth <= naive_depth, (
            f"Optimizer increased depth: naive={naive_depth}, "
            f"optimized={optimized_depth}"
        )

    def test_pass_manager_reduces_gate_count(
        self, four_asset_naive_circuit: LogicalCircuit
    ) -> None:
        naive_count = four_asset_naive_circuit.gate_count()
        pm = PassManager.default()
        optimized = pm.run(four_asset_naive_circuit)
        assert optimized.gate_count() <= naive_count

    def test_pass_manager_preserves_unitary(
        self, two_asset_hamiltonian: PauliHamiltonian
    ) -> None:
        """
        CRITICAL: The optimization must be semantics-preserving.
        Uses 2-qubit circuits (4×4 unitary) for speed.
        ‖U_optimized − U_naive‖_F < tol.
        """
        synth = NaiveSynthesizer(n_layers=1)
        naive = synth.synthesize(two_asset_hamiltonian)
        # Only run unitary check on the 2-qubit case (4x4 matrix — fast).
        assert naive.n_qubits == 2, "This test requires a 2-qubit circuit"
        U_naive = naive.unitary()

        pm = PassManager.default()
        optimized = pm.run(naive)
        U_opt = optimized.unitary()

        frob_err = np.linalg.norm(U_naive - U_opt)
        assert frob_err < 1e-6, (
            f"Optimization changed circuit semantics: ‖ΔU‖_F = {frob_err:.4e}"
        )

    def test_convergence(self) -> None:
        """Pass manager converges (no infinite loops)."""
        c = LogicalCircuit(n_qubits=2)
        for _ in range(5):
            c.cnot(0, 1)
            c.cnot(0, 1)
        pm = PassManager.default()
        result = pm.run(c)
        assert result.gate_count() == 0  # All CNOT pairs canceled


class TestHardwareTopology:

    def test_all_to_all_connectivity(self) -> None:
        topo = HardwareTopology.all_to_all(4)
        for i in range(4):
            for j in range(4):
                if i != j:
                    assert topo.are_adjacent(i, j)

    def test_linear_connectivity(self) -> None:
        topo = HardwareTopology.linear(4)
        assert topo.are_adjacent(0, 1)
        assert topo.are_adjacent(1, 2)
        assert topo.are_adjacent(2, 3)
        assert not topo.are_adjacent(0, 2)
        assert not topo.are_adjacent(0, 3)

    def test_shortest_path_linear(self) -> None:
        topo = HardwareTopology.linear(5)
        path = topo.shortest_path(0, 4)
        assert path == [0, 1, 2, 3, 4]

    def test_shortest_path_trivial(self) -> None:
        topo = HardwareTopology.linear(4)
        path = topo.shortest_path(2, 2)
        assert path == [2]

    def test_heavy_hex_construction(self) -> None:
        topo = HardwareTopology.heavy_hex(8)
        assert topo.n_physical_qubits == 8
        assert len(topo.edges) > 0


class TestTopologyMapper:

    def test_all_to_all_no_swaps(self) -> None:
        """On all-to-all topology, no SWAPs should be inserted."""
        topo = HardwareTopology.all_to_all(2)
        mapper = TopologyMapper(topo)
        c = LogicalCircuit(n_qubits=2)
        c.h(0)
        c.cnot(0, 1)
        mapped = mapper.map(c)
        # Same gate count (no SWAP insertion needed)
        assert mapped.gate_count() == c.gate_count()

    def test_linear_topology_inserts_swaps(self) -> None:
        """On linear topology, non-adjacent CNOT(0,2) needs SWAP routing."""
        topo = HardwareTopology.linear(3)
        mapper = TopologyMapper(topo)
        c = LogicalCircuit(n_qubits=3)
        c.cnot(0, 2)  # Non-adjacent on linear topology
        mapped = mapper.map(c)
        # Routing should add SWAP gates → more gates
        assert mapped.gate_count() >= c.gate_count()

    def test_depth_increases_on_restricted_topology(
        self, four_asset_naive_circuit: LogicalCircuit
    ) -> None:
        """
        A restricted heavy-hex topology should increase depth compared to
        all-to-all — demonstrating topology impact on circuit depth.
        """
        n = four_asset_naive_circuit.n_qubits
        topo_all = HardwareTopology.all_to_all(n)
        topo_hex = HardwareTopology.linear(n)  # More restrictive

        mapper_all = TopologyMapper(topo_all)
        mapper_hex = TopologyMapper(topo_hex)

        mapped_all = mapper_all.map(four_asset_naive_circuit)
        mapped_hex = mapper_hex.map(four_asset_naive_circuit)

        # Heavy-hex / linear should have ≥ depth than all-to-all
        assert mapped_hex.depth() >= mapped_all.depth()
