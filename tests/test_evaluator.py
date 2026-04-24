"""
tests/test_evaluator.py
========================

Tests for statevector evaluation, circuit verification, and solution extraction.

Covers:
- StatevectorEvaluator returns valid probabilities (sum to 1)
- Expectation value is real and finite
- VerificationResult from Z3 gate identity proofs
- Solution extraction gives valid portfolio metrics
- Numerical circuit equivalence for small circuits
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from qsynth.evaluator import (
    EvaluationResult,
    StatevectorEvaluator,
    CircuitVerifier,
    extract_solution,
    SolutionResult,
)
from qsynth.synthesizer import LogicalCircuit, NaiveSynthesizer, HGate
from qsynth.ir import PauliHamiltonian
from qsynth.parser import parse_markowitz


class TestStatevectorEvaluator:
    """Tests for StatevectorEvaluator."""

    def test_probabilities_sum_to_one(
        self,
        two_asset_hamiltonian: PauliHamiltonian,
    ) -> None:
        synth = NaiveSynthesizer(n_layers=1)
        circuit = synth.synthesize(two_asset_hamiltonian)
        evaluator = StatevectorEvaluator()
        result = evaluator.evaluate(circuit, two_asset_hamiltonian)
        prob_sum = float(np.sum(result.probabilities))
        assert prob_sum == pytest.approx(1.0, abs=1e-8)

    def test_expectation_value_is_real_finite(
        self,
        two_asset_hamiltonian: PauliHamiltonian,
    ) -> None:
        synth = NaiveSynthesizer(n_layers=1)
        circuit = synth.synthesize(two_asset_hamiltonian)
        evaluator = StatevectorEvaluator()
        result = evaluator.evaluate(circuit, two_asset_hamiltonian)
        assert np.isfinite(result.expectation_value)
        assert isinstance(result.expectation_value, float)

    def test_circuit_depth_in_result(
        self,
        two_asset_hamiltonian: PauliHamiltonian,
    ) -> None:
        synth = NaiveSynthesizer(n_layers=1)
        circuit = synth.synthesize(two_asset_hamiltonian)
        evaluator = StatevectorEvaluator()
        result = evaluator.evaluate(circuit, two_asset_hamiltonian)
        assert result.circuit_depth == circuit.depth()
        assert result.gate_count == circuit.gate_count()

    def test_four_asset_evaluation(
        self,
        four_asset_naive_circuit: LogicalCircuit,
        four_asset_hamiltonian: PauliHamiltonian,
    ) -> None:
        evaluator = StatevectorEvaluator()
        result = evaluator.evaluate(four_asset_naive_circuit, four_asset_hamiltonian)
        assert result.n_qubits == 4
        assert len(result.probabilities) == 16  # 2^4

    def test_top_k_states_sorted(
        self,
        two_asset_hamiltonian: PauliHamiltonian,
    ) -> None:
        synth = NaiveSynthesizer(n_layers=1)
        circuit = synth.synthesize(two_asset_hamiltonian)
        evaluator = StatevectorEvaluator()
        result = evaluator.evaluate(circuit, two_asset_hamiltonian)
        top3 = result.top_k_states(3)
        assert len(top3) == 3
        # Probabilities should be in descending order
        probs = [p for _, p in top3]
        assert probs == sorted(probs, reverse=True)

    def test_most_likely_state_is_valid_bitstring(
        self,
        two_asset_hamiltonian: PauliHamiltonian,
    ) -> None:
        synth = NaiveSynthesizer(n_layers=1)
        circuit = synth.synthesize(two_asset_hamiltonian)
        evaluator = StatevectorEvaluator()
        result = evaluator.evaluate(circuit, two_asset_hamiltonian)
        state = result.most_likely_state()
        assert len(state) == 2
        assert all(c in "01" for c in state)

    def test_uniform_superposition_expectation(self) -> None:
        """
        H⊗n |0⟩ is the uniform superposition |+⟩⊗n.
        For H = hZ (single-qubit), ⟨+|hZ|+⟩ = 0.
        """
        ham = PauliHamiltonian(n_qubits=1)
        ham.add_term("Z", 1.0)
        circuit = LogicalCircuit(n_qubits=1)
        circuit.h(0)
        evaluator = StatevectorEvaluator()
        result = evaluator.evaluate(circuit, ham)
        assert result.expectation_value == pytest.approx(0.0, abs=1e-8)


class TestCircuitVerifier:
    """Tests for CircuitVerifier."""

    def test_z3_gate_identities_pass(self) -> None:
        verifier = CircuitVerifier(tol=1e-6)
        result = verifier.verify_gate_identities_z3()
        assert result.passed, f"Z3 verification failed: {result.details}"

    def test_hermiticity_verified(
        self, two_asset_hamiltonian: PauliHamiltonian
    ) -> None:
        verifier = CircuitVerifier()
        result = verifier.verify_hamiltonian_hermitian(two_asset_hamiltonian)
        assert result.passed

    def test_hermiticity_four_asset(
        self, four_asset_hamiltonian: PauliHamiltonian
    ) -> None:
        verifier = CircuitVerifier()
        result = verifier.verify_hamiltonian_hermitian(four_asset_hamiltonian)
        assert result.passed

    def test_verification_result_repr(
        self, two_asset_hamiltonian: PauliHamiltonian
    ) -> None:
        verifier = CircuitVerifier()
        result = verifier.verify_hamiltonian_hermitian(two_asset_hamiltonian)
        assert "VerificationResult" in repr(result)

    def test_numerical_verification_simple_circuit(self) -> None:
        """
        For a 1-qubit H gate with H_Pauli = Z,
        e^{-i Z t} ≠ H in general — this tests that the verifier
        correctly reports FAIL when the circuit doesn't match.
        """
        ham = PauliHamiltonian(n_qubits=1)
        ham.add_term("Z", 1.0)

        # Wrong circuit: just an H gate, not e^{-iZt}
        circuit = LogicalCircuit(n_qubits=1)
        circuit.h(0)

        verifier = CircuitVerifier(tol=1e-4)
        result = verifier.verify_numerical(circuit, ham, evolution_time=1.0)
        # H ≠ e^{-iZt} → should NOT pass
        assert not result.passed
        assert result.frobenius_error > 0.0

    def test_numerical_verification_rz_gate_matches(self) -> None:
        """
        For H = Z and t = 0.5, e^{-i Z t} = Rz(2t) up to global phase.
        """
        t = 0.5
        ham = PauliHamiltonian(n_qubits=1)
        ham.add_term("Z", 1.0)

        # Rz(2t) implements e^{-i t Z} (up to global phase, no constant)
        circuit = LogicalCircuit(n_qubits=1)
        circuit.rz(2 * t, 0)

        verifier = CircuitVerifier(tol=1e-4)
        result = verifier.verify_numerical(circuit, ham, evolution_time=t)
        assert result.passed, f"Rz verification failed: {result.details}"


class TestSolutionExtractor:
    """Tests for extract_solution."""

    def test_solution_has_correct_asset_count(
        self,
        four_asset_naive_circuit: LogicalCircuit,
        four_asset_hamiltonian: PauliHamiltonian,
        four_asset_objective,
    ) -> None:
        evaluator = StatevectorEvaluator()
        eval_result = evaluator.evaluate(four_asset_naive_circuit, four_asset_hamiltonian)
        solution = extract_solution(eval_result, four_asset_objective)
        assert len(solution.asset_selection) == 4
        assert len(solution.asset_names) == 4

    def test_solution_probability_in_range(
        self,
        four_asset_naive_circuit: LogicalCircuit,
        four_asset_hamiltonian: PauliHamiltonian,
        four_asset_objective,
    ) -> None:
        evaluator = StatevectorEvaluator()
        eval_result = evaluator.evaluate(four_asset_naive_circuit, four_asset_hamiltonian)
        solution = extract_solution(eval_result, four_asset_objective)
        assert 0.0 <= solution.probability <= 1.0

    def test_solution_describes_correctly(
        self,
        four_asset_naive_circuit: LogicalCircuit,
        four_asset_hamiltonian: PauliHamiltonian,
        four_asset_objective,
    ) -> None:
        evaluator = StatevectorEvaluator()
        eval_result = evaluator.evaluate(four_asset_naive_circuit, four_asset_hamiltonian)
        solution = extract_solution(eval_result, four_asset_objective)
        desc = solution.describe()
        assert "Q-Synth Solution" in desc
        assert "State" in desc

    def test_solution_state_string_valid(
        self,
        two_asset_hamiltonian: PauliHamiltonian,
        two_asset_objective,
    ) -> None:
        synth = NaiveSynthesizer(n_layers=1)
        circuit = synth.synthesize(two_asset_hamiltonian)
        evaluator = StatevectorEvaluator()
        eval_result = evaluator.evaluate(circuit, two_asset_hamiltonian)
        solution = extract_solution(eval_result, two_asset_objective)
        assert len(solution.state_string) == 2
        assert all(c in "01" for c in solution.state_string)

    def test_top_candidates_ordered_by_probability(
        self,
        four_asset_naive_circuit: LogicalCircuit,
        four_asset_hamiltonian: PauliHamiltonian,
        four_asset_objective,
    ) -> None:
        evaluator = StatevectorEvaluator()
        eval_result = evaluator.evaluate(four_asset_naive_circuit, four_asset_hamiltonian)
        solution = extract_solution(eval_result, four_asset_objective, top_k=5)
        probs = [c["prob"] for c in solution.top_candidates]
        assert probs == sorted(probs, reverse=True)
