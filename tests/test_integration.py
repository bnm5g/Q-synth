"""
tests/test_integration.py
==========================

End-to-end integration tests for the complete Q-Synth pipeline:
    parse_markowitz → build_qubo → build_hamiltonian → synthesize
    → optimize → evaluate → extract_solution

This is the highest-fidelity test: runs the full 4-asset portfolio
optimization from raw financial inputs to decoded classical solution.
"""

from __future__ import annotations

import numpy as np
import pytest

from qsynth.parser import parse_markowitz, build_qubo
from qsynth.ir import build_hamiltonian
from qsynth.synthesizer import NaiveSynthesizer
from qsynth.compiler import PassManager, HardwareTopology, TopologyMapper
from qsynth.evaluator import (
    StatevectorEvaluator,
    CircuitVerifier,
    extract_solution,
)


class TestFullPipeline:
    """End-to-end integration tests."""

    @pytest.fixture(scope="class")
    def pipeline_4asset(self):
        """Run the full 4-asset pipeline once and return all artifacts."""
        mu = np.array([0.12, 0.18, 0.09, 0.22])
        sigma = np.array([
            [0.06, 0.02, 0.01, 0.03],
            [0.02, 0.09, 0.02, 0.01],
            [0.01, 0.02, 0.04, 0.01],
            [0.03, 0.01, 0.01, 0.12],
        ])
        objective = parse_markowitz(
            mu=mu,
            sigma=sigma,
            risk_aversion=1.0,
            asset_names=["AAPL", "GOOG", "MSFT", "AMZN"],
        )
        qubo = build_qubo(objective)
        hamiltonian = build_hamiltonian(qubo)
        synth = NaiveSynthesizer(n_layers=1)
        naive_circuit = synth.synthesize(hamiltonian)
        pm = PassManager.default()
        optimized_circuit = pm.run(naive_circuit)
        evaluator = StatevectorEvaluator()
        eval_result = evaluator.evaluate(optimized_circuit, hamiltonian)
        solution = extract_solution(eval_result, objective)
        return {
            "objective": objective,
            "qubo": qubo,
            "hamiltonian": hamiltonian,
            "naive_circuit": naive_circuit,
            "optimized_circuit": optimized_circuit,
            "eval_result": eval_result,
            "solution": solution,
        }

    def test_pipeline_runs_without_error(self, pipeline_4asset) -> None:
        """The full pipeline completes without exceptions."""
        assert pipeline_4asset["solution"] is not None

    def test_qubo_correct_size(self, pipeline_4asset) -> None:
        qubo = pipeline_4asset["qubo"]
        assert qubo.Q.shape == (4, 4)
        assert qubo.n_vars == 4

    def test_hamiltonian_hermitian(self, pipeline_4asset) -> None:
        ham = pipeline_4asset["hamiltonian"]
        assert ham.is_hermitian()

    def test_optimized_depth_lte_naive_depth(self, pipeline_4asset) -> None:
        naive = pipeline_4asset["naive_circuit"]
        opt = pipeline_4asset["optimized_circuit"]
        assert opt.depth() <= naive.depth(), (
            f"Optimization increased depth: naive={naive.depth()}, "
            f"optimized={opt.depth()}"
        )

    def test_optimized_gate_count_lte_naive(self, pipeline_4asset) -> None:
        naive = pipeline_4asset["naive_circuit"]
        opt = pipeline_4asset["optimized_circuit"]
        assert opt.gate_count() <= naive.gate_count()

    def test_probabilities_sum_to_one(self, pipeline_4asset) -> None:
        prob_sum = float(np.sum(pipeline_4asset["eval_result"].probabilities))
        assert prob_sum == pytest.approx(1.0, abs=1e-7)

    def test_solution_asset_count(self, pipeline_4asset) -> None:
        solution = pipeline_4asset["solution"]
        assert len(solution.asset_selection) == 4
        assert len(solution.asset_names) == 4
        assert solution.asset_names == ["AAPL", "GOOG", "MSFT", "AMZN"]

    def test_solution_probability_positive(self, pipeline_4asset) -> None:
        solution = pipeline_4asset["solution"]
        assert solution.probability > 0.0

    def test_verifier_hermiticity(self, pipeline_4asset) -> None:
        verifier = CircuitVerifier()
        result = verifier.verify_hamiltonian_hermitian(pipeline_4asset["hamiltonian"])
        assert result.passed

    def test_topology_depth_comparison(self, pipeline_4asset) -> None:
        """
        Demonstrate that topology constraint increases depth:
        All-to-all ≤ linear depth.
        """
        n = pipeline_4asset["optimized_circuit"].n_qubits
        c = pipeline_4asset["optimized_circuit"]

        topo_all = HardwareTopology.all_to_all(n)
        topo_lin = HardwareTopology.linear(n)

        mapper_all = TopologyMapper(topo_all)
        mapper_lin = TopologyMapper(topo_lin)

        mapped_all = mapper_all.map(c)
        mapped_lin = mapper_lin.map(c)

        # Linear topology should have ≥ depth
        assert mapped_lin.depth() >= mapped_all.depth()

    def test_full_pipeline_describe_output(self, pipeline_4asset) -> None:
        """All describe() methods should return non-empty strings."""
        assert len(pipeline_4asset["objective"].describe()) > 0
        assert len(pipeline_4asset["qubo"].describe()) > 0
        assert len(pipeline_4asset["hamiltonian"].describe()) > 0
        assert len(pipeline_4asset["solution"].describe()) > 0
