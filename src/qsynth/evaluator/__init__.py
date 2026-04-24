"""
qsynth.evaluator
================

Evaluates a :class:`LogicalCircuit` by simulating it and verifying
that it correctly minimizes the original financial objective.

Components
----------
- :class:`StatevectorEvaluator`  – simulates the circuit via dense
  statevector and computes expectation value ⟨ψ|H|ψ⟩.
- :class:`CircuitVerifier`       – verifies circuit-Hamiltonian equivalence
  using sparse unitary exponentiation and optional Z3 SMT checks.
- :func:`extract_solution`       – maps the statevector probability
  distribution to the most likely binary asset selection.
"""

from qsynth.evaluator.statevector_evaluator import StatevectorEvaluator, EvaluationResult
from qsynth.evaluator.verifier import CircuitVerifier, VerificationResult
from qsynth.evaluator.solution_extractor import extract_solution, SolutionResult

__all__ = [
    "StatevectorEvaluator",
    "EvaluationResult",
    "CircuitVerifier",
    "VerificationResult",
    "extract_solution",
    "SolutionResult",
]
