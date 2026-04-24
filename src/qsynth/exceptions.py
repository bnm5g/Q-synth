"""
Custom exception hierarchy for the Q-Synth compiler pipeline.

All domain-specific errors inherit from QSynthError so callers can
catch the entire Q-Synth exception surface with a single except clause.
"""

from __future__ import annotations


class QSynthError(Exception):
    """Base exception for all Q-Synth errors."""


# ── Parser ──────────────────────────────────────────────────────────────────


class ParseError(QSynthError):
    """Raised when a financial expression cannot be parsed into an AST."""


class UnsupportedExpressionError(ParseError):
    """Raised when the expression contains constructs the parser cannot handle."""


# ── QUBO / IR ───────────────────────────────────────────────────────────────


class InvalidCovarianceMatrixError(QSynthError):
    """Raised when the covariance matrix is not positive semi-definite."""


class InvalidReturnVectorError(QSynthError):
    """Raised when the expected-return vector has incorrect dimensions."""


class HamiltonianConstructionError(QSynthError):
    """Raised when the Pauli Hamiltonian cannot be constructed from the QUBO."""


# ── Synthesizer ──────────────────────────────────────────────────────────────


class NonUnitarySynthesisError(QSynthError):
    """
    Raised when a synthesized gate matrix is not unitary (U†U ≠ I).

    Attributes
    ----------
    matrix_name : str
        Name/label of the offending matrix.
    frobenius_error : float
        ||U†U - I||_F  — how far from identity the product is.
    """

    def __init__(self, matrix_name: str, frobenius_error: float) -> None:
        self.matrix_name = matrix_name
        self.frobenius_error = frobenius_error
        super().__init__(
            f"Matrix '{matrix_name}' is not unitary: "
            f"||U†U - I||_F = {frobenius_error:.6e} > tolerance."
        )


class DimensionMismatchError(QSynthError):
    """
    Raised when a gate matrix has wrong dimensions for the target qubit count.

    Attributes
    ----------
    expected_dim : int
    actual_dim   : int
    """

    def __init__(self, expected_dim: int, actual_dim: int, context: str = "") -> None:
        self.expected_dim = expected_dim
        self.actual_dim = actual_dim
        msg = (
            f"Dimension mismatch{f' ({context})' if context else ''}: "
            f"expected {expected_dim}×{expected_dim}, got {actual_dim}×{actual_dim}."
        )
        super().__init__(msg)


class UnsynthesizableIRError(QSynthError):
    """Raised when an IR node has no known synthesis rule."""


# ── Compiler / Optimizer ────────────────────────────────────────────────────


class OptimizationError(QSynthError):
    """Raised when a compiler optimization pass encounters an inconsistency."""


class TopologyError(QSynthError):
    """Raised when the logical circuit cannot be mapped to the target topology."""


# ── Evaluator ───────────────────────────────────────────────────────────────


class EvaluationError(QSynthError):
    """Raised when statevector simulation or expectation-value extraction fails."""


class VerificationError(QSynthError):
    """
    Raised when Z3 or numerical verification finds the synthesized circuit
    is not equivalent to the target unitary.
    """
