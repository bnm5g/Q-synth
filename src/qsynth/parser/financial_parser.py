"""
qsynth.parser.financial_parser
================================

Parses a Markowitz Mean-Variance portfolio objective into a typed AST.

Mathematical formulation
------------------------
The continuous Markowitz objective is:

    min_w  q·½·wᵀΣw − μᵀw

where w ∈ {0,1}ⁿ (binary asset selection), Σ is the covariance matrix of
asset returns, μ is the expected-return vector, and q ≥ 0 is the risk-
aversion coefficient.

This module:
1. Validates inputs (PSD check on Σ, dimension checks on μ).
2. Builds the AST from those inputs.
3. Exposes the convenience function :func:`parse_markowitz`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.linalg as la
import sympy as sp

from qsynth.exceptions import InvalidCovarianceMatrixError, InvalidReturnVectorError
from qsynth.parser.ast_nodes import (
    ASTNode,
    BinaryOp,
    BinaryOpKind,
    Constant,
    MatrixExpr,
    QuadraticForm,
    UnaryOp,
    UnaryOpKind,
    VariableVector,
)


# ── Data container ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FinancialObjective:
    """
    Container for a parsed Markowitz portfolio objective.

    Attributes
    ----------
    n_assets       : int                – number of binary asset variables.
    mu             : np.ndarray         – expected return vector (n,).
    sigma          : np.ndarray         – covariance matrix (n, n).
    risk_aversion  : float              – risk-aversion parameter q ≥ 0.
    budget         : Optional[int]      – optional cardinality constraint Σxᵢ=B.
    penalty        : float              – Lagrange penalty for budget constraint.
    asset_names    : tuple[str, ...]    – human-readable asset labels.
    ast            : ASTNode            – root of the symbolic AST.
    variable_vec   : VariableVector     – ordered binary variable vector.
    """

    n_assets: int
    mu: np.ndarray = field(compare=False)
    sigma: np.ndarray = field(compare=False)
    risk_aversion: float = 1.0
    budget: Optional[int] = None
    penalty: float = 10.0
    asset_names: tuple[str, ...] = field(default_factory=tuple)
    ast: ASTNode = field(compare=False, default=None)  # type: ignore[assignment]
    variable_vec: VariableVector = field(default=None)  # type: ignore[assignment]

    def describe(self) -> str:
        """Return a human-readable summary of the objective."""
        lines = [
            "FinancialObjective",
            f"  Assets       : {self.n_assets}  {list(self.asset_names)}",
            f"  μ (returns)  : {np.round(self.mu, 4)}",
            f"  Σ diagonal   : {np.round(np.diag(self.sigma), 4)}",
            f"  Risk aversion: {self.risk_aversion}",
            f"  Budget       : {self.budget}",
        ]
        return "\n".join(lines)


# ── Validation helpers ────────────────────────────────────────────────────────


def _validate_covariance(sigma: np.ndarray, tol: float = 1e-8) -> None:
    """
    Ensure Σ is square, symmetric, and positive semi-definite.

    Raises
    ------
    InvalidCovarianceMatrixError
    """
    n = sigma.shape[0]
    if sigma.ndim != 2 or sigma.shape[1] != n:
        raise InvalidCovarianceMatrixError(
            f"Σ must be a 2-D square matrix, got shape {sigma.shape}."
        )
    if not np.allclose(sigma, sigma.T, atol=tol):
        raise InvalidCovarianceMatrixError(
            "Σ is not symmetric: max asymmetry = "
            f"{np.max(np.abs(sigma - sigma.T)):.3e}."
        )
    eigvals = la.eigvalsh(sigma)
    min_eig = float(eigvals.min())
    if min_eig < -tol:
        raise InvalidCovarianceMatrixError(
            f"Σ is not positive semi-definite: smallest eigenvalue = {min_eig:.3e}."
        )


def _validate_returns(mu: np.ndarray, n: int) -> None:
    """
    Ensure μ is a 1-D vector of length n.

    Raises
    ------
    InvalidReturnVectorError
    """
    if mu.ndim != 1:
        raise InvalidReturnVectorError(
            f"μ must be a 1-D array, got shape {mu.shape}."
        )
    if len(mu) != n:
        raise InvalidReturnVectorError(
            f"μ has {len(mu)} elements but Σ has {n} assets."
        )


# ── AST builder ───────────────────────────────────────────────────────────────


def _build_markowitz_ast(
    variables: VariableVector,
    mu: np.ndarray,
    sigma: np.ndarray,
    risk_aversion: float,
    budget: Optional[int],
    penalty: float,
) -> ASTNode:
    """
    Build the AST for:

        q · ½·xᵀΣx  −  μᵀx  [+ penalty·(Σxᵢ − B)²]

    Returns the root :class:`ASTNode`.
    """
    # ── risk term: q·½·xᵀΣx ──────────────────────────────────────────────
    quad = QuadraticForm(
        variables=variables,
        matrix=sigma,
        scalar=0.5,
        matrix_sym=sp.Symbol("Sigma"),
    )
    risk_scalar = Constant(risk_aversion, name="q")
    risk_term: ASTNode = BinaryOp(BinaryOpKind.MUL, risk_scalar, quad)

    # ── return term: μᵀx ─────────────────────────────────────────────────
    ret_term: ASTNode = MatrixExpr(
        variables=variables,
        vector=mu,
        vector_sym=sp.Symbol("mu"),
    )

    # ── objective: risk_term − return_term ───────────────────────────────
    objective: ASTNode = BinaryOp(BinaryOpKind.SUB, risk_term, ret_term)

    # ── optional budget constraint penalty: penalty·(Σxᵢ − B)² ──────────
    if budget is not None:
        # Build Σxᵢ as sum of variables
        var_nodes = [
            sp.Symbol(name, binary=True) for name in variables.names
        ]
        # We represent the penalty term via a Constant holding its
        # symbolic value — full symbolic expansion happens in QUBO builder.
        # Here we tag the AST with a penalty node for downstream processing.
        pen_cst = Constant(penalty, name="penalty")
        bgt_cst = Constant(float(budget), name="B")
        # penalty·(Σxᵢ − B)² is encoded as a Constant at AST level;
        # the QUBO builder handles the quadratic expansion symbolically.
        penalty_node = _BudgetPenaltyNode(
            variables=variables,
            budget=budget,
            penalty=penalty,
        )
        objective = BinaryOp(BinaryOpKind.ADD, objective, penalty_node)

    return objective


# ── Budget penalty helper node (package-internal) ────────────────────────────


class _BudgetPenaltyNode(ASTNode):
    """
    Internal AST node encoding  penalty·(Σᵢ xᵢ − B)².

    This node is *not* exported — it is consumed by the QUBO builder.
    """

    def __init__(
        self,
        variables: VariableVector,
        budget: int,
        penalty: float,
    ) -> None:
        self._variables = variables
        self._budget = budget
        self._penalty = penalty

    @property
    def variables(self) -> VariableVector:
        return self._variables

    @property
    def budget(self) -> int:
        return self._budget

    @property
    def penalty(self) -> float:
        return self._penalty

    @property
    def sympy_expr(self) -> sp.Expr:
        x_syms = sp.Matrix(self._variables.symbols)
        total = sum(x_syms)
        return sp.Float(self._penalty) * (total - self._budget) ** 2

    def accept(self, visitor: "ASTVisitor") -> object:  # type: ignore[name-defined]
        # Visitors that don't handle this node fall back to visit_constant.
        return None

    def __repr__(self) -> str:
        return (
            f"BudgetPenalty(penalty={self._penalty}, "
            f"budget={self._budget}, vars={list(self._variables.names)})"
        )


# ── Public entry-point ────────────────────────────────────────────────────────


def parse_markowitz(
    mu: np.ndarray | list[float],
    sigma: np.ndarray | list[list[float]],
    risk_aversion: float = 1.0,
    asset_names: Optional[list[str]] = None,
    budget: Optional[int] = None,
    penalty: float = 10.0,
) -> FinancialObjective:
    """
    Parse a Markowitz Mean-Variance portfolio objective.

    Parameters
    ----------
    mu            : array-like, shape (n,)   – expected annual returns.
    sigma         : array-like, shape (n, n) – asset covariance matrix.
    risk_aversion : float ≥ 0               – how much variance to penalise.
    asset_names   : list[str] | None        – optional human-readable labels.
    budget        : int | None              – cardinality constraint (# assets).
    penalty       : float                   – Lagrange multiplier for budget.

    Returns
    -------
    FinancialObjective
        Fully parsed container with AST and variable vector.

    Raises
    ------
    InvalidCovarianceMatrixError
        If Σ is not symmetric positive semi-definite.
    InvalidReturnVectorError
        If μ has wrong dimensions.
    """
    mu_arr = np.asarray(mu, dtype=float)
    sigma_arr = np.asarray(sigma, dtype=float)
    n = sigma_arr.shape[0]

    _validate_covariance(sigma_arr, tol=1e-8)
    _validate_returns(mu_arr, n)

    if risk_aversion < 0:
        raise InvalidReturnVectorError(
            f"risk_aversion must be ≥ 0, got {risk_aversion}."
        )

    names: tuple[str, ...]
    if asset_names is not None:
        if len(asset_names) != n:
            raise InvalidReturnVectorError(
                f"asset_names has {len(asset_names)} items but n={n}."
            )
        names = tuple(asset_names)
    else:
        names = tuple(f"x{i}" for i in range(n))

    variables = VariableVector(names=names)
    ast_root = _build_markowitz_ast(
        variables=variables,
        mu=mu_arr,
        sigma=sigma_arr,
        risk_aversion=risk_aversion,
        budget=budget,
        penalty=penalty,
    )

    return FinancialObjective(
        n_assets=n,
        mu=mu_arr,
        sigma=sigma_arr,
        risk_aversion=risk_aversion,
        budget=budget,
        penalty=penalty,
        asset_names=names,
        ast=ast_root,
        variable_vec=variables,
    )
