"""
qsynth.parser.qubo_builder
===========================

Converts a :class:`FinancialObjective` (or any typed AST rooted at
:class:`ASTNode`) into a :class:`QUBOProblem` — the standard matrix
representation consumed by the IR layer.

QUBO formulation
----------------
A QUBO problem is expressed as:

    min_x  xᵀ Q x

where  Q ∈ ℝⁿˣⁿ  is an upper-triangular (or symmetric) matrix and
x ∈ {0,1}ⁿ.

Derivation from Markowitz objective
------------------------------------
Given  f(x) = q·½·xᵀΣx − μᵀx :

1. Quadratic diagonal/off-diagonal terms:
       Qᵢⱼ = q·Σᵢⱼ / 2   for i ≠ j   (merged via x²=x for binary vars)
       Qᵢᵢ = q·Σᵢᵢ/2 − μᵢ

2. Using the binary identity xᵢ² = xᵢ for xᵢ ∈ {0,1}:
       xᵀΣx = Σᵢ Σᵢᵢxᵢ² + 2Σᵢ<ⱼ Σᵢⱼxᵢxⱼ
             = Σᵢ Σᵢᵢxᵢ + 2Σᵢ<ⱼ Σᵢⱼxᵢxⱼ

3. The linear return term −μᵀx contributes −μᵢ to the diagonal.

4. Optional budget constraint  penalty·(Σxᵢ − B)² expands to:
       penalty·[Σᵢ xᵢ² + 2Σᵢ<ⱼ xᵢxⱼ − 2B·Σᵢ xᵢ + B²]
       = penalty·[Σᵢ(1−2B)xᵢ + 2Σᵢ<ⱼ xᵢxⱼ + B²]

The result is stored as a dense  Q  matrix (upper-triangular by convention)
plus a constant offset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import sympy as sp

from qsynth.exceptions import HamiltonianConstructionError
from qsynth.parser.ast_nodes import (
    ASTNode,
    ASTVisitor,
    BinaryOp,
    BinaryOpKind,
    Constant,
    MatrixExpr,
    QuadraticForm,
    UnaryOp,
    VariableVector,
)
from qsynth.parser.financial_parser import FinancialObjective


# ── QUBOProblem data container ────────────────────────────────────────────────


@dataclass
class QUBOProblem:
    """
    Quadratic Unconstrained Binary Optimization problem.

    The problem is:  min_x  xᵀ Q x + linear_offset

    Attributes
    ----------
    Q              : np.ndarray (n×n, upper-triangular)
                     QUBO matrix where diagonal encodes linear terms.
    variable_names : tuple[str, ...]
                     Ordered names of binary variables.
    constant       : float
                     Constant offset (does not affect optimal x).
    n_vars         : int   (derived)
    """

    Q: np.ndarray = field(repr=False)
    variable_names: tuple[str, ...]
    constant: float = 0.0

    def __post_init__(self) -> None:
        n = len(self.variable_names)
        if self.Q.shape != (n, n):
            raise HamiltonianConstructionError(
                f"Q shape {self.Q.shape} inconsistent with "
                f"{n} variables."
            )

    @property
    def n_vars(self) -> int:
        return len(self.variable_names)

    def energy(self, x: np.ndarray) -> float:
        """Evaluate xᵀQx + constant for a given binary assignment x."""
        x = np.asarray(x, dtype=float)
        return float(x @ self.Q @ x) + self.constant

    def symmetric_Q(self) -> np.ndarray:
        """Return the symmetrised form  (Q + Qᵀ)/2."""
        return (self.Q + self.Q.T) / 2.0

    def describe(self) -> str:
        lines = [
            "QUBOProblem",
            f"  Variables : {list(self.variable_names)}",
            f"  Constant  : {self.constant:.6f}",
            "  Q matrix  :",
        ]
        for row in self.Q:
            lines.append("    " + "  ".join(f"{v:8.4f}" for v in row))
        return "\n".join(lines)


# ── Visitor-based QUBO extractor ──────────────────────────────────────────────


class _QUBOExtractor(ASTVisitor):
    """
    Visitor that traverses the Markowitz AST and accumulates the QUBO
    matrix  Q  and constant offset.

    Internally works with a *symmetric* accumulation matrix and converts
    to upper-triangular at the end.
    """

    def __init__(self, n: int, variable_names: tuple[str, ...]) -> None:
        self._n = n
        self._names = variable_names
        self._idx: dict[str, int] = {name: i for i, name in enumerate(variable_names)}
        # Accumulator — symmetric, full matrix (converted at end).
        self._Q: np.ndarray = np.zeros((n, n), dtype=float)
        self._constant: float = 0.0
        # Running sign multiplier (for subtraction propagation).
        self._sign: float = 1.0

    # ── helpers ──────────────────────────────────────────────────────────

    def _add_linear(self, i: int, coeff: float) -> None:
        self._Q[i, i] += coeff

    def _add_quadratic(self, i: int, j: int, coeff: float) -> None:
        if i == j:
            self._Q[i, i] += coeff
        else:
            # Symmetric accumulation; halved to avoid double-counting.
            self._Q[i, j] += coeff / 2.0
            self._Q[j, i] += coeff / 2.0

    # ── visitor implementations ───────────────────────────────────────────

    def visit_constant(self, node: Constant) -> float:
        return node.value

    def visit_variable(self, node) -> None:  # type: ignore[override]
        # Single variables don't appear bare in the Markowitz AST.
        return None

    def visit_variable_vector(self, node: VariableVector) -> None:  # type: ignore[override]
        return None

    def visit_binary_op(self, node: BinaryOp) -> float | None:
        """
        Handle ADD/SUB at the top level by toggling sign propagation.
        MUL of (Constant × QuadraticForm) or (Constant × MatrixExpr)
        is handled by detecting the pattern.
        """
        if node.op == BinaryOpKind.ADD:
            node.left.accept(self)
            node.right.accept(self)
        elif node.op == BinaryOpKind.SUB:
            node.left.accept(self)
            saved = self._sign
            self._sign *= -1.0
            node.right.accept(self)
            self._sign = saved
        elif node.op == BinaryOpKind.MUL:
            # Pattern: Constant × (QuadraticForm | MatrixExpr)
            if isinstance(node.left, Constant):
                scalar = node.left.value * self._sign
                self._apply_scaled(node.right, scalar)
            elif isinstance(node.right, Constant):
                scalar = node.right.value * self._sign
                self._apply_scaled(node.left, scalar)
            else:
                # Fall back to symbolic evaluation
                self._apply_symbolic(node)
        elif node.op == BinaryOpKind.DIV:
            if isinstance(node.right, Constant) and node.right.value != 0:
                scalar = (1.0 / node.right.value) * self._sign
                self._apply_scaled(node.left, scalar)
        return None

    def _apply_scaled(self, node: ASTNode, scalar: float) -> None:
        """Apply a scaled visit by temporarily adjusting the sign."""
        if isinstance(node, QuadraticForm):
            self._absorb_quadratic(node, scalar)
        elif isinstance(node, MatrixExpr):
            self._absorb_linear(node, scalar)
        else:
            saved = self._sign
            self._sign = scalar
            node.accept(self)
            self._sign = saved

    def _absorb_quadratic(self, node: QuadraticForm, scalar: float) -> None:
        """
        Absorb  scalar·½·xᵀΣx  into the accumulator.

        Using binary identity x² = x:
            ½·xᵀΣx = ½·Σᵢ Σᵢᵢ·xᵢ²  +  Σᵢ<ⱼ Σᵢⱼ·xᵢxⱼ
                    = ½·Σᵢ Σᵢᵢ·xᵢ   +  Σᵢ<ⱼ Σᵢⱼ·xᵢxⱼ    (x²=x)

        Combined with node.scalar (already ½):
            scalar·node.scalar = q·½

        So:
            diagonal contribution: q·½·Σᵢᵢ  per xᵢ
            off-diag coupling:    q·½·2·Σᵢⱼ / 2 = q·½·Σᵢⱼ (symmetrised)
        """
        sigma = node.matrix
        s = scalar * node.scalar  # e.g. q·½
        n = node.n
        names = node.variables.names
        for i in range(n):
            vi = self._idx[names[i]]
            # Diagonal: ½·Σᵢᵢ·xᵢ²  →  ½·Σᵢᵢ·xᵢ  (x²=x, no extra factor).
            self._add_linear(vi, s * sigma[i, i])
            for j in range(i + 1, n):
                vj = self._idx[names[j]]
                # Off-diagonal: ½·xᵀΣx includes both (i,j) and (j,i) terms.
                # ½·[Σᵢⱼ xᵢ xⱼ + Σⱼᵢ xⱼ xᵢ] = Σᵢⱼ·xᵢxⱼ  (since Σ symmetric).
                self._add_quadratic(vi, vj, s * 2 * sigma[i, j])

    def _absorb_linear(self, node: MatrixExpr, scalar: float) -> None:
        """Absorb  scalar·μᵀx  into the diagonal of the accumulator."""
        mu = node.vector
        names = node.variables.names
        for i, name in enumerate(names):
            vi = self._idx[name]
            self._add_linear(vi, scalar * mu[i])

    def _apply_symbolic(self, node: ASTNode) -> None:
        """
        Fallback: expand via SymPy and extract polynomial coefficients.
        This handles arbitrary expressions the pattern-matcher doesn't cover.
        """
        expr = node.sympy_expr
        symbols = [sp.Symbol(n, binary=True) for n in self._names]
        poly = sp.Poly(sp.expand(expr), *symbols)
        for monom, coeff in zip(poly.monoms(), poly.coeffs()):
            c = float(coeff) * self._sign
            active = [i for i, d in enumerate(monom) if d > 0]
            if len(active) == 0:
                self._constant += c
            elif len(active) == 1:
                self._add_linear(active[0], c)
            elif len(active) == 2:
                self._add_quadratic(active[0], active[1], c)
            else:
                raise HamiltonianConstructionError(
                    f"Degree-{len(active)} term found; QUBO requires degree ≤ 2."
                )

    def visit_unary_op(self, node: UnaryOp) -> None:  # type: ignore[override]
        if node.op.name == "NEG":
            saved = self._sign
            self._sign *= -1.0
            node.operand.accept(self)
            self._sign = saved
        else:
            node.operand.accept(self)

    def visit_quadratic_form(self, node: QuadraticForm) -> None:  # type: ignore[override]
        self._absorb_quadratic(node, self._sign)

    def visit_matrix_expr(self, node: MatrixExpr) -> None:  # type: ignore[override]
        self._absorb_linear(node, self._sign)

    # ── result extraction ─────────────────────────────────────────────────

    def build(self) -> tuple[np.ndarray, float]:
        """
        Return (Q_upper_triangular, constant_offset).

        Converts the symmetric accumulator to upper-triangular form:
            Q_upper[i,j] = Q_sym[i,j] + Q_sym[j,i]   for i < j
            Q_upper[i,i] = Q_sym[i,i]
        """
        Q_ut = np.zeros((self._n, self._n), dtype=float)
        for i in range(self._n):
            Q_ut[i, i] = self._Q[i, i]
            for j in range(i + 1, self._n):
                # Merge symmetric off-diagonal into upper triangle.
                Q_ut[i, j] = self._Q[i, j] + self._Q[j, i]
        return Q_ut, self._constant


# ── Budget penalty handler ────────────────────────────────────────────────────


def _apply_budget_penalty(
    Q: np.ndarray,
    constant: float,
    variable_names: tuple[str, ...],
    budget: int,
    penalty: float,
) -> tuple[np.ndarray, float]:
    """
    Add  penalty·(Σxᵢ − B)²  to an existing QUBO matrix.

    Expanding: penalty·(Σxᵢ)² − 2B·penalty·Σxᵢ + B²·penalty
    With x²=x: penalty·Σᵢxᵢ + 2·penalty·Σᵢ<ⱼ xᵢxⱼ − 2B·penalty·Σᵢxᵢ + const

    Returns updated (Q, constant).
    """
    n = len(variable_names)
    Q_out = Q.copy()
    for i in range(n):
        # Linear: penalty·(1 − 2B) per variable.
        Q_out[i, i] += penalty * (1 - 2 * budget)
        for j in range(i + 1, n):
            # Quadratic coupling.
            Q_out[i, j] += 2 * penalty
    constant += penalty * budget ** 2
    return Q_out, constant


# ── Public entry-point ────────────────────────────────────────────────────────


def build_qubo(objective: FinancialObjective) -> QUBOProblem:
    """
    Build a :class:`QUBOProblem` from a :class:`FinancialObjective`.

    The mapping uses binary identity  xᵢ² = xᵢ  to linearise diagonal
    terms, yielding an exact QUBO formulation of the Markowitz objective.

    Parameters
    ----------
    objective : FinancialObjective
        Parsed portfolio objective.

    Returns
    -------
    QUBOProblem
        QUBO matrix Q and constant offset.

    Raises
    ------
    HamiltonianConstructionError
        If the AST contains higher-degree terms or structural issues.
    """
    n = objective.n_assets
    names = objective.variable_vec.names

    extractor = _QUBOExtractor(n, names)
    objective.ast.accept(extractor)
    Q, constant = extractor.build()

    # Budget constraint is handled separately for clarity.
    if objective.budget is not None:
        Q, constant = _apply_budget_penalty(
            Q, constant, names, objective.budget, objective.penalty
        )

    return QUBOProblem(Q=Q, variable_names=names, constant=constant)
