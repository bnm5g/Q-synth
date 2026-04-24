"""
qsynth.parser.ast_nodes
=======================

Typed AST node definitions for financial objective expressions.

The AST is a simple, immutable tree whose leaves are :class:`Variable`,
:class:`VariableVector`, and :class:`Constant` nodes, and whose interior
nodes are arithmetic composites (:class:`BinaryOp`, :class:`UnaryOp`,
:class:`QuadraticForm`, :class:`MatrixExpr`).

All nodes carry a :attr:`sympy_expr` property that returns the equivalent
SymPy expression — enabling symbolic manipulation and differentiation
downstream.

Design: nodes are frozen dataclasses for value-equality semantics and
immutability enforcement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Sequence

import sympy as sp
import numpy as np


# ── Operator enumerations ────────────────────────────────────────────────────


class BinaryOpKind(Enum):
    """Binary arithmetic operators recognised by the parser."""
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    POW = auto()


class UnaryOpKind(Enum):
    """Unary operators recognised by the parser."""
    NEG = auto()
    TRANSPOSE = auto()


# ── Abstract base ────────────────────────────────────────────────────────────


class ASTNode(ABC):
    """
    Abstract base for all AST nodes.

    All concrete nodes must implement:
    - :meth:`sympy_expr`   – SymPy representation (for symbolic algebra).
    - :meth:`accept`       – Visitor entry-point.
    - :meth:`__repr__`     – Human-readable debug string.
    """

    @property
    @abstractmethod
    def sympy_expr(self) -> sp.Expr:
        """Return the SymPy expression equivalent of this node."""

    @abstractmethod
    def accept(self, visitor: "ASTVisitor") -> Any:
        """Accept a :class:`ASTVisitor` (Visitor pattern)."""

    @abstractmethod
    def __repr__(self) -> str: ...


# ── Visitor interface ────────────────────────────────────────────────────────


class ASTVisitor(ABC):
    """
    Visitor interface for traversing AST nodes.

    Subclass this and implement each ``visit_*`` method to define a
    custom tree walk (e.g. pretty-printing, QUBO extraction, etc.).
    """

    @abstractmethod
    def visit_constant(self, node: "Constant") -> Any: ...

    @abstractmethod
    def visit_variable(self, node: "Variable") -> Any: ...

    @abstractmethod
    def visit_variable_vector(self, node: "VariableVector") -> Any: ...

    @abstractmethod
    def visit_binary_op(self, node: "BinaryOp") -> Any: ...

    @abstractmethod
    def visit_unary_op(self, node: "UnaryOp") -> Any: ...

    @abstractmethod
    def visit_quadratic_form(self, node: "QuadraticForm") -> Any: ...

    @abstractmethod
    def visit_matrix_expr(self, node: "MatrixExpr") -> Any: ...


# ── Leaf nodes ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Constant(ASTNode):
    """
    A scalar numeric constant.

    Attributes
    ----------
    value : float
        The numeric value.
    name  : str
        Optional label (e.g. 'q' for risk aversion).
    """

    value: float
    name: str = ""

    @property
    def sympy_expr(self) -> sp.Expr:
        return sp.Float(self.value)

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_constant(self)

    def __repr__(self) -> str:
        tag = f"[{self.name}]" if self.name else ""
        return f"Constant({self.value}{tag})"


@dataclass(frozen=True)
class Variable(ASTNode):
    """
    A single binary decision variable  xᵢ ∈ {0, 1}.

    Attributes
    ----------
    name  : str   – symbolic name (e.g. "x0").
    index : int   – integer index for ordering.
    """

    name: str
    index: int = 0

    @property
    def sympy_expr(self) -> sp.Expr:
        return sp.Symbol(self.name, binary=True)

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_variable(self)

    def __repr__(self) -> str:
        return f"Var({self.name})"


@dataclass(frozen=True)
class VariableVector(ASTNode):
    """
    A vector of N binary decision variables  x = [x₀, x₁, …, xₙ₋₁].

    Attributes
    ----------
    names : tuple[str, ...]   – ordered variable names.
    """

    names: tuple[str, ...]

    # ── convenience constructors ──────────────────────────────────────────

    @classmethod
    def from_count(cls, n: int, prefix: str = "x") -> "VariableVector":
        """Create a vector of *n* variables named ``{prefix}0 … {prefix}{n-1}``."""
        return cls(names=tuple(f"{prefix}{i}" for i in range(n)))

    # ── properties ───────────────────────────────────────────────────────

    @property
    def n_vars(self) -> int:
        return len(self.names)

    @property
    def symbols(self) -> list[sp.Symbol]:
        return [sp.Symbol(n, binary=True) for n in self.names]

    @property
    def sympy_expr(self) -> sp.Expr:
        # Return a SymPy column vector (MatrixSymbol won't do for symbolic ops).
        return sp.Matrix(self.symbols)

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_variable_vector(self)

    def __repr__(self) -> str:
        return f"VarVector({list(self.names)})"


# ── Interior nodes ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BinaryOp(ASTNode):
    """
    A binary arithmetic operation on two child AST nodes.

    Attributes
    ----------
    op    : BinaryOpKind
    left  : ASTNode
    right : ASTNode
    """

    op: BinaryOpKind
    left: ASTNode
    right: ASTNode

    @property
    def sympy_expr(self) -> sp.Expr:
        l, r = self.left.sympy_expr, self.right.sympy_expr
        match self.op:
            case BinaryOpKind.ADD: return l + r
            case BinaryOpKind.SUB: return l - r
            case BinaryOpKind.MUL: return l * r
            case BinaryOpKind.DIV: return l / r
            case BinaryOpKind.POW: return l ** r

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_binary_op(self)

    def __repr__(self) -> str:
        return f"({self.left!r} {self.op.name} {self.right!r})"


@dataclass(frozen=True)
class UnaryOp(ASTNode):
    """
    A unary operation on one child AST node.

    Attributes
    ----------
    op      : UnaryOpKind
    operand : ASTNode
    """

    op: UnaryOpKind
    operand: ASTNode

    @property
    def sympy_expr(self) -> sp.Expr:
        expr = self.operand.sympy_expr
        match self.op:
            case UnaryOpKind.NEG:
                return -expr
            case UnaryOpKind.TRANSPOSE:
                # Transpose makes sense only on matrices; return as-is for scalars.
                if isinstance(expr, sp.MatrixBase):
                    return expr.T
                return expr

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_unary_op(self)

    def __repr__(self) -> str:
        return f"({self.op.name} {self.operand!r})"


@dataclass(frozen=True)
class QuadraticForm(ASTNode):
    """
    The quadratic form  ½ xᵀ Σ x  appearing in the Markowitz objective.

    Attributes
    ----------
    variables  : VariableVector   – the binary variable vector.
    matrix     : np.ndarray       – the real symmetric matrix Σ (N×N).
    scalar     : float            – pre-multiplier (default 0.5 = ½).
    matrix_sym : sp.Symbol        – SymPy symbol label for the matrix.
    """

    variables: VariableVector
    matrix: np.ndarray = field(compare=False)
    scalar: float = 0.5
    matrix_sym: sp.Symbol = field(default_factory=lambda: sp.Symbol("Sigma"))

    @property
    def n(self) -> int:
        return self.variables.n_vars

    @property
    def sympy_expr(self) -> sp.Expr:
        x = sp.Matrix(self.variables.symbols)
        M = sp.Matrix(self.matrix.tolist())
        return sp.Rational(1, 2) * (x.T * M * x)[0, 0]

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_quadratic_form(self)

    def __repr__(self) -> str:
        return (
            f"QuadraticForm(½·xᵀ·{self.matrix_sym}·x, "
            f"vars={list(self.variables.names)}, scale={self.scalar})"
        )


@dataclass(frozen=True)
class MatrixExpr(ASTNode):
    """
    A linear (dot-product) term  μᵀ x  appearing in the Markowitz objective.

    Attributes
    ----------
    vector      : np.ndarray       – coefficient vector μ (N,).
    variables   : VariableVector   – the binary variable vector.
    vector_sym  : sp.Symbol        – SymPy label for the vector.
    """

    variables: VariableVector
    vector: np.ndarray = field(compare=False)
    vector_sym: sp.Symbol = field(default_factory=lambda: sp.Symbol("mu"))

    @property
    def sympy_expr(self) -> sp.Expr:
        mu = sp.Matrix(self.vector.tolist())
        x  = sp.Matrix(self.variables.symbols)
        return (mu.T * x)[0, 0]

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_matrix_expr(self)

    def __repr__(self) -> str:
        return f"MatrixExpr({self.vector_sym}ᵀ·x, vars={list(self.variables.names)})"
