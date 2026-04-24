"""
qsynth.parser
=============

Parses high-level financial objective functions into a mathematical AST
and converts them into Quadratic Unconstrained Binary Optimization (QUBO)
representations ready for IR construction.

Public surface
--------------
- :class:`FinancialObjective`   – container for μ, Σ, q, constraints
- :class:`QUBOProblem`          – QUBO matrix Q and offset
- :func:`parse_markowitz`       – entry-point for mean-variance portfolios
- :mod:`ast_nodes`              – AST node definitions
- :mod:`visitor`                – Visitor base-class for AST traversal
- :mod:`qubo_builder`           – QUBO construction from symbolic AST
"""

from qsynth.parser.ast_nodes import (
    ASTNode,
    BinaryOp,
    Constant,
    MatrixExpr,
    QuadraticForm,
    UnaryOp,
    Variable,
    VariableVector,
)
from qsynth.parser.financial_parser import FinancialObjective, parse_markowitz
from qsynth.parser.qubo_builder import QUBOProblem, build_qubo

__all__ = [
    # AST nodes
    "ASTNode",
    "BinaryOp",
    "Constant",
    "MatrixExpr",
    "QuadraticForm",
    "UnaryOp",
    "Variable",
    "VariableVector",
    # Financial domain
    "FinancialObjective",
    "parse_markowitz",
    # QUBO
    "QUBOProblem",
    "build_qubo",
]
