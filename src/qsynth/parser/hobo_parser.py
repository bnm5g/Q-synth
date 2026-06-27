"""
qsynth.parser.hobo_parser
===========================

Parses a Higher-Order Binary Optimization (HOBO) objective from a SymPy
expression into a typed AST.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import sympy as sp

from qsynth.parser.ast_nodes import (
    ASTNode,
    BinaryOp,
    BinaryOpKind,
    Constant,
    PolynomialTerm,
    Variable,
    VariableVector,
)


@dataclass(frozen=True)
class PolynomialObjective:
    """
    Container for a parsed HOBO polynomial objective.

    Attributes
    ----------
    ast          : ASTNode        – root of the symbolic AST.
    variable_vec : VariableVector – ordered binary variable vector.
    """

    ast: ASTNode
    variable_vec: VariableVector

    def describe(self) -> str:
        """Return a human-readable summary of the objective."""
        return (
            "PolynomialObjective\n"
            f"  Variables : {list(self.variable_vec.names)}\n"
            f"  AST Node  : {type(self.ast).__name__}"
        )


def parse_sympy_expression(expr: sp.Expr | str) -> PolynomialObjective:
    """
    Parse a SymPy expression or string into a PolynomialObjective.

    This function expands the expression and extracts polynomial terms,
    automatically reducing variable exponents to 1 (binary identity x^k = x).

    Parameters
    ----------
    expr : sp.Expr | str
        The expression to parse.

    Returns
    -------
    PolynomialObjective
        The parsed objective containing the AST.
    """
    if isinstance(expr, str):
        expr = sp.sympify(expr)

    # Convert to standard binary symbols if they aren't already
    free_symbols = sorted(list(expr.free_symbols), key=lambda s: s.name)
    var_names = tuple(s.name for s in free_symbols)
    variable_vec = VariableVector(names=var_names)
    var_map = {name: Variable(name, i) for i, name in enumerate(var_names)}

    if not free_symbols:
        ast_root: ASTNode = Constant(float(expr))
        return PolynomialObjective(ast=ast_root, variable_vec=variable_vec)

    # Use sp.Poly to safely extract monomials and coefficients
    poly = sp.Poly(sp.expand(expr), *free_symbols)

    ast_root = None
    for monom, coeff in zip(poly.monoms(), poly.coeffs()):
        coeff_val = float(coeff)
        term_vars = []
        for i, power in enumerate(monom):
            if power > 0:
                # Apply binary identity: any positive power reduces to degree 1
                term_vars.append(var_map[var_names[i]])

        poly_term = PolynomialTerm(
            variables=tuple(term_vars),
            coefficient=coeff_val
        )

        if ast_root is None:
            ast_root = poly_term
        else:
            ast_root = BinaryOp(BinaryOpKind.ADD, ast_root, poly_term)

    if ast_root is None:
        ast_root = Constant(0.0)

    return PolynomialObjective(ast=ast_root, variable_vec=variable_vec)
