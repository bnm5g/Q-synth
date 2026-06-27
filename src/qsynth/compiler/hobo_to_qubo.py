"""
qsynth.compiler.hobo_to_qubo
============================

Implements the Rosenberg Quadratization pass to reduce a Higher-Order 
Binary Optimization (HOBO) objective to a QUBO objective.
"""

from __future__ import annotations

from typing import Any, Literal

from qsynth.parser.ast_nodes import (
    ASTNode,
    ASTVisitor,
    BinaryOp,
    BinaryOpKind,
    Constant,
    MatrixExpr,
    PolynomialTerm,
    QuadraticForm,
    UnaryOp,
    Variable,
    VariableVector,
)
from qsynth.parser.hobo_parser import PolynomialObjective


class RosenbergReductionPass:
    """
    Compiler pass that reduces a HOBO objective to a QUBO objective.

    It iteratively finds terms of degree > 2, substitutes pairs of variables
    with a new auxiliary variable, and applies a Rosenberg penalty term to
    the overall objective to enforce the auxiliary equivalence.
    """

    def __init__(
        self,
        penalty: float | Literal["auto"] = "auto",
        default_penalty: float = 10.0,
    ) -> None:
        self.penalty_strategy = penalty
        self.default_penalty = default_penalty
        self._aux_counter = 0
        self.aux_mapping: dict[str, tuple[str, str]] = {}

    def run(self, objective: PolynomialObjective) -> PolynomialObjective:
        """Execute the reduction pass on a HOBO objective."""
        analyzer = _DegreeAnalyzer()
        objective.ast.accept(analyzer)

        if analyzer.max_degree <= 2:
            return objective  # Already QUBO

        penalty_weight = self._calculate_penalty(analyzer.sum_abs_coeffs)

        reducer = _RosenbergVisitor(self, penalty_weight)
        new_ast = objective.ast.accept(reducer)

        # Append global penalty terms to the root objective
        for pen_node in reducer.penalties:
            new_ast = BinaryOp(BinaryOpKind.ADD, new_ast, pen_node)

        # Extend variable vector with new auxiliary variables
        original_names = list(objective.variable_vec.names)
        new_names = original_names + list(self.aux_mapping.keys())
        new_vec = VariableVector(names=tuple(new_names))

        return PolynomialObjective(ast=new_ast, variable_vec=new_vec)

    def _calculate_penalty(self, sum_abs_coeffs: float) -> float:
        if self.penalty_strategy == "auto":
            # Heuristic: penalty slightly larger than max theoretical energy shift
            return sum_abs_coeffs + 0.1
        return float(self.penalty_strategy)

    def generate_aux_variable(self, v1: Variable, v2: Variable) -> Variable:
        """Create a monotonic auxiliary variable representing v1 * v2."""
        aux_name = f"aux_{self._aux_counter}"
        self._aux_counter += 1
        self.aux_mapping[aux_name] = (v1.name, v2.name)
        # Index is not strictly needed for symbolic manipulation
        return Variable(name=aux_name, index=-1)


class _DegreeAnalyzer(ASTVisitor):
    """Calculates the max degree and coefficient sum for penalty heuristics."""

    def __init__(self) -> None:
        self.max_degree = 0
        self.sum_abs_coeffs = 0.0

    def visit_constant(self, node: Constant) -> None:
        pass

    def visit_variable(self, node: Variable) -> None:
        self.max_degree = max(self.max_degree, 1)

    def visit_variable_vector(self, node: VariableVector) -> None:
        pass

    def visit_binary_op(self, node: BinaryOp) -> None:
        node.left.accept(self)
        node.right.accept(self)

    def visit_unary_op(self, node: UnaryOp) -> None:
        node.operand.accept(self)

    def visit_quadratic_form(self, node: QuadraticForm) -> None:
        self.max_degree = max(self.max_degree, 2)

    def visit_matrix_expr(self, node: MatrixExpr) -> None:
        self.max_degree = max(self.max_degree, 1)

    def visit_polynomial_term(self, node: PolynomialTerm) -> None:
        deg = len(node.variables)
        self.max_degree = max(self.max_degree, deg)
        if deg > 2:
            self.sum_abs_coeffs += abs(node.coefficient)


class _RosenbergVisitor(ASTVisitor):
    """Replaces high-degree terms locally and accumulates global penalties."""

    def __init__(
        self, pass_manager: RosenbergReductionPass, penalty_weight: float
    ) -> None:
        self.pass_manager = pass_manager
        self.P = penalty_weight
        self.penalties: list[ASTNode] = []

    def visit_constant(self, node: Constant) -> ASTNode:
        return node

    def visit_variable(self, node: Variable) -> ASTNode:
        return node

    def visit_variable_vector(self, node: VariableVector) -> ASTNode:
        return node

    def visit_binary_op(self, node: BinaryOp) -> ASTNode:
        left = node.left.accept(self)
        right = node.right.accept(self)
        return BinaryOp(node.op, left, right)

    def visit_unary_op(self, node: UnaryOp) -> ASTNode:
        operand = node.operand.accept(self)
        return UnaryOp(node.op, operand)

    def visit_quadratic_form(self, node: QuadraticForm) -> ASTNode:
        return node

    def visit_matrix_expr(self, node: MatrixExpr) -> ASTNode:
        return node

    def visit_polynomial_term(self, node: PolynomialTerm) -> ASTNode:
        if len(node.variables) <= 2:
            return node

        variables = list(node.variables)

        # Locally reduce x1*x2*x3... until degree is 2
        while len(variables) > 2:
            v1 = variables.pop(0)
            v2 = variables.pop(0)

            aux_var = self.pass_manager.generate_aux_variable(v1, v2)
            variables.insert(0, aux_var)

            # Rosenberg penalty: P * (x1*x2 - 2*x1*w - 2*x2*w + 3*w)
            pen_x1_x2 = PolynomialTerm((v1, v2), self.P)
            pen_x1_w = PolynomialTerm((v1, aux_var), -2.0 * self.P)
            pen_x2_w = PolynomialTerm((v2, aux_var), -2.0 * self.P)
            pen_w = PolynomialTerm((aux_var,), 3.0 * self.P)

            p_node_1 = BinaryOp(BinaryOpKind.ADD, pen_x1_x2, pen_x1_w)
            p_node_2 = BinaryOp(BinaryOpKind.ADD, pen_x2_w, pen_w)
            p_full = BinaryOp(BinaryOpKind.ADD, p_node_1, p_node_2)

            self.penalties.append(p_full)

        return PolynomialTerm(tuple(variables), node.coefficient)
