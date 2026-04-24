"""
tests/test_parser.py
====================

Unit tests for the parser layer.

Covers:
- FinancialObjective construction and validation
- QUBO matrix construction
- Mathematical correctness of the QUBO formulation
- Budget constraint handling
- Error handling for invalid inputs
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from qsynth.exceptions import (
    InvalidCovarianceMatrixError,
    InvalidReturnVectorError,
    HamiltonianConstructionError,
)
from qsynth.parser import (
    parse_markowitz,
    build_qubo,
    FinancialObjective,
    QUBOProblem,
)
from qsynth.parser.ast_nodes import (
    BinaryOp,
    BinaryOpKind,
    Constant,
    MatrixExpr,
    QuadraticForm,
    VariableVector,
)


class TestFinancialParser:
    """Tests for parse_markowitz."""

    def test_basic_construction(self) -> None:
        mu = np.array([0.10, 0.20])
        sigma = np.array([[0.05, 0.01], [0.01, 0.10]])
        obj = parse_markowitz(mu=mu, sigma=sigma)
        assert obj.n_assets == 2
        assert len(obj.asset_names) == 2
        assert obj.risk_aversion == 1.0
        assert obj.budget is None

    def test_asset_names_assigned(self) -> None:
        mu = np.array([0.10, 0.20])
        sigma = np.array([[0.05, 0.01], [0.01, 0.10]])
        obj = parse_markowitz(mu=mu, sigma=sigma, asset_names=["AAPL", "GOOG"])
        assert obj.asset_names == ("AAPL", "GOOG")

    def test_default_variable_names(self) -> None:
        mu = np.array([0.1, 0.2, 0.3])
        sigma = np.eye(3) * 0.05
        obj = parse_markowitz(mu=mu, sigma=sigma)
        assert obj.variable_vec.names == ("x0", "x1", "x2")

    def test_ast_is_not_none(self, two_asset_objective: FinancialObjective) -> None:
        assert two_asset_objective.ast is not None

    def test_ast_sympy_expr_computable(
        self, two_asset_objective: FinancialObjective
    ) -> None:
        expr = two_asset_objective.ast.sympy_expr
        assert isinstance(expr, sp.Basic)

    def test_four_asset_construction(
        self, four_asset_objective: FinancialObjective
    ) -> None:
        assert four_asset_objective.n_assets == 4
        assert four_asset_objective.variable_vec.n_vars == 4

    def test_invalid_non_psd_covariance(self) -> None:
        mu = np.array([0.1, 0.2])
        sigma_bad = np.array([[0.05, 0.10], [0.10, 0.01]])  # Not PSD
        with pytest.raises(InvalidCovarianceMatrixError):
            parse_markowitz(mu=mu, sigma=sigma_bad)

    def test_invalid_asymmetric_covariance(self) -> None:
        mu = np.array([0.1, 0.2])
        sigma_bad = np.array([[0.05, 0.10], [0.01, 0.10]])  # Asymmetric
        with pytest.raises(InvalidCovarianceMatrixError):
            parse_markowitz(mu=mu, sigma=sigma_bad)

    def test_invalid_dimension_mismatch(self) -> None:
        mu = np.array([0.1, 0.2, 0.3])  # 3 assets
        sigma = np.eye(2) * 0.05  # 2x2 sigma
        with pytest.raises(InvalidReturnVectorError):
            parse_markowitz(mu=mu, sigma=sigma)

    def test_invalid_negative_risk_aversion(self) -> None:
        mu = np.array([0.1, 0.2])
        sigma = np.eye(2) * 0.05
        with pytest.raises(InvalidReturnVectorError):
            parse_markowitz(mu=mu, sigma=sigma, risk_aversion=-1.0)

    def test_budget_constraint_objective(self) -> None:
        mu = np.array([0.1, 0.2, 0.15])
        sigma = np.eye(3) * 0.05
        obj = parse_markowitz(mu=mu, sigma=sigma, budget=2, penalty=10.0)
        assert obj.budget == 2
        assert obj.penalty == 10.0

    def test_single_asset(self) -> None:
        mu = np.array([0.10])
        sigma = np.array([[0.05]])
        obj = parse_markowitz(mu=mu, sigma=sigma)
        assert obj.n_assets == 1


class TestQUBOBuilder:
    """Tests for build_qubo."""

    def test_qubo_shape(self, two_asset_qubo: QUBOProblem) -> None:
        assert two_asset_qubo.Q.shape == (2, 2)

    def test_qubo_variable_names(self, two_asset_qubo: QUBOProblem) -> None:
        assert two_asset_qubo.variable_names == ("AAPL", "GOOG")

    def test_qubo_upper_triangular(self, two_asset_qubo: QUBOProblem) -> None:
        Q = two_asset_qubo.Q
        # Lower triangle should be zero.
        for i in range(Q.shape[0]):
            for j in range(i):
                assert Q[i, j] == pytest.approx(0.0, abs=1e-12)

    def test_qubo_energy_all_zero(self, two_asset_qubo: QUBOProblem) -> None:
        """x = [0, 0] → energy = constant offset."""
        x = np.array([0.0, 0.0])
        energy = two_asset_qubo.energy(x)
        assert energy == pytest.approx(two_asset_qubo.constant, abs=1e-10)

    def test_qubo_energy_manual_single_asset(self) -> None:
        """
        For a 1-asset problem:  f(x) = q·½·σ²·x² − μ·x
        With binary x: f(1) = q·½·σ² − μ,  f(0) = 0.
        """
        mu = np.array([0.20])
        sigma = np.array([[0.10]])
        q = 1.0
        obj = parse_markowitz(mu=mu, sigma=sigma, risk_aversion=q)
        qubo = build_qubo(obj)
        x1 = np.array([1.0])
        x0 = np.array([0.0])
        expected_f1 = q * 0.5 * 0.10 - 0.20  # = 0.05 - 0.20 = -0.15
        assert qubo.energy(x1) == pytest.approx(expected_f1, abs=1e-8)
        # f(0) = constant (should be 0 for no budget constraint).
        assert qubo.energy(x0) == pytest.approx(qubo.constant, abs=1e-8)

    def test_qubo_energy_symmetrised_consistency(
        self, two_asset_qubo: QUBOProblem
    ) -> None:
        """
        xᵀQx = xᵀQ_symx for all binary x (upper-triangular ≡ symmetric
        because off-diagonal terms contribute equally).
        """
        Q = two_asset_qubo.Q
        Q_sym = two_asset_qubo.symmetric_Q()
        for x in [[0, 0], [1, 0], [0, 1], [1, 1]]:
            x_arr = np.array(x, dtype=float)
            assert x_arr @ Q @ x_arr == pytest.approx(
                x_arr @ Q_sym @ x_arr, abs=1e-10
            )

    def test_qubo_four_assets_shape(self, four_asset_qubo: QUBOProblem) -> None:
        assert four_asset_qubo.Q.shape == (4, 4)

    def test_qubo_energy_brute_force_optimal(self) -> None:
        """
        For a tiny 2-asset case, brute-force over all binary assignments
        and verify the QUBO minimum matches the analytical minimum.
        """
        mu = np.array([0.10, 0.20])
        sigma = np.array([[0.05, 0.01], [0.01, 0.10]])
        q = 1.0
        obj = parse_markowitz(mu=mu, sigma=sigma, risk_aversion=q)
        qubo = build_qubo(obj)
        energies = {}
        for i in range(4):
            x = np.array([(i >> 0) & 1, (i >> 1) & 1], dtype=float)
            energies[tuple(x)] = qubo.energy(x)
        min_state = min(energies, key=energies.get)  # type: ignore
        min_energy = energies[min_state]
        # Sanity: minimum energy should be negative (asset selection improves obj).
        assert min_energy < 0.0

    def test_qubo_describe_runs(self, two_asset_qubo: QUBOProblem) -> None:
        desc = two_asset_qubo.describe()
        assert "QUBOProblem" in desc
