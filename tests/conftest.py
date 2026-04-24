"""
tests/conftest.py
=================

Shared pytest fixtures for Q-Synth tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from qsynth.parser import parse_markowitz, build_qubo, FinancialObjective, QUBOProblem
from qsynth.ir import build_hamiltonian, PauliHamiltonian
from qsynth.synthesizer import NaiveSynthesizer, LogicalCircuit


# ── 2-asset fixture ────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def two_asset_objective() -> FinancialObjective:
    """Minimal 2-asset Markowitz objective."""
    mu = np.array([0.10, 0.20])
    sigma = np.array([[0.05, 0.01], [0.01, 0.10]])
    return parse_markowitz(
        mu=mu,
        sigma=sigma,
        risk_aversion=1.0,
        asset_names=["AAPL", "GOOG"],
    )


@pytest.fixture(scope="session")
def two_asset_qubo(two_asset_objective: FinancialObjective) -> QUBOProblem:
    return build_qubo(two_asset_objective)


@pytest.fixture(scope="session")
def two_asset_hamiltonian(two_asset_qubo: QUBOProblem) -> PauliHamiltonian:
    return build_hamiltonian(two_asset_qubo)


# ── 4-asset fixture ────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def four_asset_objective() -> FinancialObjective:
    """4-asset portfolio test case."""
    mu = np.array([0.12, 0.18, 0.09, 0.22])
    sigma = np.array([
        [0.06, 0.02, 0.01, 0.03],
        [0.02, 0.09, 0.02, 0.01],
        [0.01, 0.02, 0.04, 0.01],
        [0.03, 0.01, 0.01, 0.12],
    ])
    return parse_markowitz(
        mu=mu,
        sigma=sigma,
        risk_aversion=1.0,
        asset_names=["AAPL", "GOOG", "MSFT", "AMZN"],
    )


@pytest.fixture(scope="session")
def four_asset_qubo(four_asset_objective: FinancialObjective) -> QUBOProblem:
    return build_qubo(four_asset_objective)


@pytest.fixture(scope="session")
def four_asset_hamiltonian(four_asset_qubo: QUBOProblem) -> PauliHamiltonian:
    return build_hamiltonian(four_asset_qubo)


@pytest.fixture(scope="session")
def four_asset_naive_circuit(four_asset_hamiltonian: PauliHamiltonian) -> LogicalCircuit:
    synth = NaiveSynthesizer(n_layers=1)
    return synth.synthesize(four_asset_hamiltonian)
