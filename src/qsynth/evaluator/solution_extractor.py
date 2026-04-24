"""
qsynth.evaluator.solution_extractor
=====================================

Extracts the classical solution from the quantum circuit's probability
distribution and maps it back to the original financial domain.

SolutionResult
--------------
- asset_selection  : list[bool]   – which assets to include.
- expected_return  : float        – μᵀx for the selected portfolio.
- portfolio_risk   : float        – xᵀΣx.
- objective_value  : float        – the Markowitz objective value.
- probability      : float        – amplitude² of the selected state.
- state_string     : str          – bitstring representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from qsynth.evaluator.statevector_evaluator import EvaluationResult
from qsynth.parser.financial_parser import FinancialObjective


@dataclass(frozen=True)
class SolutionResult:
    """
    Decoded classical solution from quantum measurement outcomes.

    Attributes
    ----------
    asset_selection  : list[bool]   – True for each selected asset.
    asset_names      : list[str]    – Names of assets.
    state_string     : str          – Binary string of the optimal state.
    probability      : float        – P(measured state).
    expected_return  : float        – μᵀx.
    portfolio_risk   : float        – xᵀΣx.
    sharpe_proxy     : float        – return / sqrt(risk) (no rf rate).
    objective_value  : float        – Markowitz objective q·½xᵀΣx − μᵀx.
    top_candidates   : list[dict]   – Top-5 candidate portfolios.
    """

    asset_selection: list[bool]
    asset_names: list[str]
    state_string: str
    probability: float
    expected_return: float
    portfolio_risk: float
    sharpe_proxy: float
    objective_value: float
    top_candidates: list[dict]

    def describe(self) -> str:
        selected = [
            name for name, sel in zip(self.asset_names, self.asset_selection) if sel
        ]
        lines = [
            "═══ Q-Synth Solution ═══",
            f"  State          : |{self.state_string}⟩",
            f"  Probability    : {self.probability:.4f}  ({self.probability*100:.2f}%)",
            f"  Selected assets: {selected}",
            f"  E[Return]      : {self.expected_return:.4f}",
            f"  Portfolio risk : {self.portfolio_risk:.4f}",
            f"  Sharpe proxy   : {self.sharpe_proxy:.4f}",
            f"  Objective f(x) : {self.objective_value:.4f}",
            "",
            "  Top candidates:",
        ]
        for i, cand in enumerate(self.top_candidates[:5], 1):
            lines.append(
                f"    {i}. |{cand['state']}⟩  "
                f"P={cand['prob']:.3f}  f={cand['obj']:.4f}"
            )
        return "\n".join(lines)


def extract_solution(
    eval_result: EvaluationResult,
    objective: FinancialObjective,
    top_k: int = 5,
) -> SolutionResult:
    """
    Extract the classical portfolio solution from evaluation results.

    Parameters
    ----------
    eval_result : EvaluationResult
        Output of :class:`StatevectorEvaluator`.
    objective   : FinancialObjective
        The original Markowitz objective for computing portfolio metrics.
    top_k       : int
        Number of candidate portfolios to report.

    Returns
    -------
    SolutionResult
    """
    n = eval_result.n_qubits
    probs = eval_result.probabilities
    mu = objective.mu
    sigma = objective.sigma
    q = objective.risk_aversion
    names = list(objective.asset_names)

    def _decode_state(idx: int) -> np.ndarray:
        """Decode integer state index to binary vector (qubit 0 = LSB)."""
        return np.array([(idx >> i) & 1 for i in range(n)], dtype=float)

    def _portfolio_metrics(x: np.ndarray) -> dict[str, float]:
        ret = float(mu @ x)
        risk = float(x @ sigma @ x)
        obj = q * 0.5 * risk - ret
        sharpe = ret / np.sqrt(max(risk, 1e-12))
        return {"return": ret, "risk": risk, "obj": obj, "sharpe": sharpe}

    # ── Find top-k candidates ──────────────────────────────────────────────
    top_indices = np.argsort(probs)[::-1][:top_k]
    candidates = []
    for idx in top_indices:
        x_vec = _decode_state(int(idx))
        metrics = _portfolio_metrics(x_vec)
        state_str = format(int(idx), f"0{n}b")
        candidates.append(
            {
                "state": state_str,
                "x": x_vec.tolist(),
                "prob": float(probs[idx]),
                **metrics,
            }
        )

    # ── Best state (highest probability) ─────────────────────────────────
    best = candidates[0]
    x_best = np.array(best["x"])

    return SolutionResult(
        asset_selection=[bool(v) for v in x_best],
        asset_names=names,
        state_string=best["state"],
        probability=best["prob"],
        expected_return=best["return"],
        portfolio_risk=best["risk"],
        sharpe_proxy=best["sharpe"],
        objective_value=best["obj"],
        top_candidates=candidates,
    )
