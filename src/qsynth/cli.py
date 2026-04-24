"""
qsynth.cli
==========

Command-line interface for the Q-Synth pipeline.

Usage
-----
    python -m qsynth.cli --n-assets 4 --layers 1 --topology linear

or after `pip install -e .`:
    qsynth --n-assets 4 --layers 2 --topology all-to-all
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from qsynth.parser import parse_markowitz, build_qubo
from qsynth.ir import build_hamiltonian
from qsynth.synthesizer import NaiveSynthesizer
from qsynth.compiler import PassManager, HardwareTopology, TopologyMapper
from qsynth.evaluator import StatevectorEvaluator, CircuitVerifier, extract_solution


# ── Demo datasets ──────────────────────────────────────────────────────────────

_DEMO_ASSETS_2 = {
    "names": ["AAPL", "GOOG"],
    "mu": [0.10, 0.20],
    "sigma": [[0.05, 0.01], [0.01, 0.10]],
}

_DEMO_ASSETS_4 = {
    "names": ["AAPL", "GOOG", "MSFT", "AMZN"],
    "mu": [0.12, 0.18, 0.09, 0.22],
    "sigma": [
        [0.06, 0.02, 0.01, 0.03],
        [0.02, 0.09, 0.02, 0.01],
        [0.01, 0.02, 0.04, 0.01],
        [0.03, 0.01, 0.01, 0.12],
    ],
}

_DEMO_ASSETS_6 = {
    "names": ["AAPL", "GOOG", "MSFT", "AMZN", "META", "NVDA"],
    "mu": [0.12, 0.18, 0.09, 0.22, 0.15, 0.30],
    "sigma": [
        [0.06, 0.02, 0.01, 0.03, 0.02, 0.04],
        [0.02, 0.09, 0.02, 0.01, 0.03, 0.02],
        [0.01, 0.02, 0.04, 0.01, 0.01, 0.02],
        [0.03, 0.01, 0.01, 0.12, 0.02, 0.05],
        [0.02, 0.03, 0.01, 0.02, 0.08, 0.03],
        [0.04, 0.02, 0.02, 0.05, 0.03, 0.15],
    ],
}

_DATASETS = {2: _DEMO_ASSETS_2, 4: _DEMO_ASSETS_4, 6: _DEMO_ASSETS_6}


def _get_topology(name: str, n: int) -> HardwareTopology:
    match name:
        case "all-to-all":
            return HardwareTopology.all_to_all(n)
        case "linear":
            return HardwareTopology.linear(n)
        case "heavy-hex":
            return HardwareTopology.heavy_hex(n)
        case _:
            return HardwareTopology.all_to_all(n)


def run_pipeline(
    n_assets: int = 4,
    n_layers: int = 1,
    risk_aversion: float = 1.0,
    topology: str = "all-to-all",
    verify: bool = False,
    verbose: bool = True,
) -> None:
    """
    Run the complete Q-Synth pipeline for demo assets.

    Parameters
    ----------
    n_assets      : 2, 4, or 6
    n_layers      : QAOA depth (number of alternating layers)
    risk_aversion : q parameter
    topology      : "all-to-all" | "linear" | "heavy-hex"
    verify        : Run Z3 gate identity verification
    verbose       : Print detailed output
    """
    if n_assets not in _DATASETS:
        print(f"[ERROR] n-assets must be one of {list(_DATASETS.keys())}.")
        sys.exit(1)

    data = _DATASETS[n_assets]

    print("=" * 60)
    print("  Q-Synth: Quantum Program Synthesis for Portfolio Optimization")
    print("=" * 60)
    print(f"\n[1/7] Parsing {n_assets}-asset Markowitz objective...")

    objective = parse_markowitz(
        mu=data["mu"],
        sigma=data["sigma"],
        risk_aversion=risk_aversion,
        asset_names=data["names"],
    )
    if verbose:
        print(objective.describe())

    print("\n[2/7] Building QUBO...")
    qubo = build_qubo(objective)
    if verbose:
        print(qubo.describe())

    print("\n[3/7] Mapping QUBO → Ising Pauli Hamiltonian...")
    hamiltonian = build_hamiltonian(qubo)
    if verbose:
        print(hamiltonian.describe())

    print(f"\n[4/7] Synthesizing QAOA circuit (p={n_layers} layers)...")
    synth = NaiveSynthesizer(n_layers=n_layers)
    naive_circuit = synth.synthesize(hamiltonian)
    print(f"  Naive circuit:  {naive_circuit.gate_count()} gates, depth={naive_circuit.depth()}")

    print("\n[5/7] Compiling & optimizing (algebraic depth reduction)...")
    pm = PassManager.default()
    optimized_circuit = pm.run(naive_circuit)
    reduction_pct = 100.0 * (1 - optimized_circuit.depth() / max(naive_circuit.depth(), 1))
    print(f"  Optimized circuit: {optimized_circuit.gate_count()} gates, depth={optimized_circuit.depth()}")
    print(f"  Depth reduction: {reduction_pct:.1f}%")

    # Topology mapping
    topo = _get_topology(topology, n_assets)
    mapper = TopologyMapper(topo)
    physical_circuit = mapper.map(optimized_circuit)
    print(f"\n  Topology '{topo.name}': depth={physical_circuit.depth()}")

    # Compare topologies
    topo_all = HardwareTopology.all_to_all(n_assets)
    physical_all = TopologyMapper(topo_all).map(optimized_circuit)
    print(f"  All-to-all reference: depth={physical_all.depth()}")

    print("\n[6/7] Evaluating circuit (statevector simulation)...")
    evaluator = StatevectorEvaluator()
    eval_result = evaluator.evaluate(optimized_circuit, hamiltonian)
    print(f"  ⟨H⟩ expectation value: {eval_result.expectation_value:.6f}")
    print(f"  Most likely state: |{eval_result.most_likely_state()}⟩")

    print("\n[7/7] Extracting portfolio solution...")
    solution = extract_solution(eval_result, objective)
    print(solution.describe())

    if verify:
        print("\n[Bonus] Z3 gate identity verification...")
        verifier = CircuitVerifier()
        z3_result = verifier.verify_gate_identities_z3()
        print(f"  {z3_result}")
        herm_result = verifier.verify_hamiltonian_hermitian(hamiltonian)
        print(f"  {herm_result}")

    print("\n" + "=" * 60)
    print("  Q-Synth pipeline complete.")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Q-Synth: Quantum Program Synthesis for Financial Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  qsynth --n-assets 4 --layers 1
  qsynth --n-assets 6 --layers 2 --topology linear
  qsynth --n-assets 4 --verify
        """,
    )
    parser.add_argument(
        "--n-assets",
        type=int,
        default=4,
        choices=[2, 4, 6],
        help="Number of assets (2, 4, or 6) [default: 4]",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=1,
        help="Number of QAOA layers p [default: 1]",
    )
    parser.add_argument(
        "--risk-aversion",
        type=float,
        default=1.0,
        help="Risk aversion parameter q [default: 1.0]",
    )
    parser.add_argument(
        "--topology",
        type=str,
        default="all-to-all",
        choices=["all-to-all", "linear", "heavy-hex"],
        help="Target hardware topology [default: all-to-all]",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run Z3 gate identity verification",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose intermediate output",
    )

    args = parser.parse_args()
    run_pipeline(
        n_assets=args.n_assets,
        n_layers=args.layers,
        risk_aversion=args.risk_aversion,
        topology=args.topology,
        verify=args.verify,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
