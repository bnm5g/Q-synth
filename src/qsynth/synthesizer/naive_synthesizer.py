"""
qsynth.synthesizer.naive_synthesizer
======================================

The reference (naive) synthesis engine.

Constructs a QAOA-style Ansatz from a :class:`PauliHamiltonian` without
any optimization.  This gives a *baseline* circuit whose depth is later
reduced by the Compiler.

QAOA Ansatz construction
------------------------
For a p-layer QAOA:

1. Initial state:  H⊗ⁿ |0⟩ⁿ  (uniform superposition).

2. For each layer k = 1…p:

   a. Phase-separator U_C(γₖ):
      - For each Z term (hᵢ):     Rz(2hᵢγₖ) on qubit i.
      - For each ZZ term (Jᵢⱼ):  CNOT(i,j) · Rz(2Jᵢⱼγₖ) · CNOT(i,j).

   b. Mixer U_B(βₖ):
      - Rx(2βₖ) on every qubit.

The CNOT sandwich implements  e^{-i Jᵢⱼγ ZᵢZⱼ}:

    CNOT(i→j) · Rz(2Jᵢⱼγ, j) · CNOT(i→j)
    ≡ e^{-i Jᵢⱼγ ZᵢZⱼ}   (up to global phase on diagonal)

Reference:
    Farhi et al. (2014). "A Quantum Approximate Optimization Algorithm."
    arXiv:1411.4028.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from qsynth.exceptions import UnsynthesizableIRError
from qsynth.ir.pauli_hamiltonian import PauliHamiltonian, PauliTerm
from qsynth.synthesizer.logical_circuit import LogicalCircuit


class NaiveSynthesizer:
    """
    Reference QAOA synthesizer.  Produces a *correct but unoptimized* circuit.

    Parameters
    ----------
    n_layers : int
        Number of QAOA alternating layers p (default 1).
    gamma    : float | list[float]
        Phase-separation angle(s).  Scalar → broadcast across all layers.
    beta     : float | list[float]
        Mixing angle(s).  Scalar → broadcast across all layers.
    """

    def __init__(
        self,
        n_layers: int = 1,
        gamma: float | list[float] = math.pi / 4,
        beta: float | list[float] = math.pi / 4,
    ) -> None:
        if n_layers < 1:
            raise ValueError(f"n_layers must be ≥ 1, got {n_layers}.")
        self.n_layers = n_layers

        if isinstance(gamma, (int, float)):
            self._gammas: list[float] = [float(gamma)] * n_layers
        else:
            if len(gamma) != n_layers:
                raise ValueError(
                    f"len(gamma)={len(gamma)} ≠ n_layers={n_layers}."
                )
            self._gammas = [float(g) for g in gamma]

        if isinstance(beta, (int, float)):
            self._betas: list[float] = [float(beta)] * n_layers
        else:
            if len(beta) != n_layers:
                raise ValueError(
                    f"len(beta)={len(beta)} ≠ n_layers={n_layers}."
                )
            self._betas = [float(b) for b in beta]

    # ── public API ─────────────────────────────────────────────────────────

    def synthesize(self, hamiltonian: PauliHamiltonian) -> LogicalCircuit:
        """
        Synthesize a QAOA :class:`LogicalCircuit` from *hamiltonian*.

        Parameters
        ----------
        hamiltonian : PauliHamiltonian
            The Ising Hamiltonian H = Σhᵢ Zᵢ + Σ Jᵢⱼ ZᵢZⱼ.

        Returns
        -------
        LogicalCircuit
            An unoptimized logical circuit implementing p layers of QAOA.

        Raises
        ------
        UnsynthesizableIRError
            If the Hamiltonian contains non-Z Pauli terms (Y or X couplings)
            which are outside the standard QAOA synthesis scope.
        """
        n = hamiltonian.n_qubits
        if n == 0:
            raise UnsynthesizableIRError("Empty Hamiltonian — nothing to synthesize.")

        self._validate_hamiltonian(hamiltonian)

        # Separate Z and ZZ terms.
        z_terms, zz_terms = self._partition_terms(hamiltonian)

        circuit = LogicalCircuit(n_qubits=n, name="qaoa_naive")

        # Step 1: Hadamard layer.
        for q in range(n):
            circuit.h(q)

        # Step 2: Alternating QAOA layers.
        for layer_idx in range(self.n_layers):
            gamma = self._gammas[layer_idx]
            beta = self._betas[layer_idx]
            self._append_phase_separator(circuit, z_terms, zz_terms, gamma, layer_idx)
            self._append_mixer(circuit, n, beta, layer_idx)

        return circuit

    # ── private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _validate_hamiltonian(ham: PauliHamiltonian) -> None:
        """
        Ensure the Hamiltonian contains only I, Z, and ZZ terms.

        Standard QAOA assumes a diagonal cost Hamiltonian (Z-basis).
        """
        for term in ham.terms:
            unique_ops = set(term.pauli_str) - {"I"}
            if not unique_ops.issubset({"Z"}):
                raise UnsynthesizableIRError(
                    f"Term '{term.pauli_str}' contains non-Z Pauli operators "
                    f"({unique_ops - {'Z'}}). Naive synthesizer requires an "
                    f"Ising (ZZ/Z/I) Hamiltonian."
                )
            z_count = term.pauli_str.count("Z")
            if z_count > 2:
                raise UnsynthesizableIRError(
                    f"Term '{term.pauli_str}' is a {z_count}-body interaction; "
                    f"QAOA synthesis only supports 1- and 2-body Z terms."
                )

    @staticmethod
    def _partition_terms(
        ham: PauliHamiltonian,
    ) -> tuple[list[tuple[int, float]], list[tuple[int, int, float]]]:
        """
        Split Hamiltonian terms into single-Z and ZZ terms.

        Returns
        -------
        z_terms  : list of (qubit_index, coefficient)
        zz_terms : list of (qubit_i, qubit_j, coefficient)
        """
        n = ham.n_qubits
        z_terms: list[tuple[int, float]] = []
        zz_terms: list[tuple[int, int, float]] = []

        for term in ham.terms:
            z_positions = [
                n - 1 - pos
                for pos, c in enumerate(term.pauli_str)
                if c == "Z"
            ]
            if len(z_positions) == 1:
                z_terms.append((z_positions[0], term.coefficient))
            elif len(z_positions) == 2:
                i, j = sorted(z_positions)
                zz_terms.append((i, j, term.coefficient))
            # I terms skipped (they are the constant offset).

        return z_terms, zz_terms

    @staticmethod
    def _append_phase_separator(
        circuit: LogicalCircuit,
        z_terms: list[tuple[int, float]],
        zz_terms: list[tuple[int, int, float]],
        gamma: float,
        layer_idx: int,
    ) -> None:
        """
        Append U_C(γ) = ∏ e^{-iγhᵢZᵢ} · ∏ e^{-iγJᵢⱼZᵢZⱼ}.

        Single-Z:  Rz(2γhᵢ) on qubit i.
        ZZ:        CNOT(i→j) · Rz(2γJᵢⱼ, j) · CNOT(i→j).
        """
        tag = f"UC[{layer_idx}]"

        for qubit, coeff in z_terms:
            angle = 2.0 * gamma * coeff
            circuit.rz(angle, qubit)

        for qi, qj, coeff in zz_terms:
            angle = 2.0 * gamma * coeff
            circuit.cnot(qi, qj)
            circuit.rz(angle, qj)
            circuit.cnot(qi, qj)

    @staticmethod
    def _append_mixer(
        circuit: LogicalCircuit,
        n_qubits: int,
        beta: float,
        layer_idx: int,
    ) -> None:
        """
        Append U_B(β) = ∏ᵢ e^{-iβXᵢ} = ∏ᵢ Rx(2β).
        """
        for q in range(n_qubits):
            circuit.rx(2.0 * beta, q)
