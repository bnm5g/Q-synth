"""
qsynth.ir.ising_mapper
=======================

Performs the canonical QUBO → Ising model transformation:

    xᵢ = (1 − σᵢᶻ) / 2   ⟹   σᵢᶻ = 1 − 2xᵢ

Substituting into  f(x) = xᵀQx + const  yields:

    H_Ising = Σᵢ hᵢ Zᵢ  +  Σᵢ<ⱼ Jᵢⱼ ZᵢZⱼ  +  offset·I

where:

    hᵢ  = -½ Σⱼ Q_sym[i,j]   (linear field on qubit i)
    Jᵢⱼ = ¼ Q_sym[i,j]       (ZZ coupling for i≠j)
    offset = ¼ Σᵢⱼ Q_sym[i,j] + const

Reference: Lucas, A. (2014). "Ising formulations of many NP problems."
Frontiers in Physics, 2, 5. DOI:10.3389/fphy.2014.00005
"""

from __future__ import annotations

import numpy as np

from qsynth.exceptions import HamiltonianConstructionError
from qsynth.ir.pauli_hamiltonian import PauliHamiltonian, PauliTerm
from qsynth.parser.qubo_builder import QUBOProblem


def qubo_to_ising(
    qubo: QUBOProblem,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Map a QUBO to the Ising model coefficients (h, J, offset).

    Parameters
    ----------
    qubo : QUBOProblem
        The QUBO problem with upper-triangular matrix Q.
    tol  : float
        Coefficients below this threshold are zeroed (numerical clean-up).

    Returns
    -------
    h      : np.ndarray, shape (n,)
        Linear field coefficients (Z terms).
    J      : np.ndarray, shape (n, n), upper-triangular
        Quadratic coupling coefficients (ZZ terms).
    offset : float
        Constant energy offset.

    Mathematical derivation
    -----------------------
    The QUBO is:  f(x) = Σᵢ Q[i,i]·xᵢ  +  Σᵢ<j Q[i,j]·xᵢxⱼ  +  const
    (using binary identity xᵢ²=xᵢ).

    Substituting  xᵢ = (1 − σᵢᶻ)/2:

        Q[i,i]·xᵢ       = Q[i,i]·(1−σᵢᶻ)/2
                         → linear:   −Q[i,i]/2 on σᵢᶻ
                         → const:    +Q[i,i]/2

        Q[i,j]·xᵢxⱼ     = Q[i,j]·(1−σᵢᶻ)(1−σⱼᶻ)/4
                         = Q[i,j]/4 · (1 − σᵢᶻ − σⱼᶻ + σᵢᶻσⱼᶻ)
                         → ZᵢZⱼ coupling:  +Q[i,j]/4
                         → linear on i:    −Q[i,j]/4
                         → linear on j:    −Q[i,j]/4
                         → const:          +Q[i,j]/4

    Collecting:
        hᵢ  = −Q[i,i]/2  −  Σⱼ>ᵢ Q[i,j]/4  −  Σⱼ<ᵢ Q[j,i]/4
            = −Q[i,i]/2  −  (1/4)·(row sum of upper triangle excluding diag)
                          −  (1/4)·(col sum of upper triangle excluding diag)

    Since Q is upper-triangular (Q[j,i]=0 for j>i):
        hᵢ  = −Q[i,i]/2  −  Σⱼ>ᵢ Q[i,j]/4  −  Σⱼ<ᵢ Q[j,i]/4
            = −Q[i,i]/2  −  (1/4)·(Σⱼ≠ᵢ Q_sym[i,j]·2)   ... where Q_sym = (Q+Qᵀ)/2
        But more cleanly:
        hᵢ  = −Q[i,i]/2  −  Σⱼ≠ᵢ Q_full[i,j]/4
        where Q_full[i,j] = Q[i,j] for i<j, Q[j,i] for i>j.

        Jᵢⱼ = Q[i,j]/4                         for i < j
        offset = Σᵢ Q[i,i]/2  +  Σᵢ<j Q[i,j]/4  +  const
    """
    n = qubo.n_vars
    Q = qubo.Q  # upper-triangular

    h = np.zeros(n, dtype=float)
    J = np.zeros((n, n), dtype=float)

    offset = float(qubo.constant)

    for i in range(n):
        # Diagonal contribution to h and offset.
        h[i] -= Q[i, i] / 2.0
        offset += Q[i, i] / 2.0

    for i in range(n):
        for j in range(i + 1, n):
            if abs(Q[i, j]) < tol:
                continue
            J[i, j] = Q[i, j] / 4.0
            h[i] -= Q[i, j] / 4.0
            h[j] -= Q[i, j] / 4.0
            offset += Q[i, j] / 4.0

    # Numerical clean-up
    h[np.abs(h) < tol] = 0.0
    J[np.abs(J) < tol] = 0.0

    return h, J, offset


def build_hamiltonian(
    qubo: QUBOProblem,
    tol: float = 1e-12,
) -> PauliHamiltonian:
    """
    Build a :class:`PauliHamiltonian` from a :class:`QUBOProblem`.

    Parameters
    ----------
    qubo : QUBOProblem
    tol  : float   – coefficient threshold for numerical clean-up.

    Returns
    -------
    PauliHamiltonian
        Ising Hamiltonian  H = Σ hᵢZᵢ + Σ JᵢⱼZᵢZⱼ + offset·I

    Raises
    ------
    HamiltonianConstructionError
        If n_vars == 0.
    """
    n = qubo.n_vars
    if n == 0:
        raise HamiltonianConstructionError("QUBO has zero variables.")

    h, J, offset = qubo_to_ising(qubo, tol=tol)

    ham = PauliHamiltonian(n_qubits=n, constant=offset)

    # Linear (Z) terms: hᵢ·Zᵢ
    for i in range(n):
        if abs(h[i]) > tol:
            # Build Pauli string: 'I'*(n-1-i) + 'Z' + 'I'*i
            # Qiskit convention: index 0 = rightmost character.
            pauli_str = "I" * (n - 1 - i) + "Z" + "I" * i
            ham.add_term(pauli_str, float(h[i]))

    # Quadratic (ZZ) terms: Jᵢⱼ·ZᵢZⱼ
    for i in range(n):
        for j in range(i + 1, n):
            if abs(J[i, j]) > tol:
                # Pauli string with Z at positions i and j.
                chars = ["I"] * n
                chars[n - 1 - i] = "Z"
                chars[n - 1 - j] = "Z"
                pauli_str = "".join(chars)
                ham.add_term(pauli_str, float(J[i, j]))

    return ham.simplify()
