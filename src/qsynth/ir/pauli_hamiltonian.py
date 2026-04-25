"""
qsynth.ir.pauli_hamiltonian
===========================

Defines :class:`PauliTerm` and :class:`PauliHamiltonian` — the typed
Intermediate Representation for Ising / Pauli Hamiltonians.

A Pauli term is a tensor product of single-qubit Pauli operators {I,X,Y,Z}
multiplied by a real coefficient:

    term = coeff · P₀ ⊗ P₁ ⊗ … ⊗ P_{n-1}

The term is encoded as a string of length n over the alphabet {I,X,Y,Z}
(Qiskit convention: rightmost character = qubit 0).

A :class:`PauliHamiltonian` is an ordered list of terms and provides:
- Conversion to :class:`qiskit.quantum_info.SparsePauliOp`.
- Dense matrix representation (for small n, used in verification).
- Term grouping utilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np
import scipy.sparse as sp_sparse
from qiskit.quantum_info import SparsePauliOp, Pauli

from qsynth.exceptions import HamiltonianConstructionError


# ── Single Pauli basis matrices ───────────────────────────────────────────────

_I2 = np.eye(2, dtype=complex)
_X2 = np.array([[0, 1], [1, 0]], dtype=complex)
_Y2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z2 = np.array([[1, 0], [0, -1]], dtype=complex)

PAULI_MATRICES: dict[str, np.ndarray] = {"I": _I2, "X": _X2, "Y": _Y2, "Z": _Z2}


def _pauli_kron(pauli_str: str) -> np.ndarray:
    """
    Compute the dense 2ⁿ × 2ⁿ matrix for a Pauli string.

    Qiskit convention: pauli_str[0] is the *rightmost* (qubit 0) operator.
    We reverse so that kron product goes qubit n-1 ⊗ … ⊗ qubit 0.
    """
    ops = [PAULI_MATRICES[c] for c in reversed(pauli_str)]
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(op, result)
    return result


# Sparse single-qubit Pauli matrices (pre-built for reuse)
_PAULI_SPARSE: dict[str, sp_sparse.csr_matrix] = {
    k: sp_sparse.csr_matrix(v) for k, v in PAULI_MATRICES.items()
}


def _pauli_kron_sparse(pauli_str: str) -> sp_sparse.csr_matrix:
    """
    Compute the sparse 2ⁿ × 2ⁿ CSR matrix for a Pauli string.

    Builds the Kronecker product iteratively using ``scipy.sparse.kron``
    so that no dense 2ⁿ × 2ⁿ matrix is ever allocated — efficient for
    large n (up to ~14 qubits).

    Same qubit convention as :func:`_pauli_kron` (Qiskit: rightmost = qubit 0).
    """
    ops = [_PAULI_SPARSE[c] for c in reversed(pauli_str)]
    result: sp_sparse.csr_matrix = ops[0]
    for op in ops[1:]:
        result = sp_sparse.kron(op, result, format="csr")
    return result


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class PauliTerm:
    """
    A single weighted Pauli tensor-product term.

    Attributes
    ----------
    pauli_str : str
        Pauli string of length n_qubits, e.g. "IZZI".
        Convention: index 0 = qubit 0 (rightmost in Qiskit notation).
    coefficient : float
        Real coefficient in the Hamiltonian expansion.
    n_qubits : int   (derived from pauli_str)
    """

    pauli_str: str
    coefficient: float

    def __post_init__(self) -> None:
        for c in self.pauli_str:
            if c not in ("I", "X", "Y", "Z"):
                raise HamiltonianConstructionError(
                    f"Invalid Pauli character '{c}' in string '{self.pauli_str}'."
                )

    @property
    def n_qubits(self) -> int:
        return len(self.pauli_str)

    @property
    def is_identity(self) -> bool:
        return all(c == "I" for c in self.pauli_str)

    def matrix(self) -> np.ndarray:
        """Return the dense matrix representation of this term (scaled)."""
        return self.coefficient * _pauli_kron(self.pauli_str)

    def sparse_matrix(self) -> sp_sparse.csr_matrix:
        """
        Return the sparse CSR representation.

        Built via iterated ``scipy.sparse.kron`` — no dense 2ⁿ × 2ⁿ matrix
        is allocated, making this efficient for large n.
        """
        return self.coefficient * _pauli_kron_sparse(self.pauli_str)

    def to_qiskit(self) -> SparsePauliOp:
        """Convert to a single-term Qiskit SparsePauliOp."""
        return SparsePauliOp.from_list([(self.pauli_str, self.coefficient)])

    def commutes_with(self, other: "PauliTerm") -> bool:
        """
        Return True if this term commutes with *other*.

        Two Pauli strings commute iff the number of positions where both
        are non-identity and differ is even.
        """
        if self.n_qubits != other.n_qubits:
            return False
        anti_commuting_count = sum(
            1
            for a, b in zip(self.pauli_str, other.pauli_str)
            if a != "I" and b != "I" and a != b
        )
        return anti_commuting_count % 2 == 0

    def __repr__(self) -> str:
        sign = "+" if self.coefficient >= 0 else ""
        return f"{sign}{self.coefficient:.4f}·{self.pauli_str}"


@dataclass
class PauliHamiltonian:
    """
    A Pauli Hamiltonian expressed as H = Σᵢ cᵢ Pᵢ.

    Attributes
    ----------
    terms     : list[PauliTerm]  – ordered list of Pauli terms.
    n_qubits  : int              – number of qubits (inferred from first term).
    constant  : float            – additive scalar offset (identity term).
    """

    terms: list[PauliTerm] = field(default_factory=list)
    n_qubits: int = 0
    constant: float = 0.0

    def __post_init__(self) -> None:
        if self.terms:
            expected_n = self.terms[0].n_qubits
            for t in self.terms:
                if t.n_qubits != expected_n:
                    raise HamiltonianConstructionError(
                        f"Inconsistent qubit counts in Hamiltonian: "
                        f"expected {expected_n}, got {t.n_qubits}."
                    )
            if self.n_qubits == 0:
                self.n_qubits = expected_n

    # ── construction helpers ──────────────────────────────────────────────

    def add_term(self, pauli_str: str, coefficient: float) -> None:
        """Append a term and validate its qubit count."""
        term = PauliTerm(pauli_str=pauli_str, coefficient=coefficient)
        if self.n_qubits == 0:
            self.n_qubits = term.n_qubits
        elif term.n_qubits != self.n_qubits:
            raise HamiltonianConstructionError(
                f"Cannot add {term.n_qubits}-qubit term to "
                f"{self.n_qubits}-qubit Hamiltonian."
            )
        self.terms.append(term)

    # ── iterators & properties ────────────────────────────────────────────

    def __iter__(self) -> Iterator[PauliTerm]:
        return iter(self.terms)

    def __len__(self) -> int:
        return len(self.terms)

    def n_terms(self) -> int:
        """Total number of Pauli terms (excluding constant)."""
        return len(self.terms)

    # ── algebraic operations ──────────────────────────────────────────────

    def simplify(self) -> "PauliHamiltonian":
        """
        Combine duplicate Pauli strings by summing their coefficients.
        Terms whose absolute coefficient is below 1e-12 are dropped.
        """
        acc: dict[str, float] = {}
        for t in self.terms:
            acc[t.pauli_str] = acc.get(t.pauli_str, 0.0) + t.coefficient
        merged_terms = [
            PauliTerm(ps, c)
            for ps, c in acc.items()
            if abs(c) > 1e-12
        ]
        return PauliHamiltonian(
            terms=merged_terms,
            n_qubits=self.n_qubits,
            constant=self.constant,
        )

    # ── matrix representations ────────────────────────────────────────────

    def dense_matrix(self) -> np.ndarray:
        """
        Assemble the full 2ⁿ × 2ⁿ dense Hamiltonian matrix.

        Warning: exponential memory — only use for n ≤ 12.
        """
        dim = 2 ** self.n_qubits
        H = np.zeros((dim, dim), dtype=complex)
        # Include constant as identity contribution.
        H += self.constant * np.eye(dim, dtype=complex)
        for term in self.terms:
            H += term.matrix()
        return H

    def sparse_matrix(self) -> sp_sparse.csr_matrix:
        """
        Assemble the Hamiltonian as a sparse CSR matrix.

        Efficient for n up to ~12–14 qubits.
        """
        dim = 2 ** self.n_qubits
        H = sp_sparse.csr_matrix((dim, dim), dtype=complex)
        H = H + self.constant * sp_sparse.eye(dim, format="csr", dtype=complex)
        for term in self.terms:
            H = H + term.sparse_matrix()
        return H

    def to_qiskit(self) -> SparsePauliOp:
        """
        Convert to Qiskit :class:`SparsePauliOp` for use in circuits.

        The constant offset is embedded as an identity term.
        """
        pauli_list: list[tuple[str, complex]] = []
        identity_str = "I" * self.n_qubits
        if abs(self.constant) > 1e-12:
            pauli_list.append((identity_str, complex(self.constant)))
        for term in self.terms:
            pauli_list.append((term.pauli_str, complex(term.coefficient)))
        if not pauli_list:
            pauli_list.append((identity_str, 0.0))
        return SparsePauliOp.from_list(pauli_list).simplify()

    def is_hermitian(self, tol: float = 1e-8) -> bool:
        """Check H = H† using the dense matrix representation."""
        H = self.dense_matrix()
        return bool(np.allclose(H, H.conj().T, atol=tol))

    # ── display ───────────────────────────────────────────────────────────

    def describe(self) -> str:
        lines = [f"PauliHamiltonian ({self.n_qubits} qubits, {len(self.terms)} terms)"]
        if abs(self.constant) > 1e-12:
            lines.append(f"  + {self.constant:.6f}·{'I'*self.n_qubits}")
        for t in self.terms:
            lines.append(f"  {t!r}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"PauliHamiltonian(n_qubits={self.n_qubits}, "
            f"n_terms={len(self.terms)}, constant={self.constant:.4f})"
        )
