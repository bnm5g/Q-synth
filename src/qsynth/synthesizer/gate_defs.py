"""
qsynth.synthesizer.gate_defs
=============================

Object-oriented gate representations backed by exact unitary matrices.

Each gate is a frozen dataclass that:
1. Declares its qubit arity via :attr:`n_qubits`.
2. Provides its exact 2ⁿ × 2ⁿ unitary matrix via :meth:`matrix`.
3. Validates unitarity upon construction (‖U†U − I‖_F < tol).
4. Supports comparison and hashing via value equality.

Hierarchy
---------
``Gate`` (abstract)
├── ``IdentityGate``   – I  (single qubit)
├── ``HGate``          – Hadamard
├── ``RxGate``         – Rₓ(θ) = exp(−i θ/2 X)
├── ``RyGate``         – Rᵧ(θ) = exp(−i θ/2 Y)
├── ``RzGate``         – R_z(θ) = exp(−i θ/2 Z)
├── ``CnotGate``       – CNOT (controlled-X), 2-qubit
└── ``CrzGate``        – Controlled-Rz(θ),   2-qubit

The :class:`GateType` enum provides string-free gate identification.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any

import numpy as np

from qsynth.exceptions import NonUnitarySynthesisError, DimensionMismatchError


# ── Tolerance for unitarity check ─────────────────────────────────────────────
_UNITARY_TOL: float = 1e-8


def _assert_unitary(matrix: np.ndarray, name: str, tol: float = _UNITARY_TOL) -> None:
    """
    Verify that *matrix* is unitary: ‖U†U − I‖_F < tol.

    Raises
    ------
    NonUnitarySynthesisError
    DimensionMismatchError
    """
    n, m = matrix.shape
    if n != m:
        raise DimensionMismatchError(n, m, context=name)
    err = float(np.linalg.norm(matrix.conj().T @ matrix - np.eye(n, dtype=complex)))
    if err > tol:
        raise NonUnitarySynthesisError(matrix_name=name, frobenius_error=err)


# ── Gate type enum ─────────────────────────────────────────────────────────────


class GateType(Enum):
    """Enumeration of all gate types in the Q-Synth gate set."""
    IDENTITY = auto()
    H        = auto()
    RX       = auto()
    RY       = auto()
    RZ       = auto()
    CNOT     = auto()
    CRZ      = auto()
    CUSTOM   = auto()


# ── Abstract base ──────────────────────────────────────────────────────────────


class Gate(ABC):
    """
    Abstract base for all quantum gates.

    Subclasses must implement:
    - :attr:`gate_type`  – GateType enum value.
    - :attr:`n_qubits`   – qubit arity.
    - :meth:`matrix`     – exact unitary matrix (2ⁿ × 2ⁿ, complex128).
    - :meth:`params`     – tuple of real parameters (empty for fixed gates).
    - :meth:`label`      – short human-readable string.
    - :meth:`inverse`    – return the adjoint gate.
    """

    @property
    @abstractmethod
    def gate_type(self) -> GateType: ...

    @property
    @abstractmethod
    def n_qubits(self) -> int: ...

    @abstractmethod
    def matrix(self) -> np.ndarray: ...

    @abstractmethod
    def params(self) -> tuple[float, ...]: ...

    @abstractmethod
    def label(self) -> str: ...

    @abstractmethod
    def inverse(self) -> "Gate": ...

    # ── shared helpers ─────────────────────────────────────────────────────

    def is_unitary(self, tol: float = _UNITARY_TOL) -> bool:
        """Check U†U ≈ I without raising."""
        try:
            _assert_unitary(self.matrix(), self.label(), tol)
            return True
        except NonUnitarySynthesisError:
            return False

    def commutes_with(self, other: "Gate") -> bool:
        """
        Check if this gate commutes with *other* via matrix multiplication.

        AB = BA  iff  AB − BA = 0  (up to _UNITARY_TOL).
        Only meaningful when both gates act on the same qubit space.
        """
        if self.n_qubits != other.n_qubits:
            return False
        AB = self.matrix() @ other.matrix()
        BA = other.matrix() @ self.matrix()
        return bool(np.allclose(AB, BA, atol=_UNITARY_TOL))

    def is_identity(self, tol: float = _UNITARY_TOL) -> bool:
        """Return True if this gate's matrix is the identity (up to tol)."""
        dim = 2 ** self.n_qubits
        return bool(np.allclose(self.matrix(), np.eye(dim, dtype=complex), atol=tol))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Gate):
            return NotImplemented
        return (
            self.gate_type == other.gate_type
            and self.n_qubits == other.n_qubits
            and np.allclose(self.matrix(), other.matrix(), atol=_UNITARY_TOL)
        )

    def __hash__(self) -> int:
        return hash((self.gate_type, self.params()))

    def __repr__(self) -> str:
        return self.label()


# ── Single-qubit gates ─────────────────────────────────────────────────────────


class IdentityGate(Gate):
    """Single-qubit identity  I = [[1,0],[0,1]]."""

    @property
    def gate_type(self) -> GateType:
        return GateType.IDENTITY

    @property
    def n_qubits(self) -> int:
        return 1

    def matrix(self) -> np.ndarray:
        return np.eye(2, dtype=complex)

    def params(self) -> tuple[float, ...]:
        return ()

    def label(self) -> str:
        return "I"

    def inverse(self) -> "IdentityGate":
        return IdentityGate()


class HGate(Gate):
    """Hadamard gate  H = (X + Z)/√2."""

    _M: np.ndarray = np.array(
        [[1, 1], [1, -1]], dtype=complex
    ) / math.sqrt(2)

    @property
    def gate_type(self) -> GateType:
        return GateType.H

    @property
    def n_qubits(self) -> int:
        return 1

    def matrix(self) -> np.ndarray:
        return self._M.copy()

    def params(self) -> tuple[float, ...]:
        return ()

    def label(self) -> str:
        return "H"

    def inverse(self) -> "HGate":
        return HGate()  # H is self-inverse


class RxGate(Gate):
    """
    Rₓ(θ) = exp(−iθX/2) = cos(θ/2)·I − i·sin(θ/2)·X
    """

    __slots__ = ("_theta",)

    def __init__(self, theta: float) -> None:
        self._theta = float(theta)

    @property
    def gate_type(self) -> GateType:
        return GateType.RX

    @property
    def n_qubits(self) -> int:
        return 1

    def matrix(self) -> np.ndarray:
        c = math.cos(self._theta / 2)
        s = math.sin(self._theta / 2)
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)

    def params(self) -> tuple[float, ...]:
        return (self._theta,)

    def label(self) -> str:
        return f"Rx({self._theta:.4f})"

    def inverse(self) -> "RxGate":
        return RxGate(-self._theta)


class RyGate(Gate):
    """
    Rᵧ(θ) = exp(−iθY/2) = cos(θ/2)·I − i·sin(θ/2)·Y
    """

    __slots__ = ("_theta",)

    def __init__(self, theta: float) -> None:
        self._theta = float(theta)

    @property
    def gate_type(self) -> GateType:
        return GateType.RY

    @property
    def n_qubits(self) -> int:
        return 1

    def matrix(self) -> np.ndarray:
        c = math.cos(self._theta / 2)
        s = math.sin(self._theta / 2)
        return np.array([[c, -s], [s, c]], dtype=complex)

    def params(self) -> tuple[float, ...]:
        return (self._theta,)

    def label(self) -> str:
        return f"Ry({self._theta:.4f})"

    def inverse(self) -> "RyGate":
        return RyGate(-self._theta)


class RzGate(Gate):
    """
    R_z(θ) = exp(−iθZ/2) = diag(e^{−iθ/2}, e^{+iθ/2})
    """

    __slots__ = ("_theta",)

    def __init__(self, theta: float) -> None:
        self._theta = float(theta)

    @property
    def gate_type(self) -> GateType:
        return GateType.RZ

    @property
    def n_qubits(self) -> int:
        return 1

    def matrix(self) -> np.ndarray:
        phase = self._theta / 2
        return np.array(
            [[np.exp(-1j * phase), 0], [0, np.exp(1j * phase)]],
            dtype=complex,
        )

    def params(self) -> tuple[float, ...]:
        return (self._theta,)

    def label(self) -> str:
        return f"Rz({self._theta:.4f})"

    def inverse(self) -> "RzGate":
        return RzGate(-self._theta)


# ── Two-qubit gates ────────────────────────────────────────────────────────────


class CnotGate(Gate):
    """
    CNOT (controlled-X), 2 qubits.

    Qubit ordering: (control, target).
    Matrix in computational basis {|00⟩, |01⟩, |10⟩, |11⟩}:
        [[1,0,0,0],
         [0,1,0,0],
         [0,0,0,1],
         [0,0,1,0]]
    """

    _M: np.ndarray = np.array(
        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 1],
         [0, 0, 1, 0]],
        dtype=complex,
    )

    @property
    def gate_type(self) -> GateType:
        return GateType.CNOT

    @property
    def n_qubits(self) -> int:
        return 2

    def matrix(self) -> np.ndarray:
        return self._M.copy()

    def params(self) -> tuple[float, ...]:
        return ()

    def label(self) -> str:
        return "CNOT"

    def inverse(self) -> "CnotGate":
        return CnotGate()  # CNOT² = I


class CrzGate(Gate):
    """
    Controlled-Rz(θ) gate, 2 qubits (control, target).

    This is the standard ZZ-rotation building block used in QAOA:
        e^{-i θ/2 ZZ} = CNOT · Rz(θ) · CNOT

    Matrix:
        [[1, 0, 0,           0          ],
         [0, 1, 0,           0          ],
         [0, 0, e^{-iθ/2},  0          ],
         [0, 0, 0,           e^{+iθ/2} ]]
    """

    __slots__ = ("_theta",)

    def __init__(self, theta: float) -> None:
        self._theta = float(theta)

    @property
    def gate_type(self) -> GateType:
        return GateType.CRZ

    @property
    def n_qubits(self) -> int:
        return 2

    def matrix(self) -> np.ndarray:
        phase = self._theta / 2
        return np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, np.exp(-1j * phase), 0],
             [0, 0, 0, np.exp(1j * phase)]],
            dtype=complex,
        )

    def params(self) -> tuple[float, ...]:
        return (self._theta,)

    def label(self) -> str:
        return f"CRz({self._theta:.4f})"

    def inverse(self) -> "CrzGate":
        return CrzGate(-self._theta)
