"""
qsynth.compiler
===============

The algebraic depth-reduction compiler.

Takes a :class:`LogicalCircuit` and applies a sequence of optimization
passes (Strategy pattern) to minimize depth while preserving the circuit's
unitary equivalence.

Architecture (Strategy pattern)
--------------------------------
Each optimization pass is a :class:`OptimizationPass` subclass implementing
:meth:`run(circuit) → LogicalCircuit`.  A :class:`PassManager` sequences
them.

Built-in passes
---------------
- :class:`CancellationPass`     – cancel adjacent inverse gate pairs (e.g. CNOT²=I).
- :class:`MergeRotationsPass`   – merge adjacent same-axis rotations Rz(a)·Rz(b)→Rz(a+b).
- :class:`CommutativityPass`    – swap commuting gates to expose cancellation opportunities.
- :class:`DeadCodeEliminationPass` – drop identity gates.

Transpilation
-------------
- :class:`TopologyMapper`       – maps logical qubits to physical, inserting SWAP gates
  for non-adjacent two-qubit operations on restricted topologies.

Public surface
--------------
- :class:`OptimizationPass`
- :class:`PassManager`
- :class:`CancellationPass`
- :class:`MergeRotationsPass`
- :class:`CommutativityPass`
- :class:`DeadCodeEliminationPass`
- :class:`TopologyMapper`
- :class:`HardwareTopology`
"""

from qsynth.compiler.optimization_passes import (
    OptimizationPass,
    PassManager,
    CancellationPass,
    MergeRotationsPass,
    CommutativityPass,
    DeadCodeEliminationPass,
)
from qsynth.compiler.topology import HardwareTopology, TopologyMapper

__all__ = [
    "OptimizationPass",
    "PassManager",
    "CancellationPass",
    "MergeRotationsPass",
    "CommutativityPass",
    "DeadCodeEliminationPass",
    "HardwareTopology",
    "TopologyMapper",
]
