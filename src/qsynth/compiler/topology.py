"""
qsynth.compiler.topology
=========================

Hardware topology abstraction and logical-to-physical qubit mapping.

Provides two example topologies:

1. **AllToAll**  – every pair of qubits is directly connected (ideal simulator).
2. **HeavyHex**  – IBM's heavy-hex lattice, where qubits are arranged in a
   hexagonal connectivity graph with degree ≤ 3.

The :class:`TopologyMapper` inserts SWAP gates to route two-qubit operations
across non-adjacent qubit pairs, then measures the resulting depth increase.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from qsynth.exceptions import TopologyError
from qsynth.synthesizer.gate_defs import CnotGate
from qsynth.synthesizer.logical_circuit import GateApplication, LogicalCircuit


# ── Hardware Topology ──────────────────────────────────────────────────────────


@dataclass
class HardwareTopology:
    """
    Undirected connectivity graph for a quantum processor.

    Attributes
    ----------
    n_physical_qubits : int
        Number of physical qubits on the device.
    edges : frozenset[frozenset[int]]
        Set of undirected edges (pairs of connected qubits).
    name  : str
        Human-readable topology name.
    """

    n_physical_qubits: int
    edges: frozenset[frozenset[int]]
    name: str = "custom"

    def are_adjacent(self, q1: int, q2: int) -> bool:
        """Return True if qubits q1 and q2 are directly connected."""
        return frozenset({q1, q2}) in self.edges

    def neighbors(self, q: int) -> list[int]:
        """Return all qubits adjacent to q."""
        return [
            other
            for edge in self.edges
            for other in edge
            if q in edge and other != q
        ]

    def shortest_path(self, src: int, dst: int) -> list[int]:
        """
        BFS shortest path from src to dst.

        Returns the path as a list of qubit indices [src, …, dst].
        Raises TopologyError if no path exists.
        """
        if src == dst:
            return [src]
        visited = {src}
        queue: list[list[int]] = [[src]]
        while queue:
            path = queue.pop(0)
            current = path[-1]
            for nb in self.neighbors(current):
                if nb == dst:
                    return path + [nb]
                if nb not in visited:
                    visited.add(nb)
                    queue.append(path + [nb])
        raise TopologyError(
            f"No path from qubit {src} to qubit {dst} "
            f"in topology '{self.name}'."
        )

    @classmethod
    def all_to_all(cls, n: int) -> "HardwareTopology":
        """Create a fully-connected (all-to-all) topology for n qubits."""
        edges = frozenset(
            frozenset({i, j}) for i in range(n) for j in range(i + 1, n)
        )
        return cls(
            n_physical_qubits=n,
            edges=edges,
            name=f"AllToAll-{n}",
        )

    @classmethod
    def linear(cls, n: int) -> "HardwareTopology":
        """Create a linear (chain) topology: 0−1−2−…−(n-1)."""
        edges = frozenset(frozenset({i, i + 1}) for i in range(n - 1))
        return cls(
            n_physical_qubits=n,
            edges=edges,
            name=f"Linear-{n}",
        )

    @classmethod
    def heavy_hex(cls, n: int) -> "HardwareTopology":
        """
        Approximate IBM heavy-hex topology for n qubits.

        In the heavy-hex lattice, qubits are arranged in a pattern where
        most qubits have degree 2 and "hub" qubits have degree 3.
        For n qubits, we construct a simplified heavy-hex graph:

            0 - 1 - 2 - 3
                |       |
                4   5 - 6
                |       |
                7 - 8 - 9

        For general n, we build a zigzag pattern with bridge connections.
        """
        edges: set[frozenset[int]] = set()
        row_size = max(2, math.isqrt(n))

        for i in range(n - 1):
            # Linear chain backbone
            edges.add(frozenset({i, i + 1}))

        # Add bridge connections (heavy-hex cross-links) every other row
        for i in range(0, n - row_size, row_size):
            bridge = i + row_size // 2
            if bridge < n and bridge + row_size < n:
                edges.add(frozenset({bridge, bridge + row_size}))

        return cls(
            n_physical_qubits=n,
            edges=frozenset(edges),
            name=f"HeavyHex-{n}",
        )

    def describe(self) -> str:
        lines = [
            f"HardwareTopology '{self.name}'",
            f"  Physical qubits : {self.n_physical_qubits}",
            f"  Edges           : {len(self.edges)}",
        ]
        edge_strs = sorted(
            f"({min(e)},{max(e)})" for e in self.edges
        )
        lines.append("  Connectivity    : " + "  ".join(edge_strs[:20]))
        if len(self.edges) > 20:
            lines.append(f"                    ... ({len(self.edges)-20} more)")
        return "\n".join(lines)


# ── SWAP gate helper ──────────────────────────────────────────────────────────


def _swap_gate_sequence(
    qubit_a: int, qubit_b: int
) -> list[GateApplication]:
    """
    Generate SWAP = CNOT(a,b) · CNOT(b,a) · CNOT(a,b) gate sequence.
    """
    return [
        GateApplication(gate=CnotGate(), qubits=(qubit_a, qubit_b), label="SWAP_1"),
        GateApplication(gate=CnotGate(), qubits=(qubit_b, qubit_a), label="SWAP_2"),
        GateApplication(gate=CnotGate(), qubits=(qubit_a, qubit_b), label="SWAP_3"),
    ]


# ── Topology Mapper ───────────────────────────────────────────────────────────


class TopologyMapper:
    """
    Maps logical qubits to physical qubits and routes non-adjacent
    two-qubit gates via SWAP insertion.

    Parameters
    ----------
    topology : HardwareTopology
    """

    def __init__(self, topology: HardwareTopology) -> None:
        self.topology = topology

    def map(self, circuit: LogicalCircuit) -> LogicalCircuit:
        """
        Transpile *circuit* to the given topology.

        Strategy:
        1. Assign logical qubits to physical qubits (trivial 1-to-1 for now).
        2. For each two-qubit gate on non-adjacent qubits, insert SWAP
           gates along the shortest path to bring qubits together.
        3. Track the evolving qubit permutation as SWAPs are inserted.

        Parameters
        ----------
        circuit : LogicalCircuit

        Returns
        -------
        LogicalCircuit
            Topology-constrained circuit with SWAP insertions.

        Raises
        ------
        TopologyError
            If n_qubits > n_physical_qubits, or no path exists.
        """
        n = circuit.n_qubits
        if n > self.topology.n_physical_qubits:
            raise TopologyError(
                f"Circuit has {n} qubits but topology only has "
                f"{self.topology.n_physical_qubits} physical qubits."
            )

        # Logical → physical qubit mapping (starts as identity).
        log_to_phys: list[int] = list(range(n))
        phys_to_log: list[int] = list(range(n))

        physical_ops: list[GateApplication] = []

        for op in circuit.operations:
            if op.gate.n_qubits == 1:
                phys_q = log_to_phys[op.qubits[0]]
                physical_ops.append(
                    GateApplication(gate=op.gate, qubits=(phys_q,), label=op.label)
                )
            elif op.gate.n_qubits == 2:
                log_c, log_t = op.qubits[0], op.qubits[1]
                phys_c = log_to_phys[log_c]
                phys_t = log_to_phys[log_t]

                if self.topology.are_adjacent(phys_c, phys_t):
                    physical_ops.append(
                        GateApplication(
                            gate=op.gate,
                            qubits=(phys_c, phys_t),
                            label=op.label,
                        )
                    )
                else:
                    # Route via SWAP insertions along shortest path.
                    path = self.topology.shortest_path(phys_c, phys_t)
                    # Move phys_c toward phys_t by swapping along path.
                    for step in range(len(path) - 2):
                        qa, qb = path[step], path[step + 1]
                        physical_ops.extend(_swap_gate_sequence(qa, qb))
                        # Update logical→physical mapping.
                        la = phys_to_log[qa]
                        lb = phys_to_log[qb]
                        log_to_phys[la], log_to_phys[lb] = qb, qa
                        phys_to_log[qa], phys_to_log[qb] = lb, la

                    # Now phys_c is adjacent to phys_t.
                    phys_c_new = log_to_phys[log_c]
                    phys_t_new = log_to_phys[log_t]
                    physical_ops.append(
                        GateApplication(
                            gate=op.gate,
                            qubits=(phys_c_new, phys_t_new),
                            label=op.label + "_routed",
                        )
                    )
            else:
                raise TopologyError(
                    f"TopologyMapper does not support {op.gate.n_qubits}-qubit gates."
                )

        result = LogicalCircuit(
            n_qubits=self.topology.n_physical_qubits,
            name=f"{circuit.name}@{self.topology.name}",
        )
        result.operations = physical_ops
        return result
